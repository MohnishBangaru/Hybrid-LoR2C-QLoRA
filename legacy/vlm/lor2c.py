import torch
import torch.nn as nn
import time
import re
import os
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForVision2Seq, get_linear_schedule_with_warmup
from torch.amp import autocast, GradScaler
import wandb

# === Configuration ===
config = {
    "epochs":       10,
    "batch_size":   4,
    "lr":           2e-4,
    "dataset":      "none-yet/anime-captions",
    "model_id":     "HuggingFaceTB/SmolVLM-500M-Instruct",
    "samples":      100,
    "max_length":   512,
    "image_size":   512,
    "lor2c_r":      16,
    "lor2c_alpha":  32,
    "lor2c_dropout":0.1,
    "grad_clip":    1.0,
    "log_interval": 10,
    "save_path":    "lor2c_SmolVLM_adapter_fp16"
}

# Initialize wandb
wandb.init(
    project="LoR2C-QLoRA",
    config=config,
    name="smolvlm-lor2c-run"
)

# === 1. Load data + model (FP16) ===
ds = load_dataset(config["dataset"], split=f"train[:{config['samples']}]")
processor = AutoProcessor.from_pretrained(
    config["model_id"],
    size={"longest_edge": config["image_size"]}
)
model = AutoModelForVision2Seq.from_pretrained(
    config["model_id"],
    torch_dtype=torch.float16,
    device_map="auto"
)

# Freeze base model
model.requires_grad_(False)

# === 2. LoR2C adapter setup ===
class LoR2CConfig:
    def __init__(self, r2c_r, r2c_alpha, target_modules, dropout=0.0):
        self.r2c_r = r2c_r
        self.r2c_alpha = r2c_alpha
        self.scaling = r2c_alpha / r2c_r
        self.target_modules = target_modules
        self.dropout = dropout

class LoR2CAdapter(nn.Module):
    def __init__(self, orig: nn.Linear, cfg: LoR2CConfig):
        super().__init__()
        self.orig = orig
        self.scaling = cfg.scaling
        device = orig.weight.device
        self.down    = nn.Linear(orig.in_features, cfg.r2c_r, bias=False).to(device)
        self.up      = nn.Linear(cfg.r2c_r, orig.out_features, bias=False).to(device)
        self.dropout = nn.Dropout(cfg.dropout).to(device)
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.orig(x) + self.dropout(self.up(self.down(x))) * self.scaling

def inject_lor2c(model: nn.Module, cfg: LoR2CConfig):
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and any(name.endswith(tm) for tm in cfg.target_modules):
            parent_path, _, attr = name.rpartition('.')
            parent = model if parent_path == '' else dict(model.named_modules())[parent_path]
            setattr(parent, attr, LoR2CAdapter(module, cfg))

lor2c_cfg = LoR2CConfig(
    r2c_r=config["lor2c_r"],
    r2c_alpha=config["lor2c_alpha"],
    target_modules=["q_proj", "v_proj"],
    dropout=config["lor2c_dropout"]
)
inject_lor2c(model, lor2c_cfg)

# === 3. Dataset + DataLoader ===
class SmolVLMDataset(Dataset):
    def __init__(self, ds, proc, max_len):
        self.ds = ds
        self.proc = proc
        self.max_len = max_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item["image"].convert("RGB")
        mx = config["image_size"]
        if max(img.width, img.height) > mx:
            r = mx / max(img.width, img.height)
            img = img.resize((int(img.width*r), int(img.height*r)), Image.Resampling.LANCZOS)
        txt = re.sub(r'^\s*\d+(\.\d+)*\s*[:\-–]?\s*', '', item["text"])
        msgs = [
            {"role":"user","content":[{"type":"image"},{"type":"text","text":"Describe this image."}]},
            {"role":"assistant","content":txt}
        ]
        prompt = self.proc.apply_chat_template(msgs, add_generation_prompt=False, tokenize=False)
        enc = self.proc(text=prompt, images=img, return_tensors="pt",
                        padding="max_length", truncation=True, max_length=self.max_len)
        enc = {k:v.squeeze(0) for k,v in enc.items()}

        labels = enc["input_ids"].clone()
        dec = self.proc.tokenizer.decode(enc["input_ids"], skip_special_tokens=False)
        pos = dec.find("Assistant:")
        if pos >= 0:
            cut = pos + len("Assistant:")
            ptoks = self.proc.tokenizer.encode(dec[:cut], add_special_tokens=False)
            labels[:len(ptoks)] = -100
        else:
            labels[:len(labels)//2] = -100
        labels[labels == self.proc.tokenizer.pad_token_id] = -100
        enc["labels"] = labels
        return enc

train_ds = SmolVLMDataset(ds, processor, config["max_length"])
train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)

# === 4. Training Loop ===
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=config["lr"]
)
# Add learning rate scheduler
num_training_steps = config["epochs"] * len(train_loader)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=max(1, int(0.1 * num_training_steps)),
    num_training_steps=num_training_steps
)
scaler = GradScaler()

print("Starting LoR2C training (FP16)...")
global_step = 0
best_bleu = 0.0
best_epoch = 0
no_improve_epochs = 0
for epoch in range(config["epochs"]):
    model.train()
    torch.cuda.reset_peak_memory_stats(model.device)
    epoch_loss = 0.0
    t0_epoch = time.time()

    for step, batch in enumerate(train_loader):
        batch = {k:v.to(model.device) for k,v in batch.items()}
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t0 = time.time()

        with autocast(device_type=model.device.type, dtype=torch.float16):
            out = model(**batch)
            loss = out.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()), 
            config["grad_clip"]
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()  # Step the scheduler
        torch.cuda.synchronize()
        t1 = time.time()

        epoch_loss += loss.item()
        cur = torch.cuda.memory_allocated(model.device)//1024**2
        peak = torch.cuda.max_memory_allocated(model.device)//1024**2
        dt = (t1 - t0) * 1000
        global_step += 1

        # Log step metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "step": global_step,
            "loss": loss.item(),
            "step_time_ms": dt,
            "mem_cur_MB": cur,
            "mem_peak_MB": peak,
            "lr": scheduler.get_last_lr()[0],
        })

        if step % config["log_interval"] == 0 or step == len(train_loader)-1:
            print(f"[E{epoch+1}/{config['epochs']} S{step}/{len(train_loader)}] "
                  f"Loss {loss.item():.4f} | {dt:.1f} ms | Mem {cur} MB (peak {peak} MB) | LR {scheduler.get_last_lr()[0]:.2e}")

    avg_loss = epoch_loss/len(train_loader)
    epoch_time = time.time()-t0_epoch
    peak_mem = torch.cuda.max_memory_allocated(model.device)//1024**2
    print(f"--> Epoch {epoch+1} done: avg loss {avg_loss:.4f}, "
          f"time {epoch_time:.1f}s, "
          f"peak mem {peak_mem} MB")

    # Log epoch metrics to wandb
    wandb.log({
        "epoch_avg_loss": avg_loss,
        "epoch_time_s": epoch_time,
        "epoch_peak_mem_MB": peak_mem,
    })

    # === Validation split ===
    print("\nLoading validation split (last 10% of samples)...")
    val_ds = load_dataset(config['dataset'], split=f"train[-{max(1, config['samples']//10)}:]")
    val_data = SmolVLMDataset(val_ds, processor, config["max_length"])
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

    # === Evaluation: BLEU only (with smoothing) ===
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    except ImportError:
        print("[INFO] nltk not found. Please install with: pip install nltk")
        corpus_bleu = None
        SmoothingFunction = None

    def generate_caption(batch):
        model.eval()
        with torch.no_grad():
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            pixel_values = batch["pixel_values"].to(model.device)
            with autocast(device_type=model.device.type, dtype=torch.float16):
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    max_new_tokens=64,
                    num_beams=3,
                )
            out = processor.tokenizer.decode(generated[0], skip_special_tokens=True)
            return out.strip()

    gt_captions = []
    pred_captions = []
    print("\nEvaluating on validation set...")
    for batch in val_loader:
        for k in batch:
            batch[k] = batch[k].to(model.device)
        gt = processor.tokenizer.decode(
            batch["labels"][0][batch["labels"][0] != -100],
            skip_special_tokens=True
        )
        pred = generate_caption(batch)
        print("GT:", repr(gt))
        print("PRED:", repr(pred))
        gt_captions.append([gt.split()])
        pred_captions.append(pred.split())

    bleu = 0.0
    if corpus_bleu is not None and SmoothingFunction is not None:
        smoothie = SmoothingFunction().method4
        bleu = corpus_bleu(gt_captions, pred_captions, smoothing_function=smoothie)
        print(f"BLEU score (smoothed): {bleu:.4f}")
        wandb.log({"val_BLEU": bleu})
    else:
        print("BLEU score: [nltk not installed]")

    # Early stopping based on BLEU
    if bleu > best_bleu:
        best_bleu = bleu
        best_epoch = epoch + 1
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
    if no_improve_epochs >= 3:
        print(f"Early stopping at epoch {epoch+1} (no BLEU improvement for 3 epochs)")
        break

print(f"Best validation BLEU: {best_bleu:.4f} at epoch {best_epoch}")
wandb.log({"best_val_BLEU": best_bleu, "best_val_BLEU_epoch": best_epoch})

# === 5. Save Adapter ===
os.makedirs(config["save_path"], exist_ok=True)
model.save_pretrained(config["save_path"])
print("LoR2C adapter saved to", config["save_path"])

# Log model artifact to wandb
artifact = wandb.Artifact('lor2c_adapter', type='model')
artifact.add_dir(config["save_path"])
wandb.log_artifact(artifact)

# Finish wandb run
wandb.finish()