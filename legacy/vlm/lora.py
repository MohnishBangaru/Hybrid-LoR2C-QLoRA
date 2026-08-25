import torch
import time
import re
import os
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import get_peft_model, LoraConfig
from torch.amp import autocast, GradScaler
import wandb

# === Configuration Setup ===
config = {
    "epochs": 3,
    "batch_size": 1,
    "lr": 1e-4,
    "dataset": "none-yet/anime-captions",
    "model_id": "HuggingFaceTB/SmolVLM-500M-Instruct",
    "samples": 100,
    "max_length": 512,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "gradient_clipping": 1.0,
    "save_path": "lora_SmolVLM-500M-Instruct_adapter",
    "log_interval": 10,
    "image_size": 512,
}

# Initialize wandb
wandb.init(
    project="LoR2C-QLoRA",
    config=config,
    name="smolvlm-lora-run"
)

print("Configuration:")
for k, v in config.items():
    print(f"  {k}: {v}")

# === Load dataset ===
print(f"\nLoading dataset: {config['dataset']} (first {config['samples']} samples)")
ds = load_dataset(config['dataset'], split=f"train[:{config['samples']}]")

print("\nFirst 5 raw captions:")
for i in range(5):
    print(i, repr(ds[i]["text"]))

# === Load processor & model ===
print(f"\nLoading processor and model: {config['model_id']}")
processor = AutoProcessor.from_pretrained(
    config['model_id'],
    size={"longest_edge": config['image_size']}
)
model = AutoModelForVision2Seq.from_pretrained(
    config['model_id'],
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# === Apply LoRA ===
print("\nApplying LoRA adapter...")
lora_config = LoraConfig(
    r=config["lora_r"],
    lora_alpha=config["lora_alpha"],
    target_modules=["q_proj", "v_proj"],
    lora_dropout=config["lora_dropout"],
    bias="none"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.train()

# === Dataset class ===
class SmolVLMDataset(Dataset):
    def __init__(self, hf_ds, processor, max_length):
        self.ds = hf_ds
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item["image"].convert("RGB")
        # resize to fit one 512x512 patch
        max_size = config["image_size"]
        if max(image.width, image.height) > max_size:
            ratio = max_size / max(image.width, image.height)
            image = image.resize(
                (int(image.width * ratio), int(image.height * ratio)),
                Image.Resampling.LANCZOS
            )

        # strip numeric prefixes
        raw = item["text"]
        caption_text = re.sub(r'^\s*\d+(\.\d+)*\s*[:\-–]?\s*', '', raw)

        # build prompt
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image."}
            ]},
            {"role": "assistant", "content": caption_text}
        ]
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        )

        enc = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}

        # build labels
        ids = enc["input_ids"].clone()
        decoded = self.processor.tokenizer.decode(ids, skip_special_tokens=False)
        start = decoded.find("Assistant:")
        if start != -1:
            part = decoded[: start + len("Assistant:")]
            toks = self.processor.tokenizer.encode(part, add_special_tokens=False)
            ids[: len(toks)] = -100
        else:
            ids[: len(ids)//2] = -100
        ids[ids == self.processor.tokenizer.pad_token_id] = -100
        enc["labels"] = ids

        return enc

# === Dataloader ===
train_ds = SmolVLMDataset(ds, processor, config["max_length"])
train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)

# === Optimizer & scaler ===
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=config["lr"]
)
scaler = GradScaler()

# === Training loop with GPU stats ===
global_step = 0
print("\nStarting training...")
for epoch in range(config["epochs"]):
    model.train()
    epoch_loss = 0.0
    epoch_start = time.time()
    torch.cuda.reset_peak_memory_stats(model.device)

    for step, batch in enumerate(train_loader):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        optimizer.zero_grad()
        torch.cuda.synchronize()

        t0 = time.time()
        with autocast(device_type=model.device.type, dtype=torch.bfloat16):
            outputs = model(**batch)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if config["gradient_clipping"] > 0:
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                config["gradient_clipping"]
            )
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        t1 = time.time()

        global_step += 1
        epoch_loss += loss.item()

        mem_cur = torch.cuda.memory_allocated(model.device) // 1024**2
        mem_peak = torch.cuda.max_memory_allocated(model.device) // 1024**2
        step_ms = (t1 - t0) * 1000

        # Log step metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "step": global_step,
            "loss": loss.item(),
            "step_time_ms": step_ms,
            "mem_cur_MB": mem_cur,
            "mem_peak_MB": mem_peak,
        })

        if step % config["log_interval"] == 0 or step == len(train_loader)-1:
            print(
                f"[E{epoch+1}/{config['epochs']} S{step}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f} | Time: {step_ms:.1f}ms | "
                f"Mem: {mem_cur}MB (peak {mem_peak}MB)"
            )

    epoch_dur = time.time() - epoch_start
    avg_loss = epoch_loss / len(train_loader)
    peak = torch.cuda.max_memory_allocated(model.device) // 1024**2
    print("-"*60)
    print(
        f"Epoch {epoch+1} done: avg loss {avg_loss:.4f} | "
        f"time {epoch_dur:.1f}s | peak mem {peak}MB"
    )
    print("-"*60)

    # Log epoch metrics to wandb
    wandb.log({
        "epoch_avg_loss": avg_loss,
        "epoch_time_s": epoch_dur,
        "epoch_peak_mem_MB": peak,
    })

# === Save adapter ===
os.makedirs(config["save_path"], exist_ok=True)
print(f"\nSaving LoRA adapter to {config['save_path']}...")
model.save_pretrained(config["save_path"])

# Log model artifact to wandb
artifact = wandb.Artifact('lora_adapter', type='model')
artifact.add_dir(config["save_path"])
wandb.log_artifact(artifact)

print("Done.")

# === Validation split ===
print("\nLoading validation split (last 10% of samples)...")
val_ds = load_dataset(config['dataset'], split=f"train[-{max(1, config['samples']//10)}:]")
val_data = SmolVLMDataset(val_ds, processor, config["max_length"])
val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

# === Evaluation: BLEU and CIDEr ===
try:
    from nltk.translate.bleu_score import corpus_bleu
except ImportError:
    print("[INFO] nltk not found. Please install with: pip install nltk")
    corpus_bleu = None
try:
    from pycocoevalcap.cider.cider import Cider
except ImportError:
    print("[INFO] pycocoevalcap not found. Please install with: pip install git+https://github.com/salaniz/pycocoevalcap")
    Cider = None

def generate_caption(batch):
    model.eval()
    with torch.no_grad():
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        pixel_values = batch["pixel_values"].to(model.device)
        with autocast(device_type=model.device.type, dtype=torch.bfloat16):
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

if corpus_bleu is not None:
    bleu = corpus_bleu(gt_captions, pred_captions)
    print(f"BLEU score: {bleu:.4f}")
    wandb.log({"val_BLEU": bleu})
else:
    print("BLEU score: [nltk not installed]")

if Cider is not None:
    gts = {i: [" ".join(gt[0])] for i, gt in enumerate(gt_captions)}
    res = {i: [" ".join(pred_captions[i])] for i in range(len(pred_captions))}
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts, res)
    print(f"CIDEr score: {cider_score:.4f}")
    wandb.log({"val_CIDEr": cider_score})
else:
    print("CIDEr score: [pycocoevalcap not installed]")

# Finish wandb run
wandb.finish()