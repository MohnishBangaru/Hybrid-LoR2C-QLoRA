"""LoR2C fine-tuning prototype with quantization-aware training (QAT) of the adapters.

Same pipeline as ``prototype.py``, but the LoR2C A/B projections are wrapped in
``torch.nn.qat`` linear layers and converted to quantized modules after training.
Requires the vendored ``peft-0.5.0`` fork (see archive.zip / README).
"""

import json
import os
import os.path as osp
import re
import sys
from functools import partial
from typing import List, Optional, Union

import fire
import torch
import torch.nn.qat as nnqat
import transformers
from datasets import load_dataset
from peft import (
    LoraConfig,
    MSLoraConfig,
    LlamaLoRALayer,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, set_seed
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


class LlamaForCausalLMWithLoR2C(LlamaForCausalLM):
    """Llama causal LM that carries a shared LoR2C module alongside the base weights."""

    def __init__(self, config, llama_lora_layers):
        super().__init__(config)
        self.lor2c_module = llama_lora_layers


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()

def train(
    # model/data params
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v0.6",  # the only required argument
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    seed: int = 42,
    # lora hyperparams
    mode: str = "base",
    lora_r: int = 8,
    lor2c_r: int = 16,
    lora_n: int = 1,
    lora_alpha: int = 16,
    lor2c_alpha: int = 32,
    sfs_k: Optional[int] = None,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[List[str]] = None,
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: Optional[str] = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.

    max_merge_count: int = 0,
    max_distribution_count: int = 0,
):
    if lora_target_modules is None:
        lora_target_modules = ["q_proj", "v_proj"]
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"seed: {seed}\n"
            f"mode: {mode}\n"
            f"lora_r: {lora_r}\n"
            f"lor2c_r: {lor2c_r}\n"
            f"lora_n: {lora_n}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lor2c_alpha: {lor2c_alpha}\n"
            f"sfs_k: {sfs_k}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
            f"max_merge_count: {max_merge_count}\n"
            f"max_distribution_count: {max_distribution_count}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    set_seed(seed)

    

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    

    # Load pretrained model
    model = LlamaForCausalLMWithLoR2C.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map=device_map,
        use_safetensors=True,
        llama_lora_layers=None  
    )
    
    hidden_size = model.config.hidden_size
    
    if hidden_size <= 2048:  
        lor2c_r = 4
        lor2c_alpha = 8
    else:  
        lor2c_r = 16
        lor2c_alpha = 32
    
    Llama_Lora_Layers = LlamaLoRALayer(hidden_size, hidden_size)
    model.lor2c_module = Llama_Lora_Layers  
    
    if mode == "base":
        config = LoraConfig(
            r=lor2c_r,
            lora_alpha=lor2c_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
    elif mode == "lor2c":
        config = MSLoraConfig(
            r=lor2c_r,
            lora_alpha=lor2c_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        raise ValueError(f"Unknown mode {mode!r}; expected 'base' or 'lor2c'.")
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, config)
    llama_lora_schedules = {}
    num_decoder_layers = sum(1 for m in model.modules() if isinstance(m, LlamaDecoderLayer))

    print(f"Number of decoder layers: {num_decoder_layers}")
    lora_parallel_schedule = [
    (i, i, f"floor{i+1}") for i in range(num_decoder_layers)
    ]
    for start_idx, end_idx, adapter_name in lora_parallel_schedule:
        Llama_Lora_Layers.update_layer(adapter_name, lor2c_r, lor2c_alpha, lora_dropout)
        llama_lora_schedules[adapter_name] = {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "lora_output": 0,
        }

    def extract_layer_index(name):
        match = re.search(r'\d+', name)
        if match:
            return int(match.group())
        else:
            raise ValueError(f"Could not extract layer index from name {name}")

    qat_modules = {}

    # Log qconfig to confirm dtype
    qat_config = torch.quantization.get_default_qat_qconfig("fbgemm")
    print("[QAT CONFIG] Activation and Weight dtypes:")
    print("  Activation:", qat_config.activation)
    print("  Weight:", qat_config.weight)

    def lora_hook(module, input, output, lora_layers, schedule, name):
        name = extract_layer_index(name)
        current_device = "cuda:0"
        new_output = list(output)

        for adapter_name in schedule:
            start_layer = schedule[adapter_name]["start_idx"]
            end_layer = schedule[adapter_name]["end_idx"]

            if start_layer == int(name):
                if adapter_name in lora_layers.lora_A and lora_layers.r[adapter_name] > 0:
                    if lora_layers.lora_A[adapter_name].weight.device != current_device:
                        lora_layers.lora_A[adapter_name].to(current_device)
                        lora_layers.lora_B[adapter_name].to(current_device)

                    lora_input = lora_layers.lora_dropout[adapter_name](input[0]).to(current_device)

                    qat_key = f"{adapter_name}-{name}"
                    if qat_key not in qat_modules:
                        lora_layers.lora_A[adapter_name] = lora_layers.lora_A[adapter_name].to(current_device)
                        lora_layers.lora_A[adapter_name].qconfig = qat_config
                        lora_layers.lora_B[adapter_name] = lora_layers.lora_B[adapter_name].to(current_device)
                        lora_layers.lora_B[adapter_name].qconfig = qat_config
                        qat_modules[qat_key] = {
                            "A": nnqat.Linear.from_float(lora_layers.lora_A[adapter_name], qat_config).to(
                                current_device),
                            "B": nnqat.Linear.from_float(lora_layers.lora_B[adapter_name], qat_config).to(
                                current_device),
                        }
                        qat_modules[qat_key]["A"].train()
                        qat_modules[qat_key]["B"].train()

                    middle = qat_modules[qat_key]["A"](lora_input)
                    delta = qat_modules[qat_key]["B"](middle) * lora_layers.scaling[adapter_name]
                    schedule[adapter_name]["lora_output"] = delta

            if end_layer == int(name):
                lora_output = schedule[adapter_name]["lora_output"]
                lora_output = lora_output.to(output[0].dtype).to(output[0].device)
                new_output[0] += lora_output

        return tuple(new_output)
    # Register the hook function
    def register_hooks(model, lora_layers, schedule):
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, LlamaDecoderLayer):
                print(f"name:{name} is LlamaDecoderLayer. The hook is attached successfully.")
                hook = module.register_forward_hook(partial(lora_hook, lora_layers=lora_layers, schedule=schedule, name=name))
                hooks.append(hook)
        return hooks

    register_hooks(model, Llama_Lora_Layers, llama_lora_schedules)

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=False,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )
    )

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )
    model.config.use_cache = False
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # QAT conversion after training
    model.eval()
    print("Converting QAT modules to quantized form...")
    for k in qat_modules:
        qat_modules[k]["A"] = torch.quantization.convert(qat_modules[k]["A"].eval(), inplace=False)
        qat_modules[k]["B"] = torch.quantization.convert(qat_modules[k]["B"].eval(), inplace=False)

    for adapter_name in llama_lora_schedules:
        layer_id = llama_lora_schedules[adapter_name]["start_idx"]
        qat_key = f"{adapter_name}-{layer_id}"
        if qat_key in qat_modules:
            Llama_Lora_Layers.lora_A[adapter_name] = qat_modules[qat_key]["A"]
            Llama_Lora_Layers.lora_B[adapter_name] = qat_modules[qat_key]["B"]

    model.save_pretrained(output_dir + "-qat")
    tokenizer.save_pretrained(output_dir + "-qat")
    print("QAT quantized model saved.")


if __name__ == "__main__":
    fire.Fire(train)


