"""Shared components for the LoR2C Llama training prototypes.

``prototype.py`` and ``quantization_prototype.py`` are thin drivers over these
helpers; they differ only in how the LoR2C adapter forward hook is implemented
(plain fp16 vs quantization-aware training) and in how the result is saved.

Requires the vendored ``peft-0.5.0`` fork (see archive.zip / README) which
provides ``MSLoraConfig`` and ``LlamaLoRALayer``.
"""

import json
import os
import os.path as osp
import re
import sys
from functools import partial
from typing import Union

import torch
import transformers
from datasets import load_dataset
from peft import (
    LoraConfig,
    MSLoraConfig,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer
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


_BANNER_KEYS = (
    "base_model", "data_path", "output_dir", "batch_size", "micro_batch_size",
    "num_epochs", "learning_rate", "cutoff_len", "val_set_size", "seed",
    "mode", "lora_r", "lor2c_r", "lora_n", "lora_alpha", "lor2c_alpha",
    "sfs_k", "lora_dropout", "lora_target_modules", "train_on_inputs",
    "add_eos_token", "group_by_length", "wandb_project", "wandb_run_name",
    "wandb_watch", "wandb_log_model", "resume_from_checkpoint",
    "prompt_template_name", "max_merge_count", "max_distribution_count",
)


def log_training_config(params):
    """Print the run configuration, on the main process only.

    ``params`` is the driver's parameter dict (``locals()`` at the top of
    ``train``); the banner order and labels are fixed here.
    """
    if int(os.environ.get("LOCAL_RANK", 0)) != 0:
        return
    lines = []
    for key in _BANNER_KEYS:
        value = params[key]
        if key == "resume_from_checkpoint":
            lines.append(f"resume_from_checkpoint: {value or False}")
        elif key == "prompt_template_name":
            lines.append(f"prompt template: {value}")
        else:
            lines.append(f"{key}: {value}")
    print("Training Alpaca-LoRA model with params:\n" + "\n".join(lines) + "\n")


def setup_ddp(gradient_accumulation_steps):
    """Resolve device map and DDP settings from the torchrun environment."""
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    return device_map, ddp, gradient_accumulation_steps


def setup_wandb_env(wandb_project, wandb_watch, wandb_log_model):
    """Propagate W&B settings to the environment; return whether W&B is enabled."""
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
    return use_wandb


def load_llama_tokenizer(base_model):
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference
    return tokenizer


def build_prompt_tokenizer(tokenizer, prompter, cutoff_len, train_on_inputs, add_eos_token):
    """Return the ``generate_and_tokenize_prompt`` mapper for dataset rows."""

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

    return generate_and_tokenize_prompt


def select_lor2c_dims(hidden_size):
    """LoR2C rank/alpha by model width: narrow models get a lighter adapter."""
    if hidden_size <= 2048:
        return 4, 8
    return 16, 32


def build_peft_config(mode, r, alpha, target_modules, dropout):
    if mode == "base":
        return LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
    if mode == "lor2c":
        return MSLoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
    raise ValueError(f"Unknown mode {mode!r}; expected 'base' or 'lor2c'.")


def build_lor2c_schedule(model, lora_layers, r, alpha, dropout):
    """Register one LoR2C adapter per decoder layer; return the layer schedule."""
    num_decoder_layers = sum(1 for m in model.modules() if isinstance(m, LlamaDecoderLayer))
    print(f"Number of decoder layers: {num_decoder_layers}")
    schedules = {}
    for i in range(num_decoder_layers):
        adapter_name = f"floor{i + 1}"
        lora_layers.update_layer(adapter_name, r, alpha, dropout)
        schedules[adapter_name] = {
            "start_idx": i,
            "end_idx": i,
            "lora_output": 0,
        }
    return schedules


def extract_layer_index(name):
    match = re.search(r'\d+', name)
    if match:
        return int(match.group())
    raise ValueError(f"Could not extract layer index from name {name}")


def register_decoder_hooks(model, lora_hook, lora_layers, schedule):
    """Attach ``lora_hook`` to every decoder layer, bound to that layer's name."""
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, LlamaDecoderLayer):
            print(f"name:{name} is LlamaDecoderLayer. The hook is attached successfully.")
            hook = module.register_forward_hook(
                partial(lora_hook, lora_layers=lora_layers, schedule=schedule, name=name)
            )
            hooks.append(hook)
    return hooks


def load_instruction_dataset(data_path):
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        return load_dataset("json", data_files=data_path)
    return load_dataset(data_path)


def load_adapter_checkpoint(model, resume_from_checkpoint):
    """Load full or adapter-only weights; return the (possibly cleared) resume flag."""
    if not resume_from_checkpoint:
        return resume_from_checkpoint
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
    return resume_from_checkpoint


def prepare_datasets(data, val_set_size, generate_and_tokenize_prompt):
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
    return train_data, val_data


def mark_model_parallel(model, ddp):
    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True


def build_trainer(model, tokenizer, train_data, val_data, *, micro_batch_size,
                  gradient_accumulation_steps, num_epochs, learning_rate,
                  val_set_size, output_dir, ddp, group_by_length, use_wandb,
                  wandb_run_name):
    return Trainer(
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


def patch_peft_state_dict(model):
    """Make ``state_dict`` return only the PEFT weights so saves stay adapter-sized."""
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))


def maybe_compile(model):
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    return model
