"""LoR2C fine-tuning prototype for Llama-family models on Alpaca-style data.

The shared pipeline lives in ``lor2c_common.py``; this driver contributes the
plain fp16 LoR2C forward hook. Requires the vendored ``peft-0.5.0`` fork
(see archive.zip / README) which provides ``MSLoraConfig`` and ``LlamaLoRALayer``.
"""

from typing import List, Optional

import fire
import torch
from peft import (
    LlamaLoRALayer,
    get_peft_model,
    prepare_model_for_int8_training,
)
from transformers import set_seed

from lor2c_common import (
    LlamaForCausalLMWithLoR2C,
    Prompter,
    build_lor2c_schedule,
    build_peft_config,
    build_prompt_tokenizer,
    build_trainer,
    extract_layer_index,
    load_adapter_checkpoint,
    load_instruction_dataset,
    load_llama_tokenizer,
    log_training_config,
    mark_model_parallel,
    maybe_compile,
    patch_peft_state_dict,
    prepare_datasets,
    register_decoder_hooks,
    select_lor2c_dims,
    setup_ddp,
    setup_wandb_env,
)


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
    log_training_config(dict(locals()))
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map, ddp, gradient_accumulation_steps = setup_ddp(gradient_accumulation_steps)
    use_wandb = setup_wandb_env(wandb_project, wandb_watch, wandb_log_model)
    set_seed(seed)

    tokenizer = load_llama_tokenizer(base_model)
    generate_and_tokenize_prompt = build_prompt_tokenizer(
        tokenizer, prompter, cutoff_len, train_on_inputs, add_eos_token
    )

    # Load pretrained model
    model = LlamaForCausalLMWithLoR2C.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map=device_map,
        use_safetensors=True,
        llama_lora_layers=None
    )

    hidden_size = model.config.hidden_size
    lor2c_r, lor2c_alpha = select_lor2c_dims(hidden_size)

    llama_lora_layers = LlamaLoRALayer(hidden_size, hidden_size)
    model.lor2c_module = llama_lora_layers

    config = build_peft_config(mode, lor2c_r, lor2c_alpha, lora_target_modules, lora_dropout)
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, config)
    llama_lora_schedules = build_lor2c_schedule(
        model, llama_lora_layers, lor2c_r, lor2c_alpha, lora_dropout
    )

    def lora_hook(module, input, output, lora_layers, schedule, name):
        # Get the name of the current layer
        name = extract_layer_index(name)
        current_device = "cuda:0"
        new_output = list(output)
        for adapter_name in schedule:
            # Check if the current layer is within the specified range
            start_layer = schedule[adapter_name]["start_idx"]
            end_layer = schedule[adapter_name]["end_idx"]
            if start_layer == int(name):
                if adapter_name in lora_layers.lora_A.keys() and lora_layers.r[adapter_name] > 0:
                    if lora_layers.lora_A[adapter_name].weight.device != current_device:
                        lora_layers.lora_A[adapter_name].to(current_device)
                        lora_layers.lora_B[adapter_name].to(current_device)
                    lora_input = lora_layers.lora_dropout[adapter_name](input[0])
                    lora_input = lora_input.to(current_device)
                    middle = lora_layers.lora_A[adapter_name](lora_input)
                    middle = middle.to(current_device)
                    delta = lora_layers.lora_B[adapter_name](middle) * lora_layers.scaling[adapter_name]
                    schedule[adapter_name]["lora_output"] = delta
            if end_layer == int(name):
                lora_output = schedule[adapter_name]["lora_output"]
                lora_output = lora_output.to(output[0].dtype)
                lora_output = lora_output.to(output[0].device)
                new_output[0] += lora_output  # Add the LoRA output to the original output
        return tuple(new_output)

    register_decoder_hooks(model, lora_hook, llama_lora_layers, llama_lora_schedules)

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    data = load_instruction_dataset(data_path)
    resume_from_checkpoint = load_adapter_checkpoint(model, resume_from_checkpoint)

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    train_data, val_data = prepare_datasets(data, val_set_size, generate_and_tokenize_prompt)
    mark_model_parallel(model, ddp)

    trainer = build_trainer(
        model, tokenizer, train_data, val_data,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        val_set_size=val_set_size,
        output_dir=output_dir,
        ddp=ddp,
        group_by_length=group_by_length,
        use_wandb=use_wandb,
        wandb_run_name=wandb_run_name,
    )
    model.config.use_cache = False

    patch_peft_state_dict(model)
    model = maybe_compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)
