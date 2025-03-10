"""
Provides a way to apply LoRA to existing GPT-2 model
"""

import torch
from peft import LoraConfig, get_peft_model, TaskType

def apply_lora_to_gpt2(model, rank=8, alpha=16, dropout=0.1):
    """
    Apply LoRA to a GPT-2 model. 

    Inputs include model=the gpt-2 model we apply lora to, 
    rank=the rank of low-rank decomposition matrix, alpha=the scaling factor for lora, 
    dropout=dropout probablity for lora layers. 
    
    Ouput is our model with lora integrated.
    """
    num_layers = len(model.gpt_layers) if hasattr(model, 'gpt_layers') else 0
    
    #identify the gpt-2 layers that should use lora
    target_modules = []
    for i in range(num_layers): 
        target_modules.extend([
            f"gpt_layers.{i}.self_attention.query",
            f"gpt_layers.{i}.self_attention.key",
            f"gpt_layers.{i}.self_attention.value",
            f"gpt_layers.{i}.attention_dense",
            f"gpt_layers.{i}.interm_dense",
            f"gpt_layers.{i}.out_dense"
        ])
    
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    
    #how much of the model actually updates 
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    #how big total model is
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"LoRA applied to GPT-2 model.")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")
    print(f"Total parameters: {total_params:,}")
    
    return model

def count_parameters(model, only_trainable=True):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())