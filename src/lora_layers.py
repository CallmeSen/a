"""LoRA integration for Multimodal Sentiment Analysis (Qwen2.5-7B-Instruct backbone).

Supports LoRA fine-tuning on the Qwen backbone. The GatedCrossAttentionAdapter
from QwenLMWrapper remains active alongside LoRA.

Usage:
    from src.lora_layers import apply_lora_to_llm, print_lora_summary
    llm_base = apply_lora_to_llm(llm_base, r=16, alpha=32)
"""
from typing import List

import torch
import torch.nn as nn
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)


# LoRA target modules for Qwen2.5 (q_proj, v_proj in self-attention)
QEN2_LORA_TARGETS = ["q_proj", "v_proj"]


def apply_lora_to_llm(
    llm_base,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
) -> nn.Module:
    """
    Apply LoRA to the Qwen backbone (hybrid: LoRA + GatedCrossAttentionAdapter).

    Args:
        llm_base: The Qwen base model (Qwen2Model)
        r: LoRA rank (higher = more params, better quality)
        alpha: LoRA alpha (scaling factor = alpha / r)
        dropout: LoRA dropout probability

    Returns:
        PeftModel wrapping llm_base with LoRA adapters
    """
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=QEN2_LORA_TARGETS,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    print(f"[LoRA] Applying LoRA: r={r}, alpha={alpha}, targets={QEN2_LORA_TARGETS}")
    before = sum(p.numel() for p in llm_base.parameters() if p.requires_grad)
    print(f"[LoRA] Trainable params before: {before:,}")

    llm_with_lora = get_peft_model(llm_base, lora_config)

    after_trainable = sum(p.numel() for p in llm_with_lora.parameters() if p.requires_grad)
    after_total = sum(p.numel() for p in llm_with_lora.parameters())
    print(f"[LoRA] Trainable params after:  {after_trainable:,} / {after_total:,} "
          f"({100 * after_trainable / after_total:.2f}%)")

    return llm_with_lora


def print_lora_summary(model: nn.Module):
    """Print LoRA adapter summary."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[LoRA Summary] Trainable: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()


def load_peft_checkpoint(model: nn.Module, checkpoint_path: str):
    """Load a LoRA checkpoint into a PeftModel."""
    from safetensors.torch import load_file
    state_dict = load_file(checkpoint_path)
    model.load_state_dict(state_dict, strict=False)
    print(f"[LoRA] Loaded checkpoint from {checkpoint_path}")
