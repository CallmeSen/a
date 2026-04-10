"""Qwen3-4B-Instruct loading utilities for Multimodal Sentiment Analysis."""
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import (
    device,
    COMPUTE_DTYPE,
    LLM_MODEL_NAME,
    ASPECT_START,
    ASPECT_END,
    _set_special_token_ids,
    Qwen3_4B_OFFLOAD_LM_HEAD,
)


def build_tokenizer_and_llm():
    """Build Qwen3-4B-Instruct tokenizer and full causal LM.

    Returns:
        tokenizer: AutoTokenizer instance
        llm_for_clm: AutoModelForCausalLM (full model with LM head)
        llm_base: Qwen2Model (base model without LM head)
        num_layers: int
        hidden_size: int (read dynamically from model config)
    """
    model_name = LLM_MODEL_NAME  # Qwen/Qwen3-4B-Instruct

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "right"

    # Register aspect special tokens
    tokenizer.add_special_tokens({"additional_special_tokens": [ASPECT_START, ASPECT_END]})
    aspect_start_id = tokenizer.convert_tokens_to_ids(ASPECT_START)
    aspect_end_id = tokenizer.convert_tokens_to_ids(ASPECT_END)
    _set_special_token_ids(aspect_start_id, aspect_end_id)
    print(f"[INFO] Special tokens registered: {ASPECT_START}={aspect_start_id}, {ASPECT_END}={aspect_end_id}")

    # Load full causal LM (Qwen2ForCausalLM)
    if Qwen3_4B_OFFLOAD_LM_HEAD:
        # device_map="auto" puts everything on GPU, then we explicitly
        # offload lm_head (unused in our task anyway) to CPU to free VRAM.
        llm_for_clm = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=COMPUTE_DTYPE,
            attn_implementation="eager",
            device_map="auto",
        ).eval()
        # Move lm_head (the language-model projection) to CPU — saves ~300-500 MB VRAM
        if hasattr(llm_for_clm, "lm_head") and next(llm_for_clm.parameters(), None) is not None:
            try:
                llm_for_clm.lm_head = llm_for_clm.lm_head.to("cpu")
                print("[INFO] Qwen3-4B: lm_head offloaded to CPU (saves ~400 MB VRAM)")
            except Exception:
                pass  # Already on CPU or tied weights
    else:
        llm_for_clm = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=COMPUTE_DTYPE,
            attn_implementation="eager",
        ).to(device).eval()

    # Resize embeddings to accommodate new special tokens
    llm_for_clm.resize_token_embeddings(len(tokenizer))
    print(f"[INFO] Tokenizer vocab size: {len(tokenizer)}, embedding resized")

    # Extract the base model (Qwen2Model) for the wrapper
    llm_base = llm_for_clm.model
    num_layers = len(llm_base.layers)
    hidden_size = llm_base.config.hidden_size

    print(f"[INFO] LLM base: {llm_base.__class__.__name__}, "
          f"num_layers={num_layers}, hidden_size={hidden_size}")

    return tokenizer, llm_for_clm, llm_base, num_layers, hidden_size


def build_tokenizer_only():
    """Build only the tokenizer (for inference without model)."""
    model_name = LLM_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "right"

    # Register aspect special tokens
    tokenizer.add_special_tokens({"additional_special_tokens": [ASPECT_START, ASPECT_END]})
    aspect_start_id = tokenizer.convert_tokens_to_ids(ASPECT_START)
    aspect_end_id = tokenizer.convert_tokens_to_ids(ASPECT_END)
    _set_special_token_ids(aspect_start_id, aspect_end_id)
    return tokenizer
