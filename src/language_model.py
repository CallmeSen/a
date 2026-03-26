"""LLM loading utilities for Multimodal Sentiment Analysis."""
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from .config import device, COMPUTE_DTYPE, LLM_MODEL_NAME, ASPECT_START, ASPECT_END, _set_special_token_ids


def build_tokenizer_and_llm():
    """Build InternLM2.5 tokenizer and full causal LM.

    Returns:
        tokenizer: AutoTokenizer instance
        llm_for_clm: AutoModelForCausalLM (full model with LM head)
        llm_base: InternLM2Model (base model without LM head)
    """
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
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

    # Load full causal LM (InternLM2ForCausalLM)
    llm_for_clm = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=COMPUTE_DTYPE,
        attn_implementation="eager",
    ).to(device).eval()

    # Resize embeddings to accommodate new special tokens
    llm_for_clm.resize_token_embeddings(len(tokenizer))
    print(f"[INFO] Tokenizer vocab size: {len(tokenizer)}, embedding resized")

    # Extract the base model (InternLM2Model) for the wrapper
    # InternLM2ForCausalLM.model = InternLM2Model
    llm_base = llm_for_clm.model
    num_layers = len(llm_base.layers)

    print(f"[INFO] LLM base: {llm_base.__class__.__name__}, "
          f"num_layers={num_layers}, hidden_size={llm_base.config.hidden_size}")

    return tokenizer, llm_for_clm, llm_base, num_layers


def build_tokenizer_only():
    """Build only the tokenizer (for inference without model)."""
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
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
