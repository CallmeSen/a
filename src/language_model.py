from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import device, COMPUTE_DTYPE, LLM_MODEL_NAME


def build_tokenizer_and_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "right"

    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=COMPUTE_DTYPE,
        attn_implementation="eager",
    ).to(device).eval()

    return tokenizer, llm_model
