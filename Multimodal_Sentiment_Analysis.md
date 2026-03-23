```
                                   ┌───────────────────────────────┐
                                   │        INPUT (BATCH)          │
                                   │  - images: [B, M, 3, H, W]    │
                                   │  - text: input_ids, attn_mask │
                                   └───────────────┬───────────────┘
                                                   │
                     ┌─────────────────────────────┴─────────────────────────────┐
                     │                                                           │
                     v                                                           v
        ┌──────────────────────────────┐                          ┌──────────────────────────────┐
        │         IMAGE BRANCH         │                          │          TEXT BRANCH         │
        │   (SwinV2 + Projector path)  │                          │   (Tokenizer + Embeddings)   │
        └───────────────┬──────────────┘                          └───────────────┬──────────────┘
                        │                                                         │
                        v                                                         v
        ┌──────────────────────────────┐                          ┌──────────────────────────────┐
        │VisionEncoder (SwinV2, frozen)│                          │ text_embeddings from LLM     │
        │-> image tokens [B, M, P, V]  │                          │ -> text_tokens [B, T, H]     │
        └───────────────┬──────────────┘                          └───────────────┬──────────────┘
                        │                                                         │
                        v                                                         │
        ┌──────────────────────────────┐                                          │
        │ MLPProjector                 │                                          │
        │ [B, M, P, V] -> [B, M, P, H] │                                          │
        └───────────────┬──────────────┘                                          │
                        │                                                         │
                        v                                                         │
        ┌──────────────────────────────────────────────────────────────────────────┐
        │ Text-guided Top-K Image Selection                                        │
        │ - summary(text_tokens)                                                   │
        │ - cosine with per-image summary                                          │
        │ - select K images                                                        │
        │ => img_tokens_selected: [B, K*P, H], img_attn: [B, K*P]                  │
        └───────────────┬──────────────────────────────────────────────────────────┘
                        │
                        v
        ┌──────────────────────────────────────────────────────────────────────────┐
        │ Cross-Modal Attention (before LLM)                                       │
        │ query = text_tokens, key/value = img_tokens_selected                     │
        │ fused_text = LayerNorm(Attn(text<-img) + text_tokens)                    │
        └───────────────┬──────────────────────────────────────────────────────────┘
                        │
                        v
        ┌──────────────────────────────────────────────────────────────────────────┐
        │ Multimodal Sequence Construction                                         │
        │ combined_embeds = concat([img_tokens_selected, fused_text], dim=1)       │
        │ full_attention_mask = concat([img_attn, text_attn], dim=1)               │
        └───────────────┬──────────────────────────────────────────────────────────┘
                        │
                        v
        ┌──────────────────────────────────────────────────────────────────────────┐
        │ LLM (InternLM2.5-1.8B + LoRA)                                            │
        │ outputs hidden_states[-1] = [B, L, H]                                    │
        └───────────────┬──────────────────────────────────────────────────────────┘
                        │
                        v
        ┌──────────────────────────────────────────────────────────────────────────┐
        │ Aspect-guided Branch                                                     │
        │ - aspect_token_embeddings: [A, H]                                        │
        │ - aspect queries attend to hidden_states (MultiheadAttention)            │
        │ - fusion + pooling => aspect_repr [B, A, H]                              │
        └───────────────┬──────────────────────────────────────────────────────────┘
                        │
             ┌──────────┴──────────┐
             │                     │
             v                     v
┌──────────────────────────┐   ┌──────────────────────────┐
│ ACD Head (Linear H->1)   │   │ ASC Head (Linear H->3)   │
│ acd_logits [B, A]        │   │ asc_logits [B, A, 3]     │
└──────────────┬───────────┘   └──────────────┬───────────┘
               │                              │
               └──────────────┬───────────────┘
                              v
            ┌──────────────────────────────────────┐
            │ Final 4-class per aspect prediction  │
            │ pred_presence = (acd_logits > 0)     │
            │ pred_sent = argmax(asc_logits) + 1   │
            │ final = pred_presence * pred_sent    │
            │ classes: 0=None,1=Neg,2=Neu,3=Pos    │
            └──────────────────────────────────────┘
```