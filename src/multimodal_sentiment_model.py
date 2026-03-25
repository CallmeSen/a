"""Single-head 4-class multimodal sentiment model.

Stable version:
- Vision: SwinV2 frozen
- Projector: trainable
- PerceiverResampler: trainable
- InternLM base: frozen, called via normal forward()
- Visual adapters: applied AFTER InternLM hidden states in a stable wrapper
- Aspect pooling: aspect-guided attention over final hidden states

Outputs:
    {"logits": [B, 6, 4], "bad_batch": bool}
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class MultimodalSentimentModel(nn.Module):
    def __init__(
        self,
        vision_encoder,
        projector,
        perceiver_resampler,
        internlm_wrapper,
        tokenizer,
        num_aspects: int = 6,
        num_classes: int = 4,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.projector = projector
        self.perceiver_resampler = perceiver_resampler
        self.internlm_wrapper = internlm_wrapper
        self.tokenizer = tokenizer
        self.num_aspects = num_aspects
        self.num_classes = num_classes

        self.llm_hidden_size = internlm_wrapper.hidden_size
        self.num_visual_tokens = perceiver_resampler.num_queries

        # Freeze backbone parts
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        for p in self.internlm_wrapper.internlm_base.parameters():
            p.requires_grad = False

        # Aspect-guided pooling
        self.aspect_token_embeddings = nn.Embedding(num_aspects, self.llm_hidden_size)
        nn.init.normal_(self.aspect_token_embeddings.weight, std=0.02)

        self.aspect_attn = nn.MultiheadAttention(
            embed_dim=self.llm_hidden_size,
            num_heads=8,
            batch_first=True,
        )
        self.aspect_ln = nn.LayerNorm(self.llm_hidden_size)

        self.aspect_pool = nn.Sequential(
            nn.Linear(self.llm_hidden_size * 2, self.llm_hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.llm_hidden_size),
        )

        self.classifier_head = nn.Linear(self.llm_hidden_size, num_classes)
        nn.init.xavier_uniform_(self.classifier_head.weight)
        nn.init.zeros_(self.classifier_head.bias)

        # Disable dropout everywhere
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0

        # Gradient safety hooks
        for _, param in self.named_parameters():
            if param.requires_grad:
                param.register_hook(
                    lambda grad: torch.nan_to_num(grad, nan=0.0, posinf=1e4, neginf=-1e4)
                )

    def _has_nonfinite(self, x: torch.Tensor, name: str) -> bool:
        bad = not torch.isfinite(x).all()
        if bad:
            print(f"[NaN-DETECT] {name} has NaN/Inf — batch will be skipped")
        return bad

    def _bad_batch_output(self, batch_size: int, device: torch.device) -> dict:
        logits = torch.zeros(
            batch_size,
            self.num_aspects,
            self.num_classes,
            device=device,
            dtype=torch.float32,
        )
        return {"logits": logits, "bad_batch": True}

    def _encode_images(
        self,
        pixel_values: torch.Tensor,
        image_counts: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], bool]:
        """
        Returns:
            visual_tokens: [B, N_vis, H]
            visual_mask:   [B, N_vis] bool
            bad_batch:     bool
        """
        device = pixel_values.device
        batch_size = pixel_values.shape[0]

        if pixel_values.dim() == 5:
            # [B, M, C, H, W]
            B, M = pixel_values.shape[:2]
            pixel_flat = pixel_values.reshape(B * M, *pixel_values.shape[2:])

            with torch.no_grad():
                img_tokens = self.vision_encoder(pixel_flat)  # [B*M, P, 1024]
            if self._has_nonfinite(img_tokens.float(), "vision_output"):
                return None, None, True

            img_proj = self.projector(img_tokens)  # [B*M, P, 2048]
            if self._has_nonfinite(img_proj, "projector_output"):
                return None, None, True

            visual_per_img = self.perceiver_resampler(img_proj)  # [B*M, K, 2048]
            if self._has_nonfinite(visual_per_img, "perceiver_output"):
                return None, None, True

            K = visual_per_img.size(1)
            visual_tokens = visual_per_img.reshape(B, M * K, self.llm_hidden_size)

            if image_counts is None:
                counts = torch.full((B,), M, device=device, dtype=torch.long)
            else:
                counts = image_counts.to(device=device, dtype=torch.long).clamp(min=0, max=M)

            token_idx = torch.arange(M * K, device=device).unsqueeze(0)  # [1, M*K]
            valid_lengths = (counts * K).unsqueeze(1)                    # [B, 1]
            visual_mask = token_idx < valid_lengths                      # [B, M*K]

            # Zero out padded visual tokens for extra safety
            visual_tokens = visual_tokens * visual_mask.unsqueeze(-1).to(visual_tokens.dtype)

            if self._has_nonfinite(visual_tokens, "visual_tokens"):
                return None, None, True

            return visual_tokens.to(dtype=torch.float32), visual_mask, False

        # Single-image path: [B, C, H, W]
        with torch.no_grad():
            img_tokens = self.vision_encoder(pixel_values)  # [B, P, 1024]
        if self._has_nonfinite(img_tokens.float(), "vision_output_single"):
            return None, None, True

        img_proj = self.projector(img_tokens)  # [B, P, 2048]
        if self._has_nonfinite(img_proj, "projector_output_single"):
            return None, None, True

        visual_tokens = self.perceiver_resampler(img_proj)  # [B, K, 2048]
        if self._has_nonfinite(visual_tokens, "perceiver_output_single"):
            return None, None, True

        visual_mask = torch.ones(
            batch_size,
            visual_tokens.size(1),
            device=device,
            dtype=torch.bool,
        )

        return visual_tokens.to(dtype=torch.float32), visual_mask, False

    def _aspect_representation(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Aspect-guided attention pooling:
            q = learnable aspect embeddings
            K = final hidden states
            V = final hidden states
        """
        B = hidden_states.shape[0]

        aspect_ids = torch.arange(
            self.num_aspects,
            device=hidden_states.device,
        ).unsqueeze(0).expand(B, -1)

        q = self.aspect_token_embeddings(aspect_ids).to(dtype=hidden_states.dtype)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()

        attn_out, _ = self.aspect_attn(
            query=q,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        if self._has_nonfinite(attn_out, "aspect_attn_out"):
            return None

        z = self.aspect_ln(q + attn_out)
        pooled = self.aspect_pool(torch.cat([q, z], dim=-1))
        if self._has_nonfinite(pooled, "aspect_pooled"):
            return None

        return pooled

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_counts: Optional[torch.Tensor] = None,
    ) -> dict:
        B = input_ids.size(0)
        device = input_ids.device

        # 1) Visual tokens
        visual_tokens, visual_mask, bad_batch = self._encode_images(
            pixel_values,
            image_counts=image_counts,
        )
        if bad_batch:
            return self._bad_batch_output(B, device)

        # 2) Text positions
        seq_len = input_ids.size(1)
        position_ids = torch.arange(
            seq_len,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0).expand(B, -1)

        # 3) InternLM + stable post-LLM visual adapters
        final_hidden, _ = self.internlm_wrapper(
            text_input_ids=input_ids,
            text_position_ids=position_ids,
            attention_mask=attention_mask,
            visual_tokens=visual_tokens,
            visual_mask=visual_mask,
            output_hidden_states=False,
        )

        if self._has_nonfinite(final_hidden, "final_hidden"):
            return self._bad_batch_output(B, device)

        # 4) Aspect-guided pooling
        aspect_repr = self._aspect_representation(final_hidden, attention_mask)
        if aspect_repr is None:
            return self._bad_batch_output(B, device)

        # 5) Classifier
        logits = self.classifier_head(aspect_repr)
        if self._has_nonfinite(logits, "logits"):
            return self._bad_batch_output(B, device)

        return {"logits": logits, "bad_batch": False}

    def get_trainable_params(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable