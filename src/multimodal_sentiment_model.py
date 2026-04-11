"""Diagram-compliant multimodal sentiment model.

Architecture (per diagram):
- Image: SwinV2 → MLP Projector → Perceiver Resampler / VCE Module
- Text: <ASP>aspect</ASP> comment... → Qwen3 backbone
- Cross-attention adapter: RMSNorm + gated residual (between self-attn and FFN)
- h_a extraction: mean pooling over <ASP>...</ASP> span
- z_a extraction: dot-product attention pooling (h_a · H^T)
- Classifier: Linear(D → 4)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    ASPECT_START_ID,
    ASPECT_END_ID,
    USE_LORA,
    USE_VCE,
    USE_CONTRASTIVE_LOSS,
    USE_ASPECT_ATTN,
)


class MultimodalSentimentModel(nn.Module):
    def __init__(
        self,
        vision_encoder,
        projector,
        perceiver_resampler,
        llm_wrapper,
        tokenizer,
        num_aspects: int = 6,
        num_classes: int = 4,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.projector = projector
        self.perceiver_resampler = perceiver_resampler
        self.llm_wrapper = llm_wrapper
        self.tokenizer = tokenizer
        self.num_aspects = num_aspects
        self.num_classes = num_classes
        self.use_vce = USE_VCE

        self.llm_hidden_size = llm_wrapper.hidden_size

        # VCE Module (thay the PerceiverResampler khi USE_VCE=1)
        if USE_VCE:
            from .vce_module import MultiScaleVisualFusion

            self.vce_fusion = MultiScaleVisualFusion(llm_dim=self.llm_hidden_size)
            print(f"[VCE] MultiScaleVisualFusion enabled (64 tokens from 4 SwinV2 stages)")

        # Freezing strategy:
        # - ALWAYS freeze Qwen backbone to avoid OOM on 44GB VRAM.
        # - USE_LORA=1: PEFT has already injected LoRA adapters (A,B matrices) into
        #   frozen Qwen Linear layers. Those LoRA params have requires_grad=True set by
        #   PEFT and are NOT part of qwen_base.parameters() directly — they live inside
        #   PeftModel's merged layer wrappers, so they are NOT frozen by this loop.
        # - USE_LORA=0: Only GatedCrossAttentionAdapter + classifier train.
        # - The detach() in QwenLMWrapper._make_attn_hook breaks autograd activation memory
        #   for the frozen backbone in BOTH modes, keeping VRAM low.
        self._freeze_llm_backbone = True
        for p in llm_wrapper.qwen_base.parameters():
            p.requires_grad = False
        # When LoRA is active, LoRA A/B matrices need gradients through backbone.
        # Tell the hook to NOT detach so gradients flow to LoRA params.
        if USE_LORA:
            llm_wrapper.set_allow_gradient(True)
        print("[MultimodalSentimentModel] Qwen backbone frozen — trainable params managed by PEFT (LoRA) or adapters")

        # Gradient safety hooks for trainable params: replace NaN/Inf grads with 0.
        # masked_fill_ is DDP-compatible because it modifies the grad in-place.
        for _, param in self.named_parameters():
            if param.requires_grad:
                param.register_hook(
                    lambda grad: grad.masked_fill_(
                        torch.isnan(grad) | torch.isinf(grad), 0.0
                    )
                )

        # Diagram: classifier head (trainable)
        self.classifier_head = nn.Linear(self.llm_hidden_size, num_classes)
        nn.init.xavier_uniform_(self.classifier_head.weight)
        nn.init.zeros_(self.classifier_head.bias)

        # ArcFace head for contrastive loss (USE_CONTRASTIVE_LOSS=1)
        if USE_CONTRASTIVE_LOSS:
            from .contrastive_loss import SupervisedAngularMarginLoss
            self.arcface_head = SupervisedAngularMarginLoss(
                embedding_dim=self.llm_hidden_size,
                num_classes=num_classes,
                scale=30.0,
                margin=0.5,
            )
            self.embedding_norm = nn.LayerNorm(self.llm_hidden_size)
            print(f"[Contrastive Loss] ArcFace head enabled")

        # Aspect-Guided Visual Attention (USE_ASPECT_ATTN=1)
        if USE_ASPECT_ATTN:
            from .aspect_guided_attention import AspectGuidedVisualAttention
            self.aspect_attention = AspectGuidedVisualAttention(
                hidden_size=self.llm_hidden_size,
                num_heads=8,
            )
            print(f"[Aspect Attention] Aspect-Guided Visual Attention enabled")

        # Disable dropout everywhere
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0

    def _has_nonfinite(self, x, name: str = "tensor") -> bool:
        if isinstance(x, dict):
            # Handle VCE multi-stage features dict
            for k, v in x.items():
                if not torch.isfinite(v).all():
                    print(f"[NaN-DETECT] {name}['{k}'] has NaN/Inf — batch will be skipped")
                    return True
            return False
        bad = not torch.isfinite(x).all()
        if bad:
            print(f"[NaN-DETECT] {name} has NaN/Inf — batch will be skipped")
        return bad

    def _bad_batch_output(self, batch_size: int, device: torch.device) -> dict:
        return {
            "logits": torch.zeros(batch_size, 1, 1, self.num_classes, device=device, dtype=torch.float32),
            "bad_batch": True,
        }

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

            if self.use_vce:
                # === VCE Path: Multi-scale SwinV2 features ===
                multi_stage = self.vision_encoder(pixel_flat, return_all_stages=True)
                if self._has_nonfinite(multi_stage.get("stage4", multi_stage.get("stage1", torch.tensor(0.0))), "vce_stage_features"):
                    return None, None, True

                visual_per_img = self.vce_fusion(multi_stage)  # [B*M, 64, llm_dim]
                if self._has_nonfinite(visual_per_img, "vce_output"):
                    return None, None, True
            else:
                # === Original Path: SwinV2 → MLPProjector → PerceiverResampler ===
                img_tokens = self.vision_encoder(pixel_flat)
                if self._has_nonfinite(img_tokens.float(), "vision_output"):
                    return None, None, True

                img_proj = self.projector(img_tokens)  # [B*M, P, 1024]
                if self._has_nonfinite(img_proj, "projector_output"):
                    return None, None, True

                visual_per_img = self.perceiver_resampler(img_proj)  # [B*M, K, 2560]
                if self._has_nonfinite(visual_per_img, "perceiver_output"):
                    return None, None, True

            K = visual_per_img.size(1)
            visual_tokens = visual_per_img.reshape(B, M * K, self.llm_hidden_size)

            if image_counts is None:
                counts = torch.full((B,), M, device=device, dtype=torch.long)
            else:
                counts = image_counts.to(device=device, dtype=torch.long).clamp(min=0, max=M)

            token_idx = torch.arange(M * K, device=device).unsqueeze(0)
            valid_lengths = (counts * K).unsqueeze(1)
            visual_mask = token_idx < valid_lengths

            visual_tokens = visual_tokens * visual_mask.unsqueeze(-1).to(visual_tokens.dtype)
            if self._has_nonfinite(visual_tokens, "visual_tokens"):
                return None, None, True

            return visual_tokens.to(dtype=torch.float32), visual_mask, False

        # Single-image path: [B, C, H, W]
        if self.use_vce:
            multi_stage = self.vision_encoder(pixel_values, return_all_stages=True)
            if self._has_nonfinite(multi_stage.get("stage4", multi_stage.get("stage1", torch.tensor(0.0))), "vce_stage_features_single"):
                return None, None, True

            visual_tokens = self.vce_fusion(multi_stage)  # [B, 64, llm_dim]
            if self._has_nonfinite(visual_tokens, "vce_output_single"):
                return None, None, True
        else:
            img_tokens = self.vision_encoder(pixel_values)
            if self._has_nonfinite(img_tokens.float(), "vision_output_single"):
                return None, None, True

            img_proj = self.projector(img_tokens)  # [B, P, 1024]
            if self._has_nonfinite(img_proj, "projector_output_single"):
                return None, None, True

            visual_tokens = self.perceiver_resampler(img_proj)  # [B, K, 2560]
            if self._has_nonfinite(visual_tokens, "perceiver_output_single"):
                return None, None, True

        visual_mask = torch.ones(
            batch_size, visual_tokens.size(1), device=device, dtype=torch.bool
        )

        return visual_tokens.to(dtype=torch.float32), visual_mask, False

    def _extract_z_a_vectorized(
        self,
        final_hidden: torch.Tensor,   # bf16 from Qwen forward
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        start_id: int,
        end_id: int,
    ) -> torch.Tensor:
        """
        Vectorized z_a extraction (R4).

        All computations done in bf16 to match Qwen activations (COMPUTE_DTYPE=bfloat16).
        Returns z_a in bf16, classifier_head casts to float32.
        """
        B, L, D = final_hidden.shape
        device = final_hidden.device
        # Cast to bf16 — Qwen forward outputs bf16, ensure all ops use it consistently.
        final_hidden = final_hidden.to(torch.bfloat16)
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.bfloat16)
        """
        Fully vectorized h_a extraction + dot-product attention pooling for the entire batch.

        Args:
            final_hidden: [B, L, D] — LLM output with visual info injected
            input_ids:    [B, L] — to locate <ASP>...</ASP> spans
            attention_mask: [B, L] or None
            start_id: int — token ID for <ASP>
            end_id:   int — token ID for </ASP>

        Returns:
            z_a: [B, D] — aspect-aware multimodal representation
        """
        B, L, D = final_hidden.shape
        device = final_hidden.device

        # 1) Locate span boundaries per sample
        asp_start_mask = (input_ids == start_id)    # [B, L]
        asp_end_mask   = (input_ids == end_id)      # [B, L]

        # 2) argmax per sample to find first <ASP> and first </ASP>
        start_ones = asp_start_mask.float()
        end_ones   = asp_end_mask.float()

        # Guard: if no <ASP> token found → use position 0
        start_pos = start_ones.argmax(dim=1)         # [B]
        end_pos   = end_ones.argmax(dim=1)           # [B]

        # Detect "all-zero" rows (no token found) and replace with L-1
        start_valid = asp_start_mask.any(dim=1)      # [B]
        end_valid   = asp_end_mask.any(dim=1)        # [B]
        start_pos = torch.where(start_valid, start_pos, torch.zeros_like(start_pos))
        end_pos   = torch.where(end_valid,   end_pos,   torch.full_like(end_pos, L - 1))

        # Clamp: ensure end >= start
        end_pos = torch.max(end_pos, start_pos)

        # 3) Mean-pool h_a over the span [start, end] per sample — vectorized
        span_len = (end_pos - start_pos + 1).float()  # [B]

        # Use index_select + cumulative-sum trick for per-sample mean pooling
        # Expand to [B, L, D] — element-wise multiply and sum
        arange_3d = torch.arange(L, device=device).unsqueeze(0).unsqueeze(-1)   # [1, L, 1]
        start_3d  = start_pos.unsqueeze(1).unsqueeze(-1)                         # [B, 1, 1]
        end_3d    = end_pos.unsqueeze(1).unsqueeze(-1)                           # [B, 1, 1]

        in_span = (arange_3d >= start_3d) & (arange_3d <= end_3d)               # [B, L, 1]
        span_len_3d = span_len.unsqueeze(-1).unsqueeze(-1)                       # [B, 1, 1]

        h_a_sum = (final_hidden * in_span.float()).sum(dim=1)                    # [B, D]
        h_a = h_a_sum / span_len_3d.squeeze(-1).clamp(min=1.0)               # [B, D]

        if not torch.isfinite(h_a).all():
            print("[NaN-DETECT] h_a vectorized — batch will be skipped")
            return torch.zeros(B, D, device=device, dtype=torch.bfloat16)

        # 4) Dot-product attention pooling: z_a = softmax(h_a · H^T / √D) · H
        # Cast everything to float32 for stable matmul — bf16 residual from Qwen
        # is preserved through the hook's (adapted_bf16, None) return path.
        H_T = final_hidden.to(torch.float32).transpose(1, 2)                  # [B, D, L] float32
        h_a_3d = h_a.unsqueeze(1)                                               # [B, 1, D] float32
        scores = torch.bmm(h_a_3d, H_T).squeeze(1) / (D ** 0.5)                # [B, L] float32

        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask.bool(), float("-inf"))

        attn_weights = F.softmax(scores, dim=1).unsqueeze(1)                  # [B, 1, L] float32
        z_a = torch.bmm(attn_weights, final_hidden.to(torch.float32)).squeeze(1)  # [B, D] float32

        return z_a

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_counts: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Vectorized diagram-compliant forward pass.

        Supports dual visual encoding paths:
        - USE_VCE=0: SwinV2 → MLPProjector → PerceiverResampler (original)
        - USE_VCE=1: SwinV2 (multi-stage) → VCEFusion (multi-scale fusion)

        GPU utilization improved vs original per-sample loop.
        """
        B = input_ids.size(0)
        device = input_ids.device

        # 1) Encode images ONCE (shared across aspects)
        visual_tokens, visual_mask, bad_batch = self._encode_images(
            pixel_values, image_counts=image_counts
        )
        if bad_batch:
            return self._bad_batch_output(B, device)

        # 2) Resolve special token IDs
        start_id = ASPECT_START_ID
        end_id = ASPECT_END_ID
        if start_id is None or end_id is None:
            from .config import ASPECT_START, ASPECT_END
            start_id = self.tokenizer.convert_tokens_to_ids(ASPECT_START)
            end_id = self.tokenizer.convert_tokens_to_ids(ASPECT_END)

        # 3) InternLM forward — SINGLE batched call (R4: was B per-sample calls)
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)

        final_hidden, _ = self.llm_wrapper(
            text_input_ids=input_ids,
            text_position_ids=position_ids,
            attention_mask=attention_mask,
            visual_tokens=visual_tokens,
            visual_mask=visual_mask,
            output_hidden_states=False,
        )

        if self._has_nonfinite(final_hidden, "final_hidden_vectorized"):
            return self._bad_batch_output(B, device)

        # 4) Extract h_a for aspect-guided attention (if enabled)
        if USE_ASPECT_ATTN and hasattr(self, "aspect_attention"):
            # Compute h_a (aspect span mean-pool)
            asp_start_mask = (input_ids == start_id)
            asp_end_mask = (input_ids == end_id)
            start_ones = asp_start_mask.float()
            end_ones = asp_end_mask.float()
            start_pos = start_ones.argmax(dim=1)
            end_pos = end_ones.argmax(dim=1)
            start_valid = asp_start_mask.any(dim=1)
            end_valid = asp_end_mask.any(dim=1)
            start_pos = torch.where(start_valid, start_pos, torch.zeros_like(start_pos))
            end_pos = torch.where(end_valid, end_pos, torch.full_like(end_pos, seq_len - 1))
            end_pos = torch.max(end_pos, start_pos)
            span_len = (end_pos - start_pos + 1).float().clamp(min=1.0)
            arange_3d = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(-1)
            start_3d = start_pos.unsqueeze(1).unsqueeze(-1)
            end_3d = end_pos.unsqueeze(1).unsqueeze(-1)
            in_span = (arange_3d >= start_3d) & (arange_3d <= end_3d)
            span_len_3d = span_len.unsqueeze(-1).unsqueeze(-1)
            h_a_sum = (final_hidden * in_span.float()).sum(dim=1)
            h_a = h_a_sum / span_len_3d.squeeze(-1).clamp(min=1.0)
            if not torch.isfinite(h_a).all():
                return self._bad_batch_output(B, device)

            # Aspect-guided visual refinement
            visual_tokens, _ = self.aspect_attention(
                aspect_hidden=h_a.to(torch.float32),
                visual_tokens=visual_tokens,
                visual_mask=visual_mask,
            )

            # Second LLM forward with refined visual tokens
            final_hidden, _ = self.llm_wrapper(
                text_input_ids=input_ids,
                text_position_ids=position_ids,
                attention_mask=attention_mask,
                visual_tokens=visual_tokens,
                visual_mask=visual_mask,
                output_hidden_states=False,
            )
            if self._has_nonfinite(final_hidden, "final_hidden_aspect_refined"):
                return self._bad_batch_output(B, device)

        # 5) Vectorized h_a extraction + z_a (R4: replaces per-sample loop)
        z_a = self._extract_z_a_vectorized(
            final_hidden, input_ids, attention_mask, start_id, end_id
        )

        if self._has_nonfinite(z_a, "z_a_vectorized"):
            return self._bad_batch_output(B, device)

        # 6) Classifier — batched
        # Cast to float32 to match classifier_head weight dtype (float32).
        # BF16 activations from Qwen forward are preserved in z_a.
        logits = self.classifier_head(z_a.to(torch.float32))  # [B, 4]
        logits = logits.unsqueeze(1).unsqueeze(1)               # [B, 1, 1, 4] for loss compat

        # Return embeddings for contrastive loss (USE_CONTRASTIVE_LOSS=1)
        if USE_CONTRASTIVE_LOSS:
            z_a_norm = F.normalize(self.embedding_norm(z_a), p=2, dim=1)
            return {"logits": logits, "embeddings": z_a_norm, "bad_batch": False}

        return {"logits": logits, "bad_batch": False}

    def get_trainable_params(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
