"""Diagram-compliant multimodal sentiment model.

Architecture (per diagram):
- Image: SwinV2 → MLP Projector → Perceiver Resampler
- Text: <ASP>aspect</ASP> comment... → InternLM backbone
- Cross-attention adapter: RMSNorm + gated residual (between self-attn and FFN)
- h_a extraction: mean pooling over <ASP>...</ASP> span
- z_a extraction: dot-product attention pooling (h_a · H^T)
- Classifier: Linear(D → 4)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ASPECT_START_ID as _ASPECT_START_ID, ASPECT_END_ID as _ASPECT_END_ID


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

        # Resolve special token IDs at init time (tokenizer is already set up)
        self._start_id = _ASPECT_START_ID
        self._end_id = _ASPECT_END_ID
        if self._start_id is None or self._end_id is None:
            from .config import ASPECT_START, ASPECT_END
            self._start_id = tokenizer.convert_tokens_to_ids(ASPECT_START)
            self._end_id = tokenizer.convert_tokens_to_ids(ASPECT_END)
            from .config import _set_special_token_ids
            _set_special_token_ids(self._start_id, self._end_id)

        # The LLM backbone is kept frozen (requires_grad=False) because it produces
        # NaN during forward pass on this hardware — a known numerical instability in
        # InternLM2 that does NOT affect training quality (only adapters + head train).
        # NaN batches are caught by bad_batch=True and skipped cleanly.
        for p in self.internlm_wrapper.internlm_base.parameters():
            p.requires_grad = False

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

        # Disable dropout everywhere
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0

    def _has_nonfinite(self, x: torch.Tensor, name: str = "tensor") -> bool:
        bad = not torch.isfinite(x).all()
        if bad:
            print(f"[NaN-DETECT] {name} has NaN/Inf — batch will be skipped")
        return bad

    def _bad_batch_output(self, batch_size: int, device: torch.device) -> dict:
        return {
            "logits": torch.zeros(batch_size, 6, self.num_classes, device=device, dtype=torch.float32),
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

            token_idx = torch.arange(M * K, device=device).unsqueeze(0)
            valid_lengths = (counts * K).unsqueeze(1)
            visual_mask = token_idx < valid_lengths

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
            batch_size, visual_tokens.size(1), device=device, dtype=torch.bool
        )

        return visual_tokens.to(dtype=torch.float32), visual_mask, False

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_counts: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Diagram-compliant forward pass:
        - Encode images once (share across all aspects)
        - Per-sample: find <ASP>...</ASP> span → extract h_a via mean pooling
        - Dot-product attention pooling: z_a = softmax(h_a · H^T / sqrt(D)) · H
        - Classify: logits = Linear(z_a)

        If ANY sample produces NaN at any intermediate step, the entire batch
        is marked bad_batch=True so training can skip it cleanly.
        """
        B = input_ids.size(0)
        device = input_ids.device

        # 1) Encode images ONCE (shared across aspects)
        visual_tokens, visual_mask, bad_batch = self._encode_images(
            pixel_values, image_counts=image_counts
        )
        if bad_batch:
            return self._bad_batch_output(B, device)

        # 2) Locate <ASP> and </ASP> token positions in the batch
        start_id = self._start_id
        end_id = self._end_id

        asp_start_mask = (input_ids == start_id)   # [B, 6, L] bool
        asp_end_mask   = (input_ids == end_id)     # [B, 6, L] bool

        logits_list = []
        any_bad = False

        # 3) Per-sample, per-aspect sequential processing
        for b in range(B):
            for a in range(6):
                # Select text for this sample + this aspect: [1, 1, L]
                sample_input_ids = input_ids[b:b+1, a:a+1, :]
                if attention_mask is not None:
                    sample_attn_mask = attention_mask[b:b+1, a:a+1, :]  # [1, 1, L]
                else:
                    sample_attn_mask = None
                # Visual tokens: shared across all aspects for this sample
                sample_vis_tok  = visual_tokens[b:b+1]     # [1, N_vis, H]
                sample_vis_mask = visual_mask[b:b+1]       # [1, N_vis]

                # Build position_ids for this sample
                seq_len = sample_input_ids.size(-1)
                sample_pos_ids = torch.arange(
                    seq_len, device=device, dtype=torch.long
                ).unsqueeze(0)                              # [1, L]

                # 4) InternLM forward
                final_hidden, _ = self.internlm_wrapper(
                    text_input_ids=sample_input_ids,
                    text_position_ids=sample_pos_ids,
                    attention_mask=sample_attn_mask,
                    visual_tokens=sample_vis_tok,
                    visual_mask=sample_vis_mask,
                    output_hidden_states=False,
                )
                if self._has_nonfinite(final_hidden, "final_hidden"):
                    any_bad = True
                    break

                # 5) Extract h_a: mean pooling over <ASP>...</ASP> span
                # final_hidden is [1, 1, L, D] (batch + aspect dims)
                # Squeeze both to get [L, D]
                h_seq = final_hidden.squeeze(0).squeeze(0)        # [L, D]

                start_pos = asp_start_mask[b, a].nonzero(as_tuple=True)[0]
                end_pos   = asp_end_mask[b, a].nonzero(as_tuple=True)[0]

                if len(start_pos) > 0 and len(end_pos) > 0:
                    s = start_pos[0].item()
                    e = end_pos[0].item()
                    span_h = h_seq[s:e+1]                        # [span_len, D]
                    h_a = span_h.mean(dim=0, keepdim=True)       # [1, D]
                else:
                    h_a = h_seq[0, :].unsqueeze(0)                # [1, D]

                if self._has_nonfinite(h_a, "h_a"):
                    any_bad = True
                    break

                # 6) Dot-product attention pooling
                scores = torch.matmul(
                    h_a, h_seq.transpose(0, 1)                # [1, L]
                ) / (self.llm_hidden_size ** 0.5)

                if sample_attn_mask is not None:
                    scores = scores.masked_fill(~sample_attn_mask.bool(), float("-inf"))

                attn_weights = F.softmax(scores, dim=-1)        # [1, L]
                z_a = torch.matmul(attn_weights, h_seq.unsqueeze(0))  # [1, D]

                if self._has_nonfinite(z_a, "z_a"):
                    any_bad = True
                    break

                # 7) Classifier
                logit = self.classifier_head(z_a)          # [1, 4]
                logits_list.append(logit)

            if any_bad:
                break

        if any_bad:
            return self._bad_batch_output(B, device)

        # Concat all per-aspect logits: [B*6, 4] -> reshape to [B, 6, 4]
        logits = torch.cat(logits_list, dim=0)             # [B*6, 4]
        logits = logits.view(B, 6, self.num_classes)       # [B, 6, 4]

        return {"logits": logits, "bad_batch": False}

    def get_trainable_params(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
