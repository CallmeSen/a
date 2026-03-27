"""Multitask model: MultimodalSentimentModel + auxiliary aspect detection head.

R8: Jointly predict aspect presence (binary) + sentiment (4-class).
Architecture:
  - Reuse all components from MultimodalSentimentModel (vision, projector, wrapper)
  - Override forward() to capture z_a before classifier
  - Add aspect_head(z_a) → binary aspect detection
  - Soft gate: sentiment logits adjusted by aspect_probs

Usage:
    from src.multitask_model import MultitaskSentimentModel
    from src import MultimodalSentimentModel

    base_model = MultimodalSentimentModel(...)
    multitask_model = MultitaskSentimentModel(base_model)
    outputs = model(pixel_values, input_ids, attention_mask)
    # outputs["logits"]         — sentiment [B, 1, 1, 4]
    # outputs["aspect_logits"]  — aspect detection [B, 1]
    # outputs["aspect_probs"]    — sigmoid [B, 1]
    # outputs["bad_batch"]       — bool
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .multimodal_sentiment_model import MultimodalSentimentModel
from .config import ASPECT_START_ID, ASPECT_END_ID


class MultitaskSentimentModel(MultimodalSentimentModel):
    """
    Auxiliary task: jointly predict aspect presence (binary) + sentiment (4-class).

    Backward compatible: can still be used as single-task model (aspect outputs ignored).
    """

    def __init__(self, base_model: MultimodalSentimentModel):
        super().__init__(
            vision_encoder=base_model.vision_encoder,
            projector=base_model.projector,
            perceiver_resampler=base_model.perceiver_resampler,
            llm_wrapper=base_model.llm_wrapper,
            tokenizer=base_model.tokenizer,
            num_aspects=base_model.num_aspects,
            num_classes=base_model.num_classes,
        )
        # Load base model weights so we start from pretrained checkpoint
        self.load_state_dict(base_model.state_dict(), strict=False)

        # Re-attach components from base (they may have been re-assigned in super().__init__)
        self.vision_encoder = base_model.vision_encoder
        self.projector = base_model.projector
        self.perceiver_resampler = base_model.perceiver_resampler
        self.llm_wrapper = base_model.llm_wrapper
        self.tokenizer = base_model.tokenizer

        # R8: Binary aspect detection head
        self.aspect_head = nn.Linear(self.llm_hidden_size, 1)
        nn.init.xavier_uniform_(self.aspect_head.weight)
        nn.init.zeros_(self.aspect_head.bias)

        # Freeze base model (MultitaskSentimentModel only trains aspect_head)
        # Only the new aspect_head trains; rest stays from pretrained checkpoint
        for name, param in self.named_parameters():
            if not name.startswith("aspect_head"):
                param.requires_grad = False

        print("[Multitask] MultitaskSentimentModel: only aspect_head is trainable")

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_counts: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        R8 Multitask forward: sentiment (4-class) + aspect detection (binary).

        Overrides MultimodalSentimentModel.forward() to:
          1. Capture z_a before classifier
          2. Add auxiliary aspect_head(z_a) → aspect_logits
          3. Soft-gate sentiment logits with aspect_probs
        """
        B = input_ids.size(0)
        device = input_ids.device

        # 1) Encode images ONCE
        visual_tokens, visual_mask, bad_batch = self._encode_images(
            pixel_values, image_counts=image_counts
        )
        if bad_batch:
            return {
                "logits": torch.zeros(B, 1, 1, self.num_classes, device=device, dtype=torch.float32),
                "aspect_logits": torch.zeros(B, 1, device=device, dtype=torch.float32),
                "aspect_probs": torch.zeros(B, 1, device=device, dtype=torch.float32),
                "bad_batch": True,
            }

        # 2) Resolve special token IDs
        start_id = ASPECT_START_ID
        end_id = ASPECT_END_ID
        if start_id is None or end_id is None:
            from .config import ASPECT_START, ASPECT_END
            start_id = self.tokenizer.convert_tokens_to_ids(ASPECT_START)
            end_id = self.tokenizer.convert_tokens_to_ids(ASPECT_END)

        # 3) InternLM forward — SINGLE batched call
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

        if self._has_nonfinite(final_hidden, "final_hidden_multitask"):
            return self._bad_batch_output_multitask(B, device)

        # 4) Vectorized z_a extraction
        z_a = self._extract_z_a_vectorized(
            final_hidden, input_ids, attention_mask, start_id, end_id
        )
        if self._has_nonfinite(z_a, "z_a_multitask"):
            return self._bad_batch_output_multitask(B, device)

        # 5) R8: Aspect detection head (auxiliary task)
        aspect_logits = self.aspect_head(z_a)               # [B, 1]
        aspect_probs = torch.sigmoid(aspect_logits)          # [B, 1]

        # 6) Sentiment classifier (reuse parent's head)
        sentiment_logits = self.classifier_head(z_a)       # [B, 4]

        # 7) R8: Soft gate — adjust sentiment by aspect presence
        # aspect NOT mentioned → boost "None" class, suppress others
        # aspect mentioned     → use predicted sentiment
        sentiment_logits = sentiment_logits.clone()
        sentiment_logits[:, 0] = sentiment_logits[:, 0] + (1 - aspect_probs.squeeze(-1)) * 10
        sentiment_logits[:, 1:] = sentiment_logits[:, 1:] * aspect_probs

        logits = sentiment_logits.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 4]

        return {
            "logits": logits,
            "aspect_logits": aspect_logits,
            "aspect_probs": aspect_probs,
            "z_a": z_a,
            "bad_batch": False,
        }

    def _bad_batch_output_multitask(self, batch_size: int, device: torch.device) -> dict:
        return {
            "logits": torch.zeros(batch_size, 1, 1, self.num_classes, device=device, dtype=torch.float32),
            "aspect_logits": torch.zeros(batch_size, 1, device=device, dtype=torch.float32),
            "aspect_probs": torch.zeros(batch_size, 1, device=device, dtype=torch.float32),
            "z_a": torch.zeros(batch_size, self.llm_hidden_size, device=device, dtype=torch.float32),
            "bad_batch": True,
        }
