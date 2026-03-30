"""Multitask model: MultimodalSentimentModel + auxiliary aspect detection head.

R8 (FIXED): Removed soft gate and aspect detection auxiliary head.
The original soft gate (+1.0 additive bias for non-present aspects) created
gradient conflicts and catastrophic initial distortions (aspect_probs≈0.5 at init).
The aspect detection task was also redundant — aspect name is already in input text.

This module is kept for backward compatibility: it forwards to the base model
without any auxiliary modifications.

Usage:
    from src.multitask_model import MultitaskSentimentModel
    from src import MultimodalSentimentModel

    base_model = MultimodalSentimentModel(...)
    multitask_model = MultitaskSentimentModel(base_model)
    outputs = model(pixel_values, input_ids, attention_mask)
    # outputs["logits"] — sentiment [B, 1, 1, 4]
    # outputs["bad_batch"] — bool
"""
from typing import Optional

from .multimodal_sentiment_model import MultimodalSentimentModel


class MultitaskSentimentModel(MultimodalSentimentModel):
    """
    Backward-compatible wrapper — no longer adds soft gate or aspect detection.

    The original auxiliary aspect detection head was removed because:
    1. Soft gate distorted all logits from epoch 0 (aspect_probs≈0.5 at init)
    2. Aspect name is already in input text, making the task redundant
    3. Competing gradients between auxiliary and main task degraded performance
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
        self.load_state_dict(base_model.state_dict(), strict=False)

        self.vision_encoder = base_model.vision_encoder
        self.projector = base_model.projector
        self.perceiver_resampler = base_model.perceiver_resampler
        self.llm_wrapper = base_model.llm_wrapper
        self.tokenizer = base_model.tokenizer

        trainable = [n for n, p in self.named_parameters() if p.requires_grad]
        print(f"[Multitask] Trainable params ({len(trainable)}): "
              f"LoRA + adapters + vision + projector + classifier")

    def forward(
        self,
        pixel_values,
        input_ids,
        attention_mask=None,
        image_counts=None,
    ) -> dict:
        """Forward — identical to base MultimodalSentimentModel.forward()."""
        return super().forward(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_counts=image_counts,
        )
