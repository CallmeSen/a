"""Backward-compatible exports for legacy imports.

Prefer importing from:
- src.vit_transformer
- src.projector_layer
- src.language_model
- src.multimodal_sentiment_model
"""

from .vit_transformer import VisionEncoder
from .projector_layer import MLPProjector
from .multimodal_sentiment_model import MultimodalSentimentModel
from .language_model import build_tokenizer_and_llm

__all__ = [
    "VisionEncoder",
    "MLPProjector",
    "MultimodalSentimentModel",
    "build_tokenizer_and_llm",
]
