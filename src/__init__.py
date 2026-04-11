"""Multimodal sentiment training package."""

from .vit_transformer import VisionEncoder
from .projector_layer import MLPProjector
from .perceiver_resampler import PerceiverResampler
from .vce_module import MultiScaleVisualFusion
from .dual_adapter import DualGatedCrossAttentionAdapter
from .contrastive_loss import SupervisedAngularMarginLoss, CombinedSentimentLoss
from .aspect_guided_attention import AspectGuidedVisualAttention
from .qwen_wrapper import QwenLMWrapper
from .multimodal_sentiment_model import MultimodalSentimentModel
from .llm_factory import build_tokenizer_and_llm, build_tokenizer_only
from .data import (
    SentimentDataset,
    make_collate_fn,
    load_all_splits,
    build_transform,
    build_train_transform,
    compute_class_weights,
    build_weighted_sampler,
)
from .training import LazyLambdaScheduler, train_epoch, validate, setup_optimizer
from .inference import predict_aspect_sentiment

__all__ = [
    # Core model components
    "VisionEncoder",
    "MLPProjector",
    "PerceiverResampler",
    "MultiScaleVisualFusion",
    "DualGatedCrossAttentionAdapter",
    "AspectGuidedVisualAttention",
    "QwenLMWrapper",
    "MultimodalSentimentModel",
    "build_tokenizer_and_llm",
    "build_tokenizer_only",
    # Contrastive loss
    "SupervisedAngularMarginLoss",
    "CombinedSentimentLoss",
    # Data
    "SentimentDataset",
    "make_collate_fn",
    "load_all_splits",
    "build_transform",
    "build_train_transform",
    "compute_class_weights",
    "build_weighted_sampler",
    # Training
    "LazyLambdaScheduler",
    "train_epoch",
    "validate",
    "setup_optimizer",
    # Inference
    "predict_aspect_sentiment",
]
