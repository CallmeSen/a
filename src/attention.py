"""
Retrieval and fusion modules for the ACSA aspect-loop architecture.

Components:
  - AspectQuery: learnable embedding table [NUM_ASPECTS, D_h]
  - TextRetriever: cross-attention from aspect query to text hidden states
  - ImageRetriever: cross-attention from aspect query to image summaries
  - RoiRetriever: cross-attention from aspect query to ROI features
  - GatedFusion: learnable gates combining text, image, ROI evidence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from src.config import (
    LLM_HIDDEN,
    NUM_ASPECTS,
    TEXT_RETRIEVER_HEADS,
    IMG_RETRIEVER_HEADS,
    ROI_RETRIEVER_HEADS,
    FUSION_GATE_HIDDEN_MULT,
)


class AspectQuery(nn.Module):
    """
    Trainable embedding table for the 6 aspect queries.
    Shape: [NUM_ASPECTS, D_h]
    """

    def __init__(self, d_h: int = LLM_HIDDEN, num_aspects: int = NUM_ASPECTS):
        super().__init__()
        self.embed = nn.Embedding(num_aspects, d_h)
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, aspect_idx: int) -> torch.Tensor:
        """Return embedding for aspect_idx. Shape: [D_h]"""
        return self.embed(aspect_idx)

    def all_queries(self) -> torch.Tensor:
        """Return all aspect queries. Shape: [NUM_ASPECTS, D_h]"""
        return self.embed.weight


class _CrossAttention(nn.Module):
    """
    Generic cross-attention: query (aspect vector) attends over a sequence of keys/values.
    Handles mask internally.
    """

    def __init__(self, d_h: int, num_heads: int):
        super().__init__()
        assert d_h % num_heads == 0, f"d_h={d_h} not divisible by num_heads={num_heads}"
        self.d_h = d_h
        self.num_heads = num_heads
        self.head_dim = d_h // num_heads

        self.q_proj = nn.Linear(d_h, d_h)
        self.k_proj = nn.Linear(d_h, d_h)
        self.v_proj = nn.Linear(d_h, d_h)
        self.o_proj = nn.Linear(d_h, d_h)
        self.norm = nn.LayerNorm(d_h)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: query vector(s). Shape: [B, D_h] or [B, H, D_h]
            k: key sequence. Shape: [B, L, D_h]
            v: value sequence. Shape: [B, L, D_h]
            mask: attention mask. Shape: [B, L] — 1=valid, 0=padding

        Returns:
            out: attended vector. Shape: [B, D_h]
            attn_weights: average attention weights over heads. Shape: [B, L]
        """
        B = k.size(0)

        # Expand q if needed: [B, D_h] -> [B, H, head_dim]
        if q.dim() == 2:
            q = q.unsqueeze(1)  # [B, 1, D_h]
        # Work with [B, H, 1, head_dim]
        H = self.num_heads
        D = self.head_dim
        q = q.view(B, H, 1, D).transpose(1, 2)  # [B, 1, H, D] -> [B, H, 1, D]  (for broadcast)
        # Actually: [B, H, 1, D] is what we want to broadcast over L
        # q: [B, H, 1, D]
        q = q.transpose(1, 2)  # [B, 1, H, D]

        k = k.view(B, -1, H, D).transpose(1, 2)  # [B, H, L, D]
        v = v.view(B, -1, H, D).transpose(1, 2)  # [B, H, L, D]

        # Attention scores: [B, H, 1, L]
        scale = D ** -0.5
        attn = (q * scale) @ k.transpose(-2, -1)  # [B, H, 1, L]

        # Expand mask: [B, L] -> [B, 1, 1, L]
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, L]
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn, dim=-1)  # [B, H, 1, L]
        out = attn_weights @ v  # [B, H, 1, D]

        # Average over heads and squeeze
        out = out.transpose(1, 2).contiguous().view(B, 1, H * D).squeeze(1)  # [B, D_h]

        # Project and residual-norm
        out = self.o_proj(out)  # [B, D_h]
        out = self.norm(out)

        # Average attn weights over heads for reporting: [B, L]
        attn_weights_avg = attn_weights.squeeze(2).mean(dim=1)  # [B, L]

        return out, attn_weights_avg


class TextRetriever(nn.Module):
    """
    Aspect-guided retrieval from text hidden states.
    Uses cross-attention from aspect query to token-level text features.
    """

    def __init__(
        self,
        d_h: int = LLM_HIDDEN,
        num_heads: int = TEXT_RETRIEVER_HEADS,
    ):
        super().__init__()
        self.attn = _CrossAttention(d_h, num_heads)

    def forward(
        self,
        q: torch.Tensor,
        H_txt: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            q: aspect query. Shape: [B, D_h]
            H_txt: text hidden states. Shape: [B, L_txt, D_h]
            text_mask: 1=valid, 0=pad. Shape: [B, L_txt]

        Returns:
            h_txt_a: retrieved text evidence. Shape: [B, D_h]
        """
        h, _ = self.attn(q, H_txt, H_txt, text_mask)
        return h


class ImageRetriever(nn.Module):
    """
    Aspect-guided retrieval from image summaries (one vector per image).
    Returns weighted image representation and per-image relevance weights.
    """

    def __init__(
        self,
        d_h: int = LLM_HIDDEN,
        num_heads: int = IMG_RETRIEVER_HEADS,
    ):
        super().__init__()
        self.attn = _CrossAttention(d_h, num_heads)

    def forward(
        self,
        q: torch.Tensor,
        G_img: torch.Tensor,
        img_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: aspect query. Shape: [B, D_h]
            G_img: image summaries. Shape: [B, M, D_h]
            img_mask: 1=valid, 0=pad. Shape: [B, M]

        Returns:
            h_img_a: weighted image evidence. Shape: [B, D_h]
            w_img: per-image relevance weights. Shape: [B, M]
        """
        B, M, D = G_img.shape
        h, attn_weights = self.attn(q, G_img, G_img, img_mask)

        # Apply mask to attn weights before softmax (attn already masked in _CrossAttention)
        # attn_weights shape: [B, M]
        # Zero out padded positions
        if img_mask is not None:
            attn_weights = attn_weights.masked_fill(img_mask == 0, 0.0)

        # Normalize to sum to 1 over valid images
        denom = attn_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        w_img = attn_weights / denom  # [B, M]

        return h, w_img


class RoiRetriever(nn.Module):
    """
    Aspect-guided retrieval from ROI features.
    Returns weighted ROI representation and per-ROI relevance weights.
    """

    def __init__(
        self,
        d_h: int = LLM_HIDDEN,
        num_heads: int = ROI_RETRIEVER_HEADS,
    ):
        super().__init__()
        self.attn = _CrossAttention(d_h, num_heads)

    def forward(
        self,
        q: torch.Tensor,
        R_roi: torch.Tensor,
        roi_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: aspect query. Shape: [B, D_h]
            R_roi: ROI features. Shape: [B, R_total, D_h]
            roi_mask: 1=valid, 0=pad. Shape: [B, R_total]

        Returns:
            h_roi_a: weighted ROI evidence. Shape: [B, D_h]
            w_roi: per-ROI relevance weights. Shape: [B, R_total]
        """
        h, attn_weights = self.attn(q, R_roi, R_roi, roi_mask)

        if roi_mask is not None:
            attn_weights = attn_weights.masked_fill(roi_mask == 0, 0.0)

        denom = attn_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        w_roi = attn_weights / denom  # [B, R_total]

        return h, w_roi


class GatedFusion(nn.Module):
    """
    Learnable gated fusion combining text, image, and ROI evidence.
    Each modality has an independent gate; outputs a single fused representation.
    """

    def __init__(self, d_h: int = LLM_HIDDEN, mult: int = FUSION_GATE_HIDDEN_MULT):
        super().__init__()
        self.d_h = d_h
        hidden = d_h * mult

        self.gate_txt = nn.Linear(hidden, d_h)
        self.gate_img = nn.Linear(hidden, d_h)
        self.gate_roi = nn.Linear(hidden, d_h)

        self.gate_bias = nn.Parameter(torch.zeros(d_h))

        self.fuse_norm = nn.LayerNorm(d_h)

    def forward(
        self,
        h_txt: torch.Tensor,
        h_img: torch.Tensor,
        h_roi: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h_txt: text evidence. Shape: [B, D_h]
            h_img: image evidence. Shape: [B, D_h]
            h_roi: ROI evidence. Shape: [B, D_h]

        Returns:
            h_fuse: fused representation. Shape: [B, D_h]
        """
        concat = torch.cat([h_txt, h_img, h_roi], dim=-1)  # [B, 3*D_h]

        # Three independent gates, each operating on all 3 inputs
        g_txt = torch.sigmoid(self.gate_txt(concat) + self.gate_bias)
        g_img = torch.sigmoid(self.gate_img(concat) + self.gate_bias)
        g_roi = torch.sigmoid(self.gate_roi(concat) + self.gate_bias)

        # Weighted sum
        h_fuse = g_txt * h_txt + g_img * h_img + g_roi * h_roi

        # Residual connection with norm
        return self.fuse_norm(h_txt + h_fuse)
