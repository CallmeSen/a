"""Aspect-Guided Visual Attention Module.

Cross-Attention với <ASP>aspect</ASP> token làm Query để lọc visual tokens:
- Query: aspect_hidden (h_a, từ <ASP>...</ASP> span)
- Key/Value: visual_tokens (từ VCE/Perceiver)
- Output: aspect-aware visual tokens cho LLM

Vị trí trong pipeline:
    Visual tokens → [Aspect-Guided Attn] → Refined visual tokens → QwenLMWrapper

Ưu điểm:
- Lọc bỏ regions không liên quan đến aspect đang phân tích
- Tăng signal-to-noise ratio cho visual features
- Dynamic, aspect-dependent attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AspectGuidedVisualAttention(nn.Module):
    """
    Cross-Attention module dùng aspect embedding làm Query.

    Sử dụng hidden state từ <ASP>...</ASP> span (h_a) làm query
    để attend vào visual tokens, tạo ra aspect-specific visual representation.

    Args:
        hidden_size: LLM hidden dimension (2560 for Qwen3-4B)
        num_heads: Số attention heads (8)
        dropout: Dropout probability (0.1)
        gate_init: Initial value cho output gate (sigmoid(-0.5) ≈ 0.38)
    """

    def __init__(
        self,
        hidden_size: int = 2560,
        num_heads: int = 8,
        dropout: float = 0.1,
        gate_init: float = -0.5,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Project aspect query từ hidden state
        self.query_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Project visual tokens cho Key và Value
        self.kv_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)

        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Gated residual: kiểm soát mức độ aspect-guided refinement
        # sigmoid(-0.5) ≈ 0.38: ban đầu chỉ 38% refinement, tăng dần khi training
        self.refine_gate = nn.Parameter(torch.tensor(gate_init))

        # Layer norm cho ổn định
        self.norm = nn.LayerNorm(hidden_size)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.kv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        aspect_hidden: torch.Tensor,
        visual_tokens: torch.Tensor,
        visual_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            aspect_hidden: [B, D] - Aspect embedding (h_a, từ <ASP>...</ASP> span)
            visual_tokens: [B, N_vis, D] - Visual tokens từ VCE/Perceiver
            visual_mask: [B, N_vis] bool - Valid visual token mask

        Returns:
            refined_visual: [B, N_vis, D] - Aspect-guided refined visual tokens
            attention_weights: [B, num_heads, 1, N_vis] - Attention weights (for visualization)
        """
        B, N_vis, D = visual_tokens.shape

        # Project aspect hidden state thành query
        # aspect_hidden [B, D] → [B, 1, D]
        q = self.query_proj(aspect_hidden).unsqueeze(1)  # [B, 1, D]

        # Project visual tokens thành Key và Value
        # kv [B, N, 2*D] → k [B, N, D], v [B, N, D]
        kv = self.kv_proj(visual_tokens)
        k, v = kv.chunk(2, dim=-1)

        # Reshape cho multi-head attention
        # q: [B, 1, D] → [B, num_heads, 1, head_dim]
        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        # k, v: [B, N, D] → [B, num_heads, N, head_dim]
        k = k.view(B, N_vis, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N_vis, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, 1, N]

        # Apply visual mask
        if visual_mask is not None:
            invalid_mask = ~visual_mask.bool().unsqueeze(1).unsqueeze(1)  # [B, 1, 1, N]
            scores = scores.masked_fill(invalid_mask, float("-inf"))

        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)  # [B, H, 1, N]
        attn_weights = self.dropout(attn_weights)

        # Compute attention output
        # attn_weights [B, H, 1, N] × v [B, H, N, d] → [B, H, 1, d]
        context = torch.matmul(attn_weights, v)  # [B, H, 1, d]

        # Reshape: [B, H, 1, d] → [B, 1, D]
        context = context.transpose(1, 2).contiguous()
        context = context.view(B, 1, D)

        # Output projection
        refined = self.out_proj(context.squeeze(1))  # [B, D]

        # Gated residual: visual_tokens + gate × refined_aspect
        gate_value = torch.sigmoid(self.refine_gate)  # scalar
        refined_visual = visual_tokens + gate_value * refined.unsqueeze(1)  # [B, N, D]

        # Layer norm
        refined_visual = self.norm(refined_visual)

        # Return attention weights for analysis (squeeze head dimension)
        attn_weights_out = attn_weights.squeeze(2)  # [B, num_heads, N_vis]

        return refined_visual, attn_weights_out


class AspectGuidedCrossAttention(nn.Module):
    """
    Wrapper module cho Aspect-Guided Attention với extraction logic tích hợp.

    Kết hợp:
    1. Aspect token extraction từ text hidden states
    2. Aspect-Guided Visual Attention
    3. Optional refinement sau attention
    """

    def __init__(
        self,
        hidden_size: int = 2560,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = AspectGuidedVisualAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Projection cho aspect token extraction
        self.asp_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        text_hidden: torch.Tensor,  # Qwen final hidden state [B, L, D]
        aspect_positions: torch.Tensor,  # [B, 2] - start/end positions
        visual_tokens: torch.Tensor,
        visual_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            text_hidden: Qwen output [B, L, D]
            aspect_positions: [B, 2] - start/end token positions trong sequence
            visual_tokens: [B, N_vis, D]
            visual_mask: [B, N_vis]

        Returns:
            refined_visual: [B, N_vis, D]
            attn_weights: [B, num_heads, N_vis]
        """
        B, L, D = text_hidden.shape
        device = text_hidden.device

        # Extract aspect token (mean pool over <ASP>...</ASP> span)
        start_pos = aspect_positions[:, 0]  # [B]
        end_pos = aspect_positions[:, 1]    # [B]

        arange = torch.arange(L, device=device).unsqueeze(0)  # [1, L]
        start_exp = start_pos.unsqueeze(1)  # [B, 1]
        end_exp = end_pos.unsqueeze(1)      # [B, 1]

        in_span = (arange >= start_exp) & (arange <= end_exp)  # [B, L]
        span_len = (end_pos - start_pos + 1).float().clamp(min=1.0)  # [B]

        # Mean pool
        h_a_sum = (text_hidden * in_span.unsqueeze(-1).float()).sum(dim=1)  # [B, D]
        h_a = h_a_sum / span_len.unsqueeze(-1)  # [B, D]

        # Project aspect token
        h_a = self.asp_proj(h_a)

        # Aspect-guided attention
        refined_visual, attn_weights = self.attention(
            aspect_hidden=h_a,
            visual_tokens=visual_tokens,
            visual_mask=visual_mask,
        )

        return refined_visual, attn_weights
