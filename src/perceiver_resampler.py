"""Perceiver Resampler: compress SwinV2 patch features into K=64 visual query tokens.

Architecture (Diagram.md — InternVL-style):
  [B*M, N_patches, 2048]  <- MLP Projector output (already LLM dim)
       -> learnable query tokens (Cross-Attn to vision features)
       -> FFN + residual
       -> [B*M, K, 2048]  where K = 64 << N_patches

The cross-attention queries (learnable) attend over the projected vision keys,
effectively compressing N_patches -> K tokens via attention-based selection.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceiverResampler(nn.Module):
    """Compress visual patch tokens into a fixed number of learnable query tokens.

    Input:  [B*M, N_patches, vision_dim=2048]  (from MLP Projector)
    Output: [B*M, num_queries=64, vision_dim=2048]
    """

    def __init__(
        self,
        vision_dim: int = 2048,
        num_queries: int = 64,
        num_heads: int = 8,
        expansion: int = 4,
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.head_dim = vision_dim // num_heads

        # Learnable query tokens (random init; InternVL uses Fourier positional encoding)
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, vision_dim))
        nn.init.normal_(self.query_tokens, std=0.02)

        # Project vision features for cross-attention key/value
        self.kv_proj = nn.Linear(vision_dim, vision_dim, bias=False)
        # Project queries
        self.q_proj = nn.Linear(vision_dim, vision_dim, bias=False)
        # Output projection
        self.o_proj = nn.Linear(vision_dim, vision_dim, bias=False)

        self.ln1 = nn.LayerNorm(vision_dim)
        self.ffn = nn.Sequential(
            nn.Linear(vision_dim, vision_dim * expansion),
            nn.GELU(),
            nn.Linear(vision_dim * expansion, vision_dim),
        )
        self.ln2 = nn.LayerNorm(vision_dim)

    def forward(
        self,
        vision_tokens: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            vision_tokens: [B*M, N_patches, vision_dim]  — from MLP Projector
            causal_mask: [num_queries, N_patches] bool mask. Optional.
        Returns:
            visual_queries: [B*M, num_queries, vision_dim]
        """
        B, N, _ = vision_tokens.shape

        # Project vision tokens to K and V
        kv = self.kv_proj(vision_tokens)     # [B, N, H]
        k = kv                                 # will reshape below
        v = kv

        # Expand learnable queries to batch size
        q = self.q_proj(self.query_tokens)    # [1, K, H] -> [B, K, H]
        q = q.expand(B, -1, -1)               # [B, K, H]

        # Reshape for multihead: [B, L, H] -> [B, num_heads, L, head_dim]
        q = q.view(B, self.num_queries, self.num_heads, self.head_dim).transpose(1, 2)   # [B, num_heads, K, head_dim]
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)                   # [B, num_heads, N, head_dim]
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)                   # [B, num_heads, N, head_dim]

        # Scaled dot-product cross-attention  (q:[B,H,K,d], k:[B,H,N,d] -> attn:[B,H,K,N])
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(2, 3)) * scale                                  # [B, num_heads, K, N]

        if causal_mask is not None:
            attn = attn.masked_fill(causal_mask.to(attn.device).unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn_out = torch.matmul(attn, v)                            # [B, num_heads, K, head_dim]
        attn_out = attn_out.transpose(1, 2).reshape(B, self.num_queries, -1)  # [B, K, H]
        attn_out = self.o_proj(attn_out)

        # Residual + norm  (Diagram: LN(x + attn))
        x = self.ln1(self.query_tokens.expand(B, -1, -1) + attn_out)

        # FFN + residual
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)

        return x
