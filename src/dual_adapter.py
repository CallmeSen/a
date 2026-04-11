"""Dual Branch Gated Cross-Attention Adapter.

Nâng cấp GatedCrossAttentionAdapter bằng Dual-LoRA architecture:
- Shared Branch: low-rank (rank=r) cho general cross-modal patterns
- Task Branch: low-rank cho task-specific features, gated by ReLU
- Final gate: kiểm soát tổng output

So với GatedCrossAttentionAdapter gốc (4 × full-rank projections):
- Giảm ~90% parameters nhờ LoRA decomposition
- Tách biệt knowledge extraction và task-specific refinement
- ReLU gating cho dynamic, input-dependent routing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used in Qwen2)."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * x).to(input_dtype)


class DualGatedCrossAttentionAdapter(nn.Module):
    """
    Dual Branch Gated Cross-Attention Adapter.

    Architecture:
        text_hidden [B, L, D]
            │
            ├── Shared Branch (LoRA, always active):
            │   ├── A_s (D → r), B_s (r → D) — low-rank decomposition
            │   ├── Full-rank cross-attention on projected Q, K, V
            │
            ├── Task Branch (LoRA, gated):
            │   ├── A_t (D → r), B_t (r → D) — low-rank decomposition
            │   ├── Full-rank cross-attention on projected Q, K, V
            │   └── × task_gate (from gating network)
            │
            └── Final gate (scalar)
                output = text_hidden + final_gate * (shared + task_gate × task)

    Parameters:
        hidden_size: LLM hidden dimension (2560 for Qwen3-4B)
        num_heads: số attention heads (8)
        rank: low-rank dimension (64)
    """

    def __init__(
        self,
        hidden_size: int = 2560,
        num_heads: int = 8,
        rank: int = 64,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.rank = rank

        # ── Shared Branch LoRA decomposition ────────────────────────
        # LoRA: full matrix W (D×D) ≈ A_s @ B_s where A: D→r, B: r→D
        # Low-param: only 2 × D×r instead of D×D
        self.A_s = nn.Linear(hidden_size, rank, bias=False)   # D → r
        self.B_s = nn.Linear(rank, hidden_size, bias=False)    # r → D

        # ── Task Branch LoRA decomposition ─────────────────────────
        self.A_t = nn.Linear(hidden_size, rank, bias=False)   # D → r
        self.B_t = nn.Linear(rank, hidden_size, bias=False)    # r → D

        # ── Gating Network (ReLU-based) ───────────────────────────
        self.gate_fc1 = nn.Linear(hidden_size, hidden_size // 4)
        self.gate_fc2 = nn.Linear(hidden_size // 4, 1)
        self.task_gate_bias = nn.Parameter(torch.tensor(-1.0))

        # ── RMSNorm + Final Gate ─────────────────────────────────
        self.rms_norm = RMSNorm(hidden_size)
        self.final_gate = nn.Parameter(torch.tensor(-1.0))

        self._init_weights()

    def _init_weights(self) -> None:
        # LoRA: init A with zeros, B with normal — so initial delta W ≈ 0
        nn.init.zeros_(self.A_s.weight)
        nn.init.normal_(self.B_s.weight, std=0.02)
        nn.init.zeros_(self.A_t.weight)
        nn.init.normal_(self.B_t.weight, std=0.02)
        nn.init.xavier_uniform_(self.gate_fc1.weight)
        nn.init.zeros_(self.gate_fc1.bias)
        nn.init.xavier_uniform_(self.gate_fc2.weight)
        nn.init.zeros_(self.gate_fc2.bias)

    def _cross_attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Multi-head cross-attention with full hidden dim for Q, K, V."""
        bsz, q_len, D = q.shape
        _, kv_len, _ = k.shape

        q_h = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_h = k.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_h = v.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * scale

        if mask is not None:
            invalid = ~mask.bool()
            scores = scores.masked_fill(
                invalid[:, None, None, :],
                torch.finfo(scores.dtype).min,
            )

        attn_probs = F.softmax(scores.float(), dim=-1).to(dtype=scores.dtype)
        attn_out = torch.matmul(attn_probs, v_h)
        return attn_out.transpose(1, 2).contiguous().view(bsz, q_len, D)

    def forward(
        self,
        text_hidden: torch.Tensor,
        visual_tokens: torch.Tensor,
        visual_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            text_hidden: [B, L, D] - Qwen self-attention output
            visual_tokens: [B, N_vis, D] - Visual tokens from VCE/Perceiver
            visual_mask: [B, N_vis] bool - Valid visual token mask

        Returns:
            adapted: [B, L, D] - text_hidden + gated visual injection
        """
        bsz, q_len, D = text_hidden.shape
        _, kv_len, _ = visual_tokens.shape

        x_norm = self.rms_norm(text_hidden)

        # ── Shared Branch ──────────────────────────────────────────
        # Q from text, K and V from visual tokens (LoRA-projected)
        Q_s = self.B_s(self.A_s(x_norm))              # [B, L, D]
        K_s = self.B_s(self.A_s(visual_tokens))       # [B, N, D]
        V_s = self.B_s(self.A_s(visual_tokens))       # [B, N, D]
        shared_out = self._cross_attend(Q_s, K_s, V_s, visual_mask)

        # ── Task Branch ──────────────────────────────────────────
        # Q from text, K and V from visual tokens (LoRA-projected)
        Q_t = self.B_t(self.A_t(x_norm))              # [B, L, D]
        K_t = self.B_t(self.A_t(visual_tokens))        # [B, N, D]
        V_t = self.B_t(self.A_t(visual_tokens))        # [B, N, D]
        task_out = self._cross_attend(Q_t, K_t, V_t, visual_mask)

        # ── ReLU Gating ──────────────────────────────────────────
        text_agg = x_norm.mean(dim=1)                       # [B, D]
        gate_hidden = F.relu(self.gate_fc1(text_agg))       # [B, D/4]
        task_gate = torch.sigmoid(
            self.gate_fc2(gate_hidden) + self.task_gate_bias
        ).unsqueeze(-1)                                     # [B, 1, 1] for broadcast

        # ── Combine Branches ─────────────────────────────────────
        combined = shared_out + task_gate * task_out         # [B, L, D]

        # ── Final Gate ───────────────────────────────────────────
        final_scale = torch.sigmoid(self.final_gate)         # scalar
        adapted = text_hidden + final_scale * combined      # [B, L, D]

        return adapted
