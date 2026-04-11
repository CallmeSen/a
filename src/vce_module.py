"""Visual Cue Enhancement (VCE) Module.

Trích xuất multi-scale features từ SwinV2 và fusion thành 64 visual tokens.
Thay thế PerceiverResampler bằng multi-stage feature fusion.

SwinV2 stages:
- Stage 1: [B, 256, 64, 64] - texture, fine-grained details
- Stage 2: [B, 512, 32, 32] - local structure
- Stage 3: [B, 1024, 16, 16] - semantic features
- Stage 4: [B, 1024, 8, 8]   - global context
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleVisualFusion(nn.Module):
    """
    VCE Module: Fusion multi-scale SwinV2 features into 64 visual tokens.

    Mỗi stage được projection về cùng hidden dimension, sau đó:
    - Stage 1-3: Spatial pooling từ N patches về 64 tokens
    - Stage 4: Giữ nguyên 64 tokens (đã là 8x8)
    - Concat tất cả 4 stage → Project về LLM dimension
    """

    def __init__(
        self,
        llm_dim: int = 2560,
        stage_dims: tuple = (256, 512, 1024, 1024),
        num_tokens: int = 64,
    ):
        super().__init__()
        f1_dim, f2_dim, f3_dim, f4_dim = stage_dims

        # Stage 1 projection: 256 → 640 (4 × 160)
        self.proj1 = nn.Sequential(
            nn.Linear(f1_dim, 640),
            nn.GELU(),
            nn.Linear(640, llm_dim),
        )

        # Stage 2 projection: 512 → 640
        self.proj2 = nn.Sequential(
            nn.Linear(f2_dim, 640),
            nn.GELU(),
            nn.Linear(640, llm_dim),
        )

        # Stage 3 projection: 1024 → 640
        self.proj3 = nn.Sequential(
            nn.Linear(f3_dim, 640),
            nn.GELU(),
            nn.Linear(640, llm_dim),
        )

        # Stage 4 projection: 1024 → 640
        self.proj4 = nn.Sequential(
            nn.Linear(f4_dim, 640),
            nn.GELU(),
            nn.Linear(640, llm_dim),
        )

        # Final fusion: 4 × 640 = 2560 → llm_dim
        self.fusion_proj = nn.Linear(640 * 4, llm_dim)
        self.norm = nn.LayerNorm(llm_dim)

        # Learnable weighting cho mỗi stage (để model tự học importance)
        self.stage_weights = nn.Parameter(torch.ones(4) / 4)

    def _spatial_pool(
        self, x: torch.Tensor, target_tokens: int
    ) -> torch.Tensor:
        """Pool [B, N, D] → [B, target_tokens, D] qua adaptive average pooling."""
        B, N, D = x.shape
        x_t = x.transpose(1, 2)  # [B, D, N]
        x_pooled = F.adaptive_avg_pool1d(x_t, target_tokens)
        return x_pooled.transpose(1, 2)  # [B, target_tokens, D]

    def forward(self, multi_stage_features: dict) -> torch.Tensor:
        """
        Args:
            multi_stage_features: dict với keys 'stage1', 'stage2', 'stage3', 'stage4'
                - stage1: [B, 256, 64, 64] (flattened: [B, 4096, 256])
                - stage2: [B, 512, 32, 32] (flattened: [B, 1024, 512])
                - stage3: [B, 1024, 16, 16] (flattened: [B, 256, 1024])
                - stage4: [B, 1024, 8, 8] (flattened: [B, 64, 1024])
                Hoặc input có thể là [B, H*W, D] đã flatten sẵn

        Returns:
            fused: [B, 64, llm_dim]
        """
        # Handle both [B, H, W, D] and [B, N, D] formats
        def flatten_if_4d(x):
            if x.dim() == 4:
                B, C, H, W = x.shape
                return x.reshape(B, H * W, C)
            return x

        s1 = flatten_if_4d(multi_stage_features["stage1"])  # [B, N1, 256]
        s2 = flatten_if_4d(multi_stage_features["stage2"])  # [B, N2, 512]
        s3 = flatten_if_4d(multi_stage_features["stage3"])  # [B, N3, 1024]
        s4 = flatten_if_4d(multi_stage_features["stage4"])  # [B, N4, 1024]

        # Project each stage to llm_dim
        f1 = self.proj1(s1)  # [B, N1, llm_dim]
        f2 = self.proj2(s2)  # [B, N2, llm_dim]
        f3 = self.proj3(s3)  # [B, N3, llm_dim]
        f4 = self.proj4(s4)  # [B, N4, llm_dim]

        # Pool to 64 tokens for stages 1-3; keep all tokens for stage 4
        p1 = self._spatial_pool(f1, 16)   # [B, 16, llm_dim]
        p2 = self._spatial_pool(f2, 16)   # [B, 16, llm_dim]
        p3 = self._spatial_pool(f3, 16)   # [B, 16, llm_dim]
        p4 = self._spatial_pool(f4, 16)   # [B, 16, llm_dim] (from 64 tokens)

        # Concat: 16+16+16+16 = 64 tokens
        fused = torch.cat([p1, p2, p3, p4], dim=1)  # [B, 64, llm_dim]

        # Final projection
        out = self.norm(self.fusion_proj(fused))
        return out
