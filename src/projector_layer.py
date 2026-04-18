import torch
import torch.nn as nn

from src.config import VIT_HIDDEN, LLM_HIDDEN


class MLPProjector(nn.Module):
    """
    MLP projector for image patch tokens: D_v -> D_h -> D_h, GELU activation.
    """

    def __init__(self, d_v: int = VIT_HIDDEN, d_h: int = LLM_HIDDEN):
        super().__init__()
        self.fc1 = nn.Linear(d_v, d_h)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_h, d_h)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: [B, M, P, D_v]
        Returns:
            patch_tokens_h: [B, M, P, D_h]
        """
        B, M, P, D_v = patch_tokens.shape
        patch_flat = patch_tokens.reshape(B * M, P, D_v)
        patch_proj = self.act(self.fc1(patch_flat))
        patch_proj = self.fc2(patch_proj)
        return patch_proj.reshape(B, M, P, -1)


class RoIProjector(nn.Module):
    """
    MLP projector for RoI + pooled-image tokens: D_v -> D_h -> D_h, GELU activation.
    Token 0 is pooled full-image summary; tokens 1..K are per-RoI features.
    """

    def __init__(self, d_v: int = VIT_HIDDEN, d_h: int = LLM_HIDDEN):
        super().__init__()
        self.fc1 = nn.Linear(d_v, d_h)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_h, d_h)

    def forward(self, roi_img_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            roi_img_seq: [B, M, K_max, D_v]
        Returns:
            roi_lang: [B, M, K_max, D_h]
        """
        B, M, K, D_v = roi_img_seq.shape
        roi_flat = roi_img_seq.reshape(B * M, K, D_v)
        roi_proj = self.act(self.fc1(roi_flat))
        roi_proj = self.fc2(roi_proj)
        return roi_proj.reshape(B, M, K, -1)
