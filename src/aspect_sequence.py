"""
Aspect-specific sequence builder for the ACSA model.

Each aspect gets a short fixed-length sequence of 5 semantic tokens:
    [ASP_a]  — the aspect query embedding
    [TXT_EVI_a] — text evidence retrieved for aspect a
    [IMG_EVI_a] — image evidence retrieved for aspect a
    [ROI_EVI_a] — ROI evidence retrieved for aspect a
    [FUSE_a]  — gated fusion of the three evidence streams

All 5 tokens are pre-computed vectors [D_h] that are simply stacked and fed
as inputs_embeds to the LLM decoder. No tokenizer involvement.
"""

import torch
from typing import Tuple


def build_aspect_sequence(
    asp: torch.Tensor,
    txt_evi: torch.Tensor,
    img_evi: torch.Tensor,
    roi_evi: torch.Tensor,
    fuse: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Stack 5 aspect-specific tokens into a sequence for the LLM decoder.

    Args:
        asp:     aspect query embedding.      Shape: [B, D_h]
        txt_evi: text evidence vector.         Shape: [B, D_h]
        img_evi: image evidence vector.        Shape: [B, D_h]
        roi_evi: ROI evidence vector.          Shape: [B, D_h]
        fuse:    fused evidence vector.         Shape: [B, D_h]
        device:  torch device

    Returns:
        seq:       stacked sequence.  Shape: [B, 5, D_h]
        mask:      attention mask.     Shape: [B, 5]  (all 1s — no padding in this seq)
        position:  position ids.       Shape: [B, 5]
    """
    seq = torch.stack([asp, txt_evi, img_evi, roi_evi, fuse], dim=1)  # [B, 5, D_h]
    mask = torch.ones(seq.size(0), 5, dtype=torch.long, device=device)
    position = torch.arange(5, device=device).unsqueeze(0).expand(seq.size(0), -1)  # [B, 5]
    return seq, mask, position
