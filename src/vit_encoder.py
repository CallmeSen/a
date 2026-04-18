import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple

from src.config import VIT_NAME, VIT_HIDDEN, IMAGE_SIZE


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
ROI_CROP_SIZE = 256


def _normalize_single(img_np: np.ndarray) -> torch.Tensor:
    """Normalize a [H,W,3] uint8 image to ImageNet norm, return [3,H,W] tensor."""
    img = img_np.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img.transpose(2, 0, 1))


class SigLIPEncoder(nn.Module):
    """SigLIP2-Large encoder — shared for patch and RoI encoding."""

    def __init__(self, model_name: str = VIT_NAME, image_size: int = IMAGE_SIZE):
        super().__init__()
        from transformers import AutoModel, AutoProcessor

        self.model_name = model_name
        self.image_size = image_size

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        self.hidden_size = self._get_hidden_size()
        self.num_patches = self._compute_num_patches()

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def _get_hidden_size(self) -> int:
        config = self.model.config
        if hasattr(config, "hidden_size"):
            return config.hidden_size
        if hasattr(config, "vision_config") and hasattr(config.vision_config, "hidden_size"):
            return config.vision_config.hidden_size
        return VIT_HIDDEN

    def _get_patch_size(self) -> int:
        config = self.model.config
        if hasattr(config, "patch_size"):
            return config.patch_size
        if hasattr(config, "vision_config") and hasattr(config.vision_config, "patch_size"):
            return config.vision_config.patch_size
        return 16

    def _compute_num_patches(self) -> int:
        patch_size = self._get_patch_size()
        return (self.image_size // patch_size) ** 2

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pixel_values: [B, M, 3, H, W] — normalized via SigLIP processor
        Returns:
            patch_tokens: [B, M, P, D_v]
            img_summaries: [B, M, D_v]
        """
        B, M, C, H, W = pixel_values.shape
        pixel_values_flat = pixel_values.view(B * M, C, H, W)

        with torch.no_grad():
            outputs = self.model.vision_model(pixel_values_flat)
            last_hidden = outputs.last_hidden_state

        seq_len = last_hidden.shape[1]

        if seq_len == self.num_patches + 1:
            img_summary = last_hidden[:, 0, :]
            patches = last_hidden[:, 1:, :]
        elif seq_len == self.num_patches:
            img_summary = last_hidden.mean(dim=1)
            patches = last_hidden
        else:
            import logging
            logging.getLogger(__name__).warning(
                f"Unexpected seq_len={seq_len}, expected {self.num_patches} or {self.num_patches+1}. "
                "Falling back to mean-pool."
            )
            img_summary = last_hidden.mean(dim=1)
            patches = last_hidden

        patch_tokens = patches.reshape(B, M, self.num_patches, self.hidden_size)
        img_summaries = img_summary.reshape(B, M, self.hidden_size)

        return patch_tokens, img_summaries

    # ── RoI encoding (shared vision backbone) ────────────────────────────────

    def _crop_and_resize_roi(
        self,
        full_img: np.ndarray,
        boxes: List[List[float]],
        crop_size: int = ROI_CROP_SIZE,
    ) -> torch.Tensor:
        """Crop and resize RoI regions. Returns [K, 3, crop_size, crop_size] ImageNet-normed."""
        H, W = full_img.shape[:2]
        crops = []
        for box in boxes:
            x1, y1, x2, y2 = box
            px1 = max(0, int(round(x1 * W)))
            py1 = max(0, int(round(y1 * H)))
            px2 = min(W, int(round(x2 * W)))
            py2 = min(H, int(round(y2 * H)))
            if px2 <= px1 or py2 <= py1:
                continue
            crop = full_img[py1:py2, px1:px2]
            pil_crop = Image.fromarray(crop).resize((crop_size, crop_size), Image.LANCZOS)
            crop_np = np.array(pil_crop)
            crops.append(_normalize_single(crop_np))

        if len(crops) == 0:
            return torch.zeros(0, 3, crop_size, crop_size, dtype=torch.float32)
        return torch.stack(crops)

    def _encode_roi_crops(self, crops: torch.Tensor) -> torch.Tensor:
        """Encode RoI crops using the shared vision_model. Returns [K, D_v]."""
        if crops.size(0) == 0:
            return torch.zeros(0, self.hidden_size, dtype=torch.float32)

        with torch.no_grad():
            outputs = self.model.vision_model(crops)
            last_hidden = outputs.last_hidden_state
            if last_hidden.size(1) > 1:
                roi_feats = last_hidden[:, 0, :]
            else:
                roi_feats = last_hidden.squeeze(1)
        return roi_feats

    def _encode_full_image_for_roi(self, full_img_tensor: torch.Tensor) -> torch.Tensor:
        """Encode full image for RoI branch — returns pooled summary [D_v]."""
        if full_img_tensor.dim() == 3:
            full_img_tensor = full_img_tensor.unsqueeze(0)
        with torch.no_grad():
            outputs = self.model.vision_model(full_img_tensor)
            last_hidden = outputs.last_hidden_state
            if last_hidden.size(1) > 1:
                summary = last_hidden[:, 0, :]
            else:
                summary = last_hidden.mean(dim=1)
        return summary.squeeze(0)

    def encode_roi(
        self,
        pixel_values: torch.Tensor,
        roi_data: List[List[Dict[str, Any]]],
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, M, 3, H, W] — ImageNet-normalized tensors
            roi_data: List[List[dict]] — per sample, per image:
                [{"boxes": [[x1,y1,x2,y2], ...], "labels": [...]}, ...]

        Returns:
            roi_img_seq: [B, M, K_max, D_v]
                Token 0: pooled full-image summary [D_v]
                Token 1..K: per-RoI features [K, D_v]
                Padded with zeros to K_max.
        """
        B, M, C, H, W = pixel_values.shape
        D_v = self.hidden_size

        max_k = 1
        for sample_rois in roi_data:
            for img_roi in sample_rois:
                boxes = img_roi.get("boxes", [])
                max_k = max(max_k, len(boxes) + 1)

        K_max = max_k
        device = pixel_values.device
        roi_img_seq = torch.zeros(B, M, K_max, D_v, device=device, dtype=torch.float32)

        for b in range(B):
            sample_rois = roi_data[b] if b < len(roi_data) else []
            for m in range(M):
                # Convert normalized tensor [3,H,W] → [H,W,3] uint8
                img_normed = pixel_values[b, m]  # [3,H,W] normalized
                img_np = (img_normed.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

                summary_feat = self._encode_full_image_for_roi(img_normed.unsqueeze(0).to(device))
                roi_img_seq[b, m, 0, :] = summary_feat

                img_roi = sample_rois[m] if m < len(sample_rois) else {"boxes": [], "labels": []}
                boxes = img_roi.get("boxes", [])
                if len(boxes) == 0:
                    continue

                crops = self._crop_and_resize_roi(img_np, boxes, ROI_CROP_SIZE)
                if crops.size(0) == 0:
                    continue

                crops = crops.to(device)
                roi_feats = self._encode_roi_crops(crops)
                K_actual = min(roi_feats.size(0), K_max - 1)
                roi_img_seq[b, m, 1:1 + K_actual, :] = roi_feats[:K_actual]

        return roi_img_seq
