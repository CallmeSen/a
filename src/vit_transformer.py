import torch
import torch.nn as nn
from transformers import AutoModel

from .config import device, COMPUTE_DTYPE, VISION_MODEL_NAME, IMAGE_SIZE


class VisionEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = VISION_MODEL_NAME,
        run_device: str = device,
        torch_dtype: torch.dtype = COMPUTE_DTYPE,
    ):
        super().__init__()
        self.device = run_device
        self.torch_dtype = torch.float32

        print(f"Loading Swin Transformer V2: {model_name} (torch_dtype={self.torch_dtype})")
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32).to(run_device)

        # Unfreeze ALL stages for domain-specific visual feature learning.
        # SwinV2-Base pretrained on ImageNet does not capture hotel-specific features
        # (pool views, room decor, food presentation, lobby aesthetics).
        # With vision_lr_ratio=0.1 in setup_optimizer, the low LR prevents
        # pretrained representations from being corrupted too quickly.
        frozen_count = 0
        trainable_count = 0
        for param in self.model.parameters():
            param.requires_grad = True
            trainable_count += 1

        # Use train mode so all norm layers (LayerNorm, BatchNorm) update their
        # running statistics during training. This is essential for domain adaptation.
        self.model.train()
        print(f"Vision encoder: {trainable_count} params trainable (all stages)")

        self.hidden_size = self.model.config.hidden_size
        # Dummy forward in train mode to initialize BatchNorm running stats properly
        dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32, device=run_device)
        dummy_out = self.model(pixel_values=dummy).last_hidden_state
        self.num_patches = dummy_out.shape[1]

        print(f"Loaded: hidden_size={self.hidden_size}, num_patches={self.num_patches}")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.to(self.device, dtype=self.torch_dtype)
        outputs = self.model(pixel_values=pixel_values)
        return outputs.last_hidden_state
