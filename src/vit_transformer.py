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

        # Unfreeze last 2 encoder stages (stages 2 and 3) for task-specific visual features.
        # Stages 0-1 remain frozen to preserve pretrained general-purpose representations.
        # Stages 2-3 train with a lower learning rate (managed by setup_optimizer).
        frozen_count = 0
        trainable_count = 0
        for name, param in self.model.named_parameters():
            if "encoder.layers.2" in name or "encoder.layers.3" in name:
                param.requires_grad = True
                trainable_count += 1
            else:
                param.requires_grad = False
                frozen_count += 1

        # Keep in eval mode so BatchNorm/rms_norm statistics are used as-is.
        # This prevents train mode from corrupting pretrained BN statistics.
        self.model.eval()
        print(f"Vision encoder: {frozen_count} params frozen (stages 0-1), "
              f"{trainable_count} params trainable (stages 2-3)")

        self.hidden_size = self.model.config.hidden_size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32, device=run_device)
            dummy_out = self.model(pixel_values=dummy).last_hidden_state
            self.num_patches = dummy_out.shape[1]

        print(f"Loaded: hidden_size={self.hidden_size}, num_patches={self.num_patches}")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.to(self.device, dtype=self.torch_dtype)
        outputs = self.model(pixel_values=pixel_values)
        return outputs.last_hidden_state
