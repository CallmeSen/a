import torch
import torch.nn as nn

from .config import VISION_HIDDEN_SIZE, LLM_HIDDEN_SIZE


class MLPProjector(nn.Module):
    def __init__(self, vision_dim: int = VISION_HIDDEN_SIZE, llm_dim: int = LLM_HIDDEN_SIZE):
        super().__init__()
        self.vision_dim = vision_dim
        self.llm_dim = llm_dim

        self.projector = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

        for module in self.projector:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, img_tokens: torch.Tensor) -> torch.Tensor:
        img_tokens = img_tokens.to(dtype=self.projector[0].weight.dtype)
        return self.projector(img_tokens)
