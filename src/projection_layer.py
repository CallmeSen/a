"""
Projection Layer Module - MLP Bridge

Chuyển đổi visual features từ ViT-L/16 (1024 dim) 
sang không gian embedding của InternVL3 LLM (4096 dim).
"""

import torch
import torch.nn as nn


class MLPProjector(nn.Module):
    """
    MLP Projector để align visual features với LLM embedding space.
    
    Architecture:
        Linear(1024 → 4096) → GELU → Linear(4096 → 4096)
    
    Input: Visual features [B, 197, 1024] từ ViT-L/16
    Output: Aligned features [B, 197, 4096] cho LLM
    """
    
    def __init__(
        self,
        vision_dim: int = 1024,   # ViT-L/16 hidden dim
        llm_dim: int = 4096,      # InternVL3 hidden dim
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.llm_dim = llm_dim
        
        # 2-layer MLP với GELU activation
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_dim, llm_dim),
            nn.Dropout(dropout)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization cho các Linear layers."""
        for module in self.projector:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Project visual features sang LLM embedding space.
        
        Args:
            visual_features: [B, num_tokens, 1024] từ ViT-L/16
            
        Returns:
            Projected features [B, num_tokens, 4096]
        """
        return self.projector(visual_features)


class PixelShuffleProjector(nn.Module):
    """
    Alternative projector với pixel shuffle để giảm số tokens.
    
    Sử dụng khi cần giảm sequence length cho efficiency.
    """
    
    def __init__(
        self,
        vision_dim: int = 1024,
        llm_dim: int = 4096,
        scale_factor: int = 2  # Giảm tokens x4 (2x2)
    ):
        super().__init__()
        
        self.scale_factor = scale_factor
        
        # Projector với reshape
        self.projector = nn.Sequential(
            nn.Linear(vision_dim * scale_factor * scale_factor, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )
    
    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Project và giảm số tokens.
        
        Args:
            visual_features: [B, 196, 1024] (không có CLS)
            
        Returns:
            [B, 49, 4096] (giảm 4x số tokens)
        """
        B, N, C = visual_features.shape
        
        # Reshape thành grid (bỏ CLS token nếu có)
        H = W = int(N ** 0.5)  # 14x14 cho 196 tokens
        
        # Reshape và merge patches
        x = visual_features.view(B, H, W, C)
        
        # Group neighboring patches
        H_new = H // self.scale_factor
        W_new = W // self.scale_factor
        
        x = x.view(B, H_new, self.scale_factor, W_new, self.scale_factor, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H_new * W_new, -1)  # [B, 49, 4096]
        
        return self.projector(x)


# Test module
if __name__ == "__main__":
    print("\n=== Testing Projection Layer ===")
    
    # Test MLPProjector
    projector = MLPProjector(vision_dim=1024, llm_dim=4096)
    print(f"Number of parameters: {sum(p.numel() for p in projector.parameters()):,}")
    
    # Dummy input như output từ ViT-L/16
    dummy_features = torch.randn(2, 197, 1024)  # [batch, num_tokens, vision_dim]
    
    output = projector(dummy_features)
    print(f"Input shape: {dummy_features.shape}")
    print(f"Output shape: {output.shape}")  # Expected: [2, 197, 4096]
    
    # Test PixelShuffleProjector
    print("\n--- PixelShuffleProjector ---")
    ps_projector = PixelShuffleProjector(vision_dim=1024, llm_dim=4096, scale_factor=2)
    
    # Input without CLS token
    patch_features = torch.randn(2, 196, 1024)  # [batch, 196 patches, vision_dim]
    ps_output = ps_projector(patch_features)
    print(f"Input shape: {patch_features.shape}")
    print(f"Output shape: {ps_output.shape}")  # Expected: [2, 49, 4096]
    
    print("✓ Projection Layer test passed!")
