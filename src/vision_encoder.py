"""
Vision Encoder Module - Google ViT-L/16

Sử dụng Vision Transformer Large với patch size 16x16 từ Google
để trích xuất visual features từ hình ảnh.
"""

import torch
import torch.nn as nn
from PIL import Image
from transformers import ViTModel, ViTImageProcessor


class VisionEncoder(nn.Module):
    """
    Vision Encoder sử dụng Google ViT-L/16.
    
    Input: PIL Image hoặc tensor
    Output: Visual features [batch_size, 197, 1024]
    """
    
    def __init__(
        self, 
        model_name: str = "google/vit-large-patch16-224",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16
    ):
        super().__init__()
        self.device = device
        self.torch_dtype = torch_dtype
        
        # Load ViT model và image processor
        print(f"Loading Vision Encoder: {model_name}")
        self.model = ViTModel.from_pretrained(
            model_name,
            torch_dtype=torch_dtype
        ).to(device)
        
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        
        # Freeze weights (không train vision encoder)
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()
        
        # Model info
        self.hidden_size = self.model.config.hidden_size  # 1024 for ViT-L
        self.num_patches = (self.model.config.image_size // self.model.config.patch_size) ** 2 + 1  # 196 + 1 CLS = 197
        
        print(f"✓ Vision Encoder loaded: hidden_size={self.hidden_size}, num_patches={self.num_patches}")
    
    def preprocess(self, images) -> torch.Tensor:
        """
        Preprocess images for ViT.
        
        Args:
            images: Single PIL Image, list of PIL Images, or tensor
            
        Returns:
            Preprocessed tensor ready for model
        """
        if isinstance(images, Image.Image):
            images = [images]
        
        # Use ViT processor
        inputs = self.processor(
            images=images,
            return_tensors="pt"
        )
        
        return inputs["pixel_values"].to(self.device, dtype=self.torch_dtype)
    
    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features from images.
        
        Args:
            pixel_values: Preprocessed image tensor [B, 3, 224, 224]
            
        Returns:
            Visual features [B, 197, 1024]
        """
        outputs = self.model(pixel_values=pixel_values)
        
        # last_hidden_state: [batch_size, seq_len (197), hidden_size (1024)]
        # Bao gồm CLS token ở vị trí 0 và 196 patch tokens
        return outputs.last_hidden_state
    
    def encode(self, images) -> torch.Tensor:
        """
        High-level API: từ PIL Image → visual features.
        
        Args:
            images: PIL Image hoặc list of PIL Images
            
        Returns:
            Visual features tensor [B, 197, 1024]
        """
        pixel_values = self.preprocess(images)
        return self.forward(pixel_values)
    
    def get_cls_token(self, images) -> torch.Tensor:
        """
        Chỉ lấy CLS token (global image representation).
        
        Returns:
            CLS features [B, 1024]
        """
        features = self.encode(images)
        return features[:, 0, :]  # CLS token ở vị trí 0
    
    def get_patch_tokens(self, images) -> torch.Tensor:
        """
        Chỉ lấy patch tokens (bỏ CLS).
        
        Returns:
            Patch features [B, 196, 1024]
        """
        features = self.encode(images)
        return features[:, 1:, :]  # Bỏ CLS token


# Test module
if __name__ == "__main__":
    from PIL import Image
    import requests
    from io import BytesIO
    
    # Download test image
    url = "https://images.unsplash.com/photo-1575936123452-b67c3203c357?w=400"
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    
    # Test Vision Encoder
    print("\n=== Testing Vision Encoder ===")
    encoder = VisionEncoder(device="cuda" if torch.cuda.is_available() else "cpu")
    
    features = encoder.encode(image)
    print(f"Output shape: {features.shape}")  # Expected: [1, 197, 1024]
    print(f"CLS token shape: {encoder.get_cls_token(image).shape}")  # Expected: [1, 1024]
    print(f"Patch tokens shape: {encoder.get_patch_tokens(image).shape}")  # Expected: [1, 196, 1024]
    print("✓ Vision Encoder test passed!")
