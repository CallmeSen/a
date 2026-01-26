"""
LLM Inference Module - InternVL3-8B

Tích hợp InternVL3-8B-Instruct để thực hiện multimodal reasoning.
"""

import torch
from typing import Optional, Union, List
from PIL import Image
from transformers import AutoModel, AutoTokenizer


class InternVL3LLM:
    """
    Wrapper cho InternVL3-8B-Instruct.
    
    InternVL3 là model multimodal hoàn chỉnh với vision encoder tích hợp.
    Tuy nhiên, trong kiến trúc này ta sử dụng vision features từ ViT-L/16 riêng.
    """
    
    def __init__(
        self,
        model_name: str = "OpenGVLab/InternVL3-8B-Instruct",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        self.device = device
        self.torch_dtype = torch_dtype
        self.model_name = model_name
        
        print(f"Loading LLM: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Load model với quantization options
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
        }
        
        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
            load_kwargs["device_map"] = "auto"
        elif load_in_4bit:
            load_kwargs["load_in_4bit"] = True
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = "auto"
        
        self.model = AutoModel.from_pretrained(
            model_name,
            **load_kwargs
        )
        
        self.model.eval()
        print(f"✓ LLM loaded successfully")
    
    def generate(
        self,
        pixel_values: torch.Tensor,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        do_sample: bool = False
    ) -> str:
        """
        Generate text response từ image và prompt.
        
        Args:
            pixel_values: Preprocessed image tensor
            prompt: Text prompt/question
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            Generated text response
        """
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
        }
        
        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            prompt,
            generation_config
        )
        
        return response
    
    def chat(
        self,
        image: Image.Image,
        question: str,
        history: Optional[List] = None,
        max_new_tokens: int = 512
    ) -> tuple:
        """
        Interactive chat với image context.
        
        Args:
            image: PIL Image
            question: User question
            history: Chat history (optional)
            max_new_tokens: Max tokens to generate
            
        Returns:
            (response, updated_history)
        """
        if history is None:
            history = []
        
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }
        
        response, history = self.model.chat(
            self.tokenizer,
            image,
            question,
            generation_config,
            history=history,
            return_history=True
        )
        
        return response, history


class InternVL3WithCustomVision:
    """
    InternVL3 sử dụng custom Vision Encoder (Google ViT-L/16).
    
    Kết hợp:
    - Vision Encoder: Google ViT-L/16 (riêng)
    - Projection Layer: MLP Bridge (riêng)
    - LLM: InternVL3 language model
    """
    
    def __init__(
        self,
        llm_name: str = "OpenGVLab/InternVL3-8B-Instruct",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16
    ):
        from .vision_encoder import VisionEncoder
        from .projection_layer import MLPProjector
        
        self.device = device
        self.torch_dtype = torch_dtype
        
        # Load Vision Encoder
        print("Loading Vision Encoder...")
        self.vision_encoder = VisionEncoder(
            model_name="google/vit-large-patch16-224",
            device=device,
            torch_dtype=torch_dtype
        )
        
        # Load Projection Layer
        print("Loading Projection Layer...")
        self.projector = MLPProjector(
            vision_dim=1024,  # ViT-L/16
            llm_dim=4096      # InternVL3
        ).to(device).to(torch_dtype)
        
        # Load LLM
        print("Loading LLM...")
        self.llm = InternVL3LLM(
            model_name=llm_name,
            device=device,
            torch_dtype=torch_dtype
        )
        
        print("✓ Full model loaded!")
    
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        Encode image thành visual features đã aligned.
        
        Args:
            image: PIL Image
            
        Returns:
            Aligned visual features [1, 197, 4096]
        """
        # Extract features từ ViT-L/16
        visual_features = self.vision_encoder.encode(image)
        
        # Project sang LLM embedding space
        aligned_features = self.projector(visual_features)
        
        return aligned_features


# Test module
if __name__ == "__main__":
    print("\n=== Testing LLM Module ===")
    print("Note: This requires downloading InternVL3-8B (~16GB)")
    print("Skipping full test to avoid large download.")
    print("Run sentiment_analyzer.py for full integration test.")
