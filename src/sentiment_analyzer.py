"""
Multimodal Sentiment Analyzer

Pipeline chính kết hợp Vision Encoder, Projection Layer, và LLM
để phân tích sentiment từ hình ảnh và văn bản.
"""

import torch
from typing import Optional, Dict, Any
from PIL import Image
from dataclasses import dataclass


@dataclass
class SentimentResult:
    """Kết quả phân tích sentiment."""
    sentiment: str          # positive, negative, neutral, mixed
    confidence: str         # high, medium, low
    reasoning: str          # Giải thích chi tiết
    raw_response: str       # Response gốc từ LLM


# Prompt template cho sentiment analysis
SENTIMENT_PROMPT_VI = """Phân tích cảm xúc/ý kiến được thể hiện trong hình ảnh này.
Xem xét cả nội dung hình ảnh và bất kỳ văn bản nào xuất hiện.

Phân loại cảm xúc tổng thể thành:
- Tích cực (positive): Thể hiện sự hài lòng, vui vẻ, đồng ý
- Tiêu cực (negative): Thể hiện sự không hài lòng, tức giận, phản đối
- Trung lập (neutral): Khách quan, không có cảm xúc rõ ràng
- Hỗn hợp (mixed): Chứa cả yếu tố tích cực và tiêu cực

Trả lời theo định dạng sau:
Cảm xúc: [phân loại]
Độ tin cậy: [cao/trung bình/thấp]
Giải thích: [giải thích ngắn gọn]
"""

SENTIMENT_PROMPT_EN = """Analyze the sentiment/opinion expressed in this image.
Consider both the visual content and any text present.

Classify the overall sentiment as:
- Positive: Expressing satisfaction, happiness, approval
- Negative: Expressing dissatisfaction, anger, disapproval
- Neutral: Factual, no clear emotional tone
- Mixed: Contains both positive and negative elements

Provide your analysis in the following format:
Sentiment: [classification]
Confidence: [high/medium/low]
Reasoning: [brief explanation]
"""


class MultimodalSentimentAnalyzer:
    """
    Phân tích sentiment từ hình ảnh sử dụng:
    - Google ViT-L/16 (Vision Encoder)
    - MLP Projector (Bridge)
    - InternVL3-8B-Instruct (LLM)
    """
    
    def __init__(
        self,
        model_name: str = "OpenGVLab/InternVL3-8B-Instruct",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        use_custom_vision: bool = False  # True để dùng ViT-L/16 riêng
    ):
        self.device = device
        self.use_custom_vision = use_custom_vision
        
        # Convert dtype string to torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }
        self.torch_dtype = dtype_map.get(torch_dtype, torch.bfloat16)
        
        print(f"Initializing Multimodal Sentiment Analyzer...")
        print(f"  Device: {device}")
        print(f"  Dtype: {torch_dtype}")
        print(f"  Model: {model_name}")
        
        if use_custom_vision:
            # Sử dụng custom ViT-L/16 + Projector + InternVL3 LLM
            from .llm_inference import InternVL3WithCustomVision
            self.model = InternVL3WithCustomVision(
                llm_name=model_name,
                device=device,
                torch_dtype=self.torch_dtype
            )
        else:
            # Sử dụng InternVL3 nguyên bản (có vision encoder tích hợp)
            from transformers import AutoModel, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map="auto"
            ).eval()
        
        print("✓ Sentiment Analyzer ready!")
    
    def analyze(
        self,
        image: Image.Image,
        custom_prompt: Optional[str] = None,
        language: str = "vi",  # "vi" hoặc "en"
        max_new_tokens: int = 512
    ) -> SentimentResult:
        """
        Phân tích sentiment từ hình ảnh.
        
        Args:
            image: PIL Image để phân tích
            custom_prompt: Prompt tuỳ chỉnh (optional)
            language: "vi" cho tiếng Việt, "en" cho tiếng Anh
            max_new_tokens: Số token tối đa để generate
            
        Returns:
            SentimentResult với classification và reasoning
        """
        # Chọn prompt
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = SENTIMENT_PROMPT_VI if language == "vi" else SENTIMENT_PROMPT_EN
        
        # Generate response
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }
        
        try:
            response = self.model.chat(
                self.tokenizer,
                image,
                prompt,
                generation_config
            )
        except Exception as e:
            return SentimentResult(
                sentiment="error",
                confidence="low",
                reasoning=f"Error during analysis: {str(e)}",
                raw_response=""
            )
        
        # Parse response
        result = self._parse_response(response, language)
        return result
    
    def _parse_response(self, response: str, language: str) -> SentimentResult:
        """Parse LLM response thành structured result."""
        lines = response.strip().split("\n")
        
        sentiment = "unknown"
        confidence = "medium"
        reasoning = ""
        
        # Mapping cho tiếng Việt và English
        sentiment_keywords = {
            "vi": {
                "positive": ["tích cực", "positive", "hài lòng", "vui"],
                "negative": ["tiêu cực", "negative", "không hài lòng", "tức giận"],
                "neutral": ["trung lập", "neutral", "khách quan"],
                "mixed": ["hỗn hợp", "mixed", "lẫn lộn"]
            },
            "en": {
                "positive": ["positive", "satisfied", "happy", "approval"],
                "negative": ["negative", "dissatisfied", "angry", "disapproval"],
                "neutral": ["neutral", "factual", "objective"],
                "mixed": ["mixed", "both"]
            }
        }
        
        for line in lines:
            line_lower = line.lower()
            
            # Parse sentiment
            if "cảm xúc:" in line_lower or "sentiment:" in line_lower:
                for sent, keywords in sentiment_keywords.get(language, sentiment_keywords["en"]).items():
                    if any(kw in line_lower for kw in keywords):
                        sentiment = sent
                        break
            
            # Parse confidence
            if "độ tin cậy:" in line_lower or "confidence:" in line_lower:
                if "cao" in line_lower or "high" in line_lower:
                    confidence = "high"
                elif "thấp" in line_lower or "low" in line_lower:
                    confidence = "low"
                else:
                    confidence = "medium"
            
            # Parse reasoning
            if "giải thích:" in line_lower or "reasoning:" in line_lower:
                reasoning = line.split(":", 1)[-1].strip()
        
        # Nếu không parse được, dùng response đầy đủ làm reasoning
        if not reasoning:
            reasoning = response
        
        return SentimentResult(
            sentiment=sentiment,
            confidence=confidence,
            reasoning=reasoning,
            raw_response=response
        )
    
    def batch_analyze(
        self,
        images: list,
        language: str = "vi"
    ) -> list:
        """
        Phân tích nhiều hình ảnh.
        
        Args:
            images: List of PIL Images
            language: "vi" hoặc "en"
            
        Returns:
            List of SentimentResult
        """
        results = []
        for i, image in enumerate(images):
            print(f"Analyzing image {i+1}/{len(images)}...")
            result = self.analyze(image, language=language)
            results.append(result)
        return results


# Quick test
if __name__ == "__main__":
    print("\n=== Sentiment Analyzer Module ===")
    print("To test, run: python app.py")
    print("This will start the Gradio web interface.")
