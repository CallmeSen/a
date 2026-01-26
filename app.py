"""
Multimodal Sentiment Analysis - Gradio Web Interface

Giao diện web để demo hệ thống phân tích ý kiến người dùng
qua hình ảnh sử dụng Google ViT-L/16 + InternVL3-8B.
"""

import gradio as gr
from PIL import Image
import torch
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_model():
    """Load sentiment analyzer model."""
    from src.sentiment_analyzer import MultimodalSentimentAnalyzer
    
    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use bfloat16 if available, else float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        torch_dtype = "bfloat16"
    else:
        torch_dtype = "float16"
    
    print(f"Loading model on {device} with {torch_dtype}...")
    
    analyzer = MultimodalSentimentAnalyzer(
        model_name="OpenGVLab/InternVL3-8B-Instruct",
        device=device,
        torch_dtype=torch_dtype,
        use_custom_vision=False  # Dùng InternVL3 native vision
    )
    
    return analyzer


# Global model instance
MODEL = None


def ensure_model_loaded():
    """Lazy load model khi cần."""
    global MODEL
    if MODEL is None:
        MODEL = load_model()
    return MODEL


def analyze_sentiment(
    image: Image.Image,
    language: str = "Tiếng Việt",
    custom_prompt: str = ""
) -> tuple:
    """
    Analyze sentiment từ hình ảnh.
    
    Returns:
        (sentiment, confidence, reasoning, raw_response)
    """
    if image is None:
        return "❌ Vui lòng upload hình ảnh", "", "", ""
    
    try:
        analyzer = ensure_model_loaded()
        
        # Map language
        lang = "vi" if language == "Tiếng Việt" else "en"
        
        # Analyze
        result = analyzer.analyze(
            image=image,
            custom_prompt=custom_prompt if custom_prompt.strip() else None,
            language=lang
        )
        
        # Format sentiment với emoji
        sentiment_emoji = {
            "positive": "😊 Tích cực (Positive)",
            "negative": "😞 Tiêu cực (Negative)",
            "neutral": "😐 Trung lập (Neutral)",
            "mixed": "🤔 Hỗn hợp (Mixed)",
            "unknown": "❓ Không xác định"
        }
        sentiment_display = sentiment_emoji.get(result.sentiment, result.sentiment)
        
        # Format confidence
        confidence_display = {
            "high": "🟢 Cao (High)",
            "medium": "🟡 Trung bình (Medium)",
            "low": "🔴 Thấp (Low)"
        }.get(result.confidence, result.confidence)
        
        return sentiment_display, confidence_display, result.reasoning, result.raw_response
        
    except Exception as e:
        error_msg = f"❌ Lỗi: {str(e)}"
        return error_msg, "", "", str(e)


def create_demo():
    """Create Gradio demo interface."""
    
    with gr.Blocks(
        title="Phân Tích Ý Kiến Đa Phương Thức",
        theme=gr.themes.Soft(),
        css="""
        .main-title {
            text-align: center;
            margin-bottom: 20px;
        }
        .result-box {
            padding: 15px;
            border-radius: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        """
    ) as demo:
        
        # Header
        gr.Markdown(
            """
            # 🎯 Phân Tích Ý Kiến Người Dùng Qua Hình Ảnh
            
            **Multimodal Sentiment Analysis** sử dụng:
            - 👁️ **Vision Encoder**: Google ViT-L/16
            - 🔗 **Projection Layer**: MLP Bridge
            - 🧠 **LLM**: InternVL3-8B-Instruct
            
            Upload hình ảnh chứa nội dung review, comment, hoặc bất kỳ ý kiến nào để phân tích.
            """,
            elem_classes="main-title"
        )
        
        with gr.Row():
            # Left column - Input
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="📷 Upload Hình Ảnh",
                    height=400
                )
                
                language_select = gr.Radio(
                    choices=["Tiếng Việt", "English"],
                    value="Tiếng Việt",
                    label="🌐 Ngôn ngữ phân tích"
                )
                
                custom_prompt = gr.Textbox(
                    label="📝 Prompt tuỳ chỉnh (để trống để dùng mặc định)",
                    placeholder="Nhập prompt của bạn...",
                    lines=3
                )
                
                analyze_btn = gr.Button(
                    "🔍 Phân Tích Sentiment",
                    variant="primary",
                    size="lg"
                )
            
            # Right column - Output
            with gr.Column(scale=1):
                sentiment_output = gr.Textbox(
                    label="🎯 Kết Quả Sentiment",
                    interactive=False,
                    lines=1
                )
                
                confidence_output = gr.Textbox(
                    label="📊 Độ Tin Cậy",
                    interactive=False,
                    lines=1
                )
                
                reasoning_output = gr.Textbox(
                    label="💬 Giải Thích",
                    interactive=False,
                    lines=5
                )
                
                with gr.Accordion("📄 Raw Response", open=False):
                    raw_output = gr.Textbox(
                        label="Response gốc từ LLM",
                        interactive=False,
                        lines=10
                    )
        
        # Examples
        gr.Markdown("### 📌 Ví dụ sử dụng")
        gr.Markdown(
            """
            - **Review sản phẩm**: Upload screenshot review từ Shopee, Lazada, Amazon...
            - **Comment mạng xã hội**: Screenshot comment Facebook, Twitter, Instagram...
            - **Meme/Hình ảnh có text**: Hình ảnh có chứa văn bản thể hiện ý kiến
            - **Biểu cảm khuôn mặt**: Ảnh người với biểu cảm rõ ràng
            """
        )
        
        # Event handler
        analyze_btn.click(
            fn=analyze_sentiment,
            inputs=[image_input, language_select, custom_prompt],
            outputs=[sentiment_output, confidence_output, reasoning_output, raw_output]
        )
        
        # Footer
        gr.Markdown(
            """
            ---
            **Architecture**: Google ViT-L/16 → MLP Projector → InternVL3-8B-Instruct
            
            Made with ❤️ using Gradio
            """
        )
    
    return demo


if __name__ == "__main__":
    print("=" * 50)
    print("🚀 Starting Multimodal Sentiment Analysis Demo")
    print("=" * 50)
    
    demo = create_demo()
    
    # Launch
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set True để tạo public URL
        inbrowser=True
    )
