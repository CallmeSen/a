# Multimodal Sentiment Analysis

Hệ thống phân tích ý kiến người dùng qua văn bản và hình ảnh sử dụng kiến trúc Vision-Language Model.

## Kiến Trúc

### Tổng quan hệ thống

```
                         MULTIMODAL SENTIMENT ANALYSIS SYSTEM
 
                       [ Embeddings ]-- ( Word-pieces )-->[ de-Tokenizer ]
                            ^                                    |
                            |                                    v
                     ( Lang tokens )                      (( Output Text ))
                            ^
                            |
    +-------------------------------------------------------+
    |                         LLM                           |
    +-------------------------------------------------------+
         ^                               ^
         |                               |
   (Lang tokens)                   (Lang tokens)
         |                               |
    [ Projection ]                [  Embeddings  ]
         ^                               ^
    (Img tokens)                   (word-pieces)
         |                               |
    [    ViT     ]                [  Tokenizer   ]
         ^                               ^
   (14px patches)                        |
         |                               |
    [Preprocessor]                       |
         ^                               |
         |                               |
  ((   Image   ))                 (( Input text  ))
```

### Chi tiết Inference Pipeline

```
                           INFERENCE WITH VLM (InternVL3-8B)
    ┌────────────────────────────────────────────────────────────────────────────┐
    │                                                                            │
    │   IMAGE INPUT              VISION ENCODER           MLP PROJECTOR          │
    │   ┌─────────┐             ┌─────────────┐          ┌─────────────┐        │
    │   │ 512×512 │   Resize    │  ViT-L/16   │  Project │   Linear    │        │
    │   │   RGB   │ ─────────▶  │  (Frozen)   │ ───────▶ │ 1024→4096   │        │
    │   └─────────┘   224×224   │   307M      │          │   + GELU    │        │
    │                           └─────────────┘          └──────┬──────┘        │
    │                                                           │               │
    │   [B, 3, 224, 224]        [B, 197, 1024]           [B, 197, 4096]         │
    │                                                           │               │
    │                                                           ▼               │
    │                    ┌──────────────────────────────────────────────────┐   │
    │                    │              InternVL3-8B (VLM)                  │   │
    │                    │  ┌────────────────────────────────────────────┐ │   │
    │                    │  │         Visual Tokens (from MLP)           │ │   │
    │                    │  │              [197, 4096]                   │ │   │
    │                    │  │                   +                        │ │   │
    │                    │  │         Text Tokens (Prompt)               │ │   │
    │                    │  │   "Phân tích cảm xúc của hình ảnh này"     │ │   │
    │                    │  └────────────────────────────────────────────┘ │   │
    │                    │                      │                          │   │
    │                    │                      ▼                          │   │
    │                    │  ┌────────────────────────────────────────────┐ │   │
    │                    │  │           LLM Decoder (8B params)          │ │   │
    │                    │  │         Autoregressive Generation          │ │   │
    │                    │  └────────────────────────────────────────────┘ │   │
    │                    └──────────────────────────────────────────────────┘   │
    │                                           │                               │
    │                                           ▼                               │
    │                              ┌─────────────────────────┐                 │
    │                              │    Generated Text       │                 │
    │                              │  "Positive sentiment    │                 │
    │                              │   about Room#Service"   │                 │
    │                              └─────────────────────────┘                 │
    └────────────────────────────────────────────────────────────────────────────┘
```

### Chi tiết Vision Encoder (ViT-L/16)

```
                              VISION TRANSFORMER LARGE
    ┌────────────────────────────────────────────────────────────────────────┐
    │                                                                        │
    │   INPUT IMAGE                    PATCH EMBEDDING                       │
    │   ┌─────────┐                   ┌─────────────────┐                   │
    │   │ 224×224 │    Split into     │  196 patches    │                   │
    │   │   RGB   │ ───────────────▶  │   (14 × 14)     │                   │
    │   └─────────┘    16×16 patches  │  + 1 [CLS] token│                   │
    │                                 └────────┬────────┘                   │
    │                                          │                             │
    │                                          ▼                             │
    │   ┌──────────────────────────────────────────────────────────────┐    │
    │   │              TRANSFORMER ENCODER (24 Layers)                 │    │
    │   │  ┌─────────┐   ┌─────────┐   ┌─────────┐       ┌─────────┐  │    │
    │   │  │ Layer 1 │──▶│ Layer 2 │──▶│ Layer 3 │──...──│Layer 24 │  │    │
    │   │  │ 1024-d  │   │ 1024-d  │   │ 1024-d  │       │ 1024-d  │  │    │
    │   │  └─────────┘   └─────────┘   └─────────┘       └─────────┘  │    │
    │   │     │              │              │                 │       │    │
    │   │     ▼              ▼              ▼                 ▼       │    │
    │   │  ┌─────┐        ┌─────┐        ┌─────┐           ┌─────┐   │    │
    │   │  │MHSA │        │MHSA │        │MHSA │           │MHSA │   │    │
    │   │  │16-h │        │16-h │        │16-h │           │16-h │   │    │
    │   │  └─────┘        └─────┘        └─────┘           └─────┘   │    │
    │   └──────────────────────────────────────────────────────────────┘    │
    │                                          │                             │
    │                                          ▼                             │
    │                              ┌─────────────────────┐                  │
    │                              │  OUTPUT: [197, 1024]│                  │
    │                              │  (197 tokens × 1024)│                  │
    │                              └─────────────────────┘                  │
    └────────────────────────────────────────────────────────────────────────┘
    
    MHSA = Multi-Head Self-Attention (16 heads × 64 dim = 1024)
```

### Chi tiết MLP Projector (cho VLM)

```
                         MLP PROJECTOR (Vision-Language Bridge)
    ┌────────────────────────────────────────────────────────────────┐
    │                                                                │
    │   PURPOSE: Map visual features to VLM embedding space          │
    │                                                                │
    │   INPUT: [B, 197, 1024]  (from ViT-L/16)                      │
    │          └──────┬──────┘                                      │
    │                 │                                              │
    │                 ▼                                              │
    │   ┌─────────────────────────────┐                             │
    │   │      Linear Layer 1         │                             │
    │   │   1024 ──────────▶ 4096     │   (~4.2M params)            │
    │   └──────────────┬──────────────┘                             │
    │                  │                                             │
    │                  ▼                                             │
    │   ┌─────────────────────────────┐                             │
    │   │           GELU              │   (Activation)              │
    │   └──────────────┬──────────────┘                             │
    │                  │                                             │
    │                  ▼                                             │
    │   ┌─────────────────────────────┐                             │
    │   │      Linear Layer 2         │                             │
    │   │   4096 ──────────▶ 4096     │   (~16.8M params)           │
    │   └──────────────┬──────────────┘                             │
    │                  │                                             │
    │                  ▼                                             │
    │   OUTPUT: [B, 197, 4096]  ─────▶  InternVL3-8B as visual tokens│
    │                                                                │
    │   Note: 4096 = embedding dimension of InternVL3-8B            │
    └────────────────────────────────────────────────────────────────┘
```

### Chi tiết Training Pipeline (Classifier)

```
                         TRAINING PIPELINE (without VLM)
    ┌────────────────────────────────────────────────────────────────────────────┐
    │                                                                            │
    │   Mục đích: Train classifier để phân loại 18 labels mà không cần VLM      │
    │                                                                            │
    │   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                 │
    │   │  ViT-L/16   │     │ Mean Pooling│     │ Classifier  │                 │
    │   │  (FROZEN)   │────▶│   Layer     │────▶│    Head     │                 │
    │   │   307M      │     │             │     │ (TRAINABLE) │                 │
    │   └─────────────┘     └─────────────┘     └──────┬──────┘                 │
    │                                                  │                         │
    │   [B, 197, 1024]       [B, 1024]                [B, 18]                   │
    │                                                  │                         │
    │                                                  ▼                         │
    │                                                                            │
    │   CLASSIFIER HEAD DETAIL:                                                  │
    │   ┌────────────────────────────────────────────────────────────────────┐  │
    │   │                                                                    │  │
    │   │   INPUT: [B, 1024] (pooled ViT features)                          │  │
    │   │          └──────┬──────┘                                          │  │
    │   │                 │                                                  │  │
    │   │                 ▼                                                  │  │
    │   │   ┌─────────────────────────────┐                                 │  │
    │   │   │        Dropout(0.3)         │                                 │  │
    │   │   └──────────────┬──────────────┘                                 │  │
    │   │                  │                                                 │  │
    │   │                  ▼                                                 │  │
    │   │   ┌─────────────────────────────┐                                 │  │
    │   │   │      Linear Layer 1         │                                 │  │
    │   │   │   1024 ──────────▶ 512      │   (~524K params)                │  │
    │   │   └──────────────┬──────────────┘                                 │  │
    │   │                  │                                                 │  │
    │   │                  ▼                                                 │  │
    │   │   ┌─────────────────────────────┐                                 │  │
    │   │   │           ReLU              │                                 │  │
    │   │   └──────────────┬──────────────┘                                 │  │
    │   │                  │                                                 │  │
    │   │                  ▼                                                 │  │
    │   │   ┌─────────────────────────────┐                                 │  │
    │   │   │        Dropout(0.3)         │                                 │  │
    │   │   └──────────────┬──────────────┘                                 │  │
    │   │                  │                                                 │  │
    │   │                  ▼                                                 │  │
    │   │   ┌─────────────────────────────┐                                 │  │
    │   │   │      Linear Layer 2         │                                 │  │
    │   │   │    512 ──────────▶ 18       │   (~9K params)                  │  │
    │   │   └──────────────┬──────────────┘                                 │  │
    │   │                  │                                                 │  │
    │   │                  ▼                                                 │  │
    │   │   OUTPUT: [B, 18]  (logits for BCEWithLogitsLoss)                 │  │
    │   └────────────────────────────────────────────────────────────────────┘  │
    │                                                                            │
    └────────────────────────────────────────────────────────────────────────────┘
```

### So sánh 2 Pipeline

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE COMPARISON                                   │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   INFERENCE PIPELINE                    TRAINING PIPELINE                     │
│   ┌───────────────────┐                ┌───────────────────┐                 │
│   │ ViT → MLP → VLM   │                │ ViT → Pool → Cls  │                 │
│   └───────────────────┘                └───────────────────┘                 │
│                                                                               │
│   Components:                          Components:                            │
│   • ViT-L/16 (307M)                   • ViT-L/16 (307M)                      │
│   • MLP Projector (~21M)              • Mean Pooling                         │
│   • InternVL3-8B (~8B)                • Classifier Head (~533K)              │
│                                                                               │
│   Output:                              Output:                                │
│   • Free-form text                    • 18 label probabilities               │
│   • Natural language                  • Multi-hot vector                     │
│                                                                               │
│   Use case:                            Use case:                              │
│   • Explainable sentiment             • Fast classification                  │
│   • Detailed reasoning                • Batch processing                     │
│   • Requires GPU 24GB+                • Requires GPU 8GB+                    │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Labels (18 classes = 6 Aspects × 3 Sentiments)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              LABEL STRUCTURE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ASPECTS (6)                    SENTIMENTS (3)                             │
│   ┌─────────────────┐           ┌─────────────────┐                        │
│   │ Facilities      │     ×     │ Positive        │                        │
│   │ Food            │           │ Negative        │     =  18 Labels       │
│   │ Location        │           │ Neutral         │                        │
│   │ Public_area     │           └─────────────────┘                        │
│   │ Room            │                                                       │
│   │ Service         │                                                       │
│   └─────────────────┘                                                       │
│                                                                             │
│   LABELS:                                                                   │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │ 0: Facilities#Positive   │  6: Food#Positive      │ 12: Room#Pos   │   │
│   │ 1: Facilities#Negative   │  7: Food#Negative      │ 13: Room#Neg   │   │
│   │ 2: Facilities#Neutral    │  8: Food#Neutral       │ 14: Room#Neu   │   │
│   │ 3: Location#Positive     │  9: Public_area#Pos    │ 15: Service#Pos│   │
│   │ 4: Location#Negative     │ 10: Public_area#Neg    │ 16: Service#Neg│   │
│   │ 5: Location#Neutral      │ 11: Public_area#Neu    │ 17: Service#Neu│   │
│   └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Training Configuration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TRAINING CONFIGURATION                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   OPTIMIZER                          SCHEDULER                              │
│   ┌─────────────────────┐           ┌─────────────────────┐                │
│   │ AdamW               │           │ CosineAnnealingLR   │                │
│   │ lr = 1e-4           │           │ T_max = total_steps │                │
│   │ weight_decay = 0.01 │           └─────────────────────┘                │
│   └─────────────────────┘                                                   │
│                                                                             │
│   LOSS FUNCTION                      HYPERPARAMETERS                        │
│   ┌─────────────────────┐           ┌─────────────────────┐                │
│   │ BCEWithLogitsLoss   │           │ Batch Size: 4       │                │
│   │ (Multi-label)       │           │ Epochs: 10          │                │
│   └─────────────────────┘           │ Grad Clip: 1.0      │                │
│                                      └─────────────────────┘                │
│                                                                             │
│   TRAINABLE PARAMETERS (Training Pipeline)                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ Component          │ Parameters  │ Trainable │ Status              │  │
│   ├─────────────────────────────────────────────────────────────────────┤  │
│   │ ViT-L/16           │ ~307M       │ No        │ FROZEN              │  │
│   │ Classifier Head    │ ~533K       │ Yes       │ TRAINABLE           │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Cài Đặt

```bash
# Clone repo
cd e:/GitHub/a

# Tạo virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

## Sử Dụng

### 1. Chạy Web Interface (Gradio)

```bash
python app.py
```

Mở browser tại `http://localhost:7860`

### 2. Sử dụng trong code

```python
from src.sentiment_analyzer import MultimodalSentimentAnalyzer
from PIL import Image

# Load analyzer
analyzer = MultimodalSentimentAnalyzer(
    model_name="OpenGVLab/InternVL3-8B-Instruct",
    device="cuda"
)

# Load image
image = Image.open("path/to/image.jpg")

# Analyze
result = analyzer.analyze(image, language="vi")

print(f"Sentiment: {result.sentiment}")
print(f"Confidence: {result.confidence}")
print(f"Reasoning: {result.reasoning}")
```

## Yêu Cầu Phần Cứng

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 18GB | 24GB+ |
| RAM | 16GB | 32GB |
| Storage | 20GB | 50GB |

## Cấu Trúc Project

```
├── app.py                    # Gradio web interface
├── requirements.txt          # Dependencies
├── Multimodal_Sentiment_Analysis.ipynb  # Training notebook
├── config/
│   └── model_config.yaml     # Model configuration
├── datasets/
│   ├── train.json            # Training data
│   ├── dev.json              # Validation data
│   ├── test.json             # Test data
│   └── image/                # Image folder
├── src/
│   ├── __init__.py
│   ├── vision_encoder.py     # Google ViT-L/16
│   ├── projection_layer.py   # MLP Bridge
│   ├── llm_inference.py      # InternVL3 integration
│   └── sentiment_analyzer.py # Main pipeline
└── README.md
```

## Dataset: ViMACSA

Vietnamese Multimodal Aspect Category Sentiment Analysis

| Split | Samples |
|-------|---------|
| Train | ~3000 |
| Dev | ~500 |
| Test | ~500 |

## License

MIT License
