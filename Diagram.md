# Kiến trúc hiện tại của hệ thống

Tài liệu này được cập nhật theo code thực tế trong `src/` và luồng khởi tạo trong `multimodal_sentiment.py`.

## 1. Tổng quan end-to-end

Hệ thống hiện tại là mô hình multimodal sentiment analysis theo từng aspect.

- Input text luôn được đổi thành dạng: `"<ASP>{aspect}</ASP> {comment}"`
- Mỗi sample gốc được flatten thành nhiều sample con, mỗi sample con ứng với 1 aspect trong 6 aspect:
  `Facilities`, `Public_area`, `Location`, `Food`, `Room`, `Service`
- Mỗi lần forward chỉ dự đoán sentiment cho 1 aspect:
  `None`, `Negative`, `Neutral`, `Positive`
- Ảnh được encode bằng `SwinV2`
- Text backbone là `Qwen/Qwen2.5-7B-Instruct`
- Fusion ảnh-văn bản được thực hiện bằng `GatedCrossAttentionAdapter` chèn vào các layer cuối của Qwen qua forward hook

## 2. Sơ đồ kiến trúc mô hình

```text
+-----------------------------+         +--------------------------------------+
| Raw sample                  |         | Raw sample                            |
| - list_img                  |         | - comment                             |
| - comment                   |         | - selected aspect                     |
| - text_img_label            |         |                                      |
+-----------------------------+         +--------------------------------------+
               |                                           |
               v                                           v
+-----------------------------+         +--------------------------------------+
| SentimentDataset            |         | make_collate_fn + tokenizer          |
| flatten mỗi sample thành    |         | text = "<ASP>{aspect}</ASP> comment" |
| 1 record / aspect           |         | padding/truncation                   |
+-----------------------------+         +--------------------------------------+
               |                                           |
               | pixel_values [B, M, 3, H, W]              | input_ids [B, L]
               | image_counts [B]                          | attention_mask [B, L]
               v                                           v
+-----------------------------+         +--------------------------------------+
| VisionEncoder               |         | Qwen embed_tokens                    |
| src/vit_transformer.py      |         | src/llm_factory.py / QwenLMWrapper   |
| SwinV2-Base                 |         | special tokens: <ASP>, </ASP>        |
| trainable                   |         +--------------------------------------+
+-----------------------------+                             |
               | image tokens [B*M, P, 1024]               |
               v                                            |
+-----------------------------+                             |
| MLPProjector                |                             |
| src/projector_layer.py      |                             |
| 1024 -> 3584                |                             |
+-----------------------------+                             |
               | projected tokens [B*M, P, 3584]           |
               v                                            |
+-----------------------------+                             |
| PerceiverResampler          |                             |
| src/perceiver_resampler.py  |                             |
| mỗi ảnh -> 64 query tokens  |                             |
+-----------------------------+                             |
               |                                            |
               | visual tokens [B, M*64, 3584]             |
               | visual mask theo image_counts             |
               +--------------------------+-----------------+
                                          |
                                          v
                       +-----------------------------------------------+
                       | QwenLMWrapper                                 |
                       | src/qwen_wrapper.py                           |
                       | Qwen2.5-7B-Instruct backbone                  |
                       | backbone frozen                               |
                       | adapters gắn ở 4 decoder layers cuối          |
                       +-----------------------------------------------+
                                          |
                                          | tại mỗi layer được chọn:
                                          | 1. self-attention của Qwen
                                          | 2. hook inject GatedCrossAttentionAdapter
                                          |    - Q = text hidden states
                                          |    - K,V = visual tokens
                                          |    - output = text + sigmoid(gate) * cross_attn
                                          | 3. đi tiếp vào phần còn lại của decoder layer
                                          v
                       +-----------------------------------------------+
                       | final_hidden [B, L, 3584]                    |
                       +-----------------------------------------------+
                                          |
                                          v
                       +-----------------------------------------------+
                       | Aspect span extraction                        |
                       | src/multimodal_sentiment_model.py             |
                       | tìm span giữa <ASP> ... </ASP>                |
                       | mean-pooling -> h_a [B, 3584]                 |
                       +-----------------------------------------------+
                                          |
                                          v
                       +-----------------------------------------------+
                       | Aspect-guided attention pooling               |
                       | scores = h_a @ H^T / sqrt(d)                  |
                       | z_a = softmax(scores) @ H                     |
                       +-----------------------------------------------+
                                          |
                                          v
                       +-----------------------------------------------+
                       | classifier_head                               |
                       | Linear(3584 -> 4)                             |
                       +-----------------------------------------------+
                                          |
                                          v
                       +-----------------------------------------------+
                       | logits [B, 1, 1, 4]                           |
                       | classes: None / Negative / Neutral / Positive |
                       +-----------------------------------------------+
```

## 3. Luồng khởi tạo trong `multimodal_sentiment.py`

```text
setup_runtime()
  -> load_all_splits()
  -> compute_class_weights()
  -> build_tokenizer_and_llm()
  -> optional apply_lora_to_llm()
  -> VisionEncoder()
  -> MLPProjector()
  -> PerceiverResampler(num_queries=64)
  -> QwenLMWrapper(use_adapter_layers=last_4_layers)
  -> MultimodalSentimentModel(...)
  -> optional MultitaskSentimentModel(...)  # wrapper tương thích ngược
  -> build train/dev/test dataloaders
  -> setup_optimizer()
  -> LazyLambdaScheduler(warmup + cosine decay)
  -> train_epoch() / validate()
  -> save best trainable weights
  -> test evaluation
  -> demo predict_aspect_sentiment()
```

## 4. Chi tiết từng khối

### 4.1 Dữ liệu và batching

Nguồn chính: `src/data.py`

- `load_all_splits()` đọc `datasets/train.json`, `dev.json`, `test.json`
- `SentimentDataset` bỏ sample không có ảnh hợp lệ hoặc không có label hợp lệ
- Mỗi sample gốc được expand thành 6 sample con, tương ứng 6 aspect
- Nếu aspect xuất hiện trong `text_img_label` thì label là `Negative/Neutral/Positive`
- Nếu aspect không xuất hiện thì label là `None`
- `make_collate_fn()`:
  - pad số lượng ảnh theo batch
  - tạo `image_counts`
  - build text dạng `"<ASP>{aspect}</ASP> {comment}"`
  - tokenize với `MAX_TEXT_LENGTH=256`

Kết quả batch:

- `pixel_values`: `[B, M, 3, 256, 256]`
- `image_counts`: `[B]`
- `input_ids`: `[B, L]`
- `attention_mask`: `[B, L]`
- `labels`: `[B]`

### 4.2 Nhánh ảnh

Nguồn chính: `src/vit_transformer.py`, `src/projector_layer.py`, `src/perceiver_resampler.py`

```text
pixel_values [B, M, 3, H, W]
  -> reshape thành [B*M, 3, H, W]
  -> SwinV2 last_hidden_state         : [B*M, P, 1024]
  -> MLPProjector                     : [B*M, P, 3584]
  -> PerceiverResampler (64 queries)  : [B*M, 64, 3584]
  -> reshape + concat theo số ảnh     : [B, M*64, 3584]
  -> visual_mask                      : [B, M*64]
```

Lưu ý:

- `VisionEncoder` hiện đang để `requires_grad=True` cho toàn bộ SwinV2
- `PerceiverResampler` nén patch tokens của từng ảnh về số token cố định là 64
- `MAX_IMAGES=5`, nên số visual token tối đa mỗi sample là `5 * 64 = 320`

### 4.3 Nhánh text và fusion với ảnh

Nguồn chính: `src/llm_factory.py`, `src/qwen_wrapper.py`

- Tokenizer thêm 2 special token:
  - `<ASP>`
  - `</ASP>`
- LLM backbone là `Qwen/Qwen2.5-7B-Instruct`
- `QwenLMWrapper` không thay toàn bộ decoder layer
- Thay vào đó wrapper dùng `register_forward_hook()` trên `self_attn` của các layer cuối
- Các layer được dùng adapter mặc định là `range(num_layers - 4, num_layers)`

`GatedCrossAttentionAdapter` thực hiện:

```text
text_hidden --RMSNorm--> Q
visual_tokens ---------> K, V
cross_attn = softmax(QK^T / sqrt(d)) V
output = text_hidden + sigmoid(gate) * cross_attn
```

Ý nghĩa:

- Text vẫn đi theo backbone Qwen pretrained
- Ảnh không được nối trực tiếp vào chuỗi token text
- Ảnh được bơm vào hidden states của text ở các layer cuối thông qua cross-attention có gate học được

### 4.4 Aspect-aware pooling và classifier

Nguồn chính: `src/multimodal_sentiment_model.py`

Sau khi lấy được `final_hidden = H`:

1. Xác định vị trí token `<ASP>` và `</ASP>`
2. Mean-pool hidden states trong span đó để lấy `h_a`
3. Dùng `h_a` làm query để attention lên toàn bộ `H`
4. Lấy `z_a` làm biểu diễn cuối cùng cho aspect
5. Dùng `classifier_head` để phân loại 4 lớp

Công thức:

```text
h_a = mean(H[start:end])
scores = h_a H^T / sqrt(d)
alpha = softmax(scores)
z_a = alpha H
logits = Linear(z_a)
```

Điểm quan trọng:

- Đây là kiến trúc dự đoán theo từng aspect
- Không còn đầu ra `[B, 6, 4]` trong một forward
- Đầu ra hiện tại là `[B, 1, 1, 4]` để tương thích với code loss hiện có

## 5. Thành phần trainable và frozen

Mặc định hiện tại:

- `SwinV2` trong `VisionEncoder`: trainable
- `MLPProjector`: trainable
- `PerceiverResampler`: trainable
- `GatedCrossAttentionAdapter` trong `QwenLMWrapper`: trainable
- `classifier_head`: trainable
- `Qwen backbone`: frozen

Nếu `USE_LORA=1`:

- áp dụng LoRA vào `q_proj`, `v_proj`, `o_proj` của Qwen
- backbone gốc vẫn frozen
- LoRA parameters trở thành trainable
- hook detach trong wrapper được vô hiệu hóa để gradient đi qua các nhánh LoRA

## 6. Training và đánh giá

Nguồn chính: `src/training.py`

- Optimizer: `AdamW`
- Có thể tách learning rate riêng cho:
  - nhóm vision
  - nhóm LoRA
  - các tham số trainable còn lại
- Scheduler: `LazyLambdaScheduler`
  - warmup
  - cosine decay
- Loss chính:
  - `focal_loss_with_smoothing()`
  - trong `compute_loss()` hiện đang chạy với:
    - `alpha=1.0`
    - `gamma=1.0`
    - `label_smoothing=0.0`
- `WeightedRandomSampler` là tùy chọn qua `USE_WEIGHTED_SAMPLER=1`
- Metric chính khi validate/test: `macro_f1`

## 7. Inference

Nguồn chính: `src/inference.py`

`predict_aspect_sentiment(image_path, comment, aspect_name, model, tokenizer)`:

```text
1. build text: "<ASP>{aspect}</ASP> {comment}"
2. preprocess 1 ảnh
3. forward qua model
4. squeeze logits [1,1,1,4] -> [1,4]
5. softmax -> xác suất 4 lớp
```

## 8. Mapping file -> trách nhiệm

- `multimodal_sentiment.py`: entrypoint train/test/demo inference
- `src/config.py`: cấu hình, label space, path, hyperparameters
- `src/data.py`: dataset, transform, collate, sampler
- `src/llm_factory.py`: tokenizer + Qwen loader + special tokens
- `src/vit_transformer.py`: SwinV2 vision encoder
- `src/projector_layer.py`: projector từ vision dim sang LLM dim
- `src/perceiver_resampler.py`: nén patch tokens thành visual query tokens
- `src/qwen_wrapper.py`: hook-based visual fusion trên Qwen
- `src/multimodal_sentiment_model.py`: forward multimodal hoàn chỉnh
- `src/lora_layers.py`: LoRA cho Qwen
- `src/multitask_model.py`: wrapper tương thích ngược
- `src/training.py`: optimizer, scheduler, loss, train/validate
- `src/inference.py`: tiện ích inference theo từng aspect

## 9. Khác biệt chính so với sơ đồ cũ

Sơ đồ cũ trong file này không còn đúng ở các điểm sau:

- Text backbone không còn là `InternLM`, mà là `Qwen2.5-7B-Instruct`
- Hidden size không còn là `2048`, mà là `3584`
- `VisionEncoder` không frozen; hiện tại SwinV2 đang được fine-tune
- Fusion không diễn ra ở mọi layer, mà chỉ ở 4 layer cuối của Qwen
- Mô hình không dự đoán đồng thời 6 aspect trong một output tensor
- Output hiện tại là phân loại sentiment cho 1 aspect mỗi forward