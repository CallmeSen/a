```
+---------------------------+              +--------------------------------+
|      Image Input          |              |   Text Input + Aspect Prompt   |
+---------------------------+              +--------------------------------+
              |                                          |
              v                                          v
+---------------------------+              +--------------------------------+
| Image Preprocessing       |              | Tokenize                      |
| (resize, normalize)      |              +--------------------------------+
+---------------------------+                                       |
              |                                                   v
              v                               +--------------------------------+
+---------------------------+              |  InternLM tok_embeddings       |
|        SwinV2              |              |  (pretrained, frozen)          |
|  Vision Encoder (frozen)   |              +--------------------------------+
+---------------------------+                                        |
              |                                                    |
              v                                                    |
+---------------------------+                                        |
|     Image Tokens          |                                        |
|  [B, N_patches, 1024]    |                                        |
+---------------------------+                                        |
              |                                                    |
              v                                                    |
+---------------------------+                                        |
|    MLP Projector          |                                        |
|  (trainable, 1024->2048)  |                                        |
+---------------------------+                                        |
              |                                                    |
              v                                                    |
+---------------------------+                                        |
|   PerceiverResampler       |                                        |
| (trainable, compresses      |                                        |
|  N_patches -> K query tokens)                                      |
|  Cross-attn + FFN + LN    |                                        |
+---------------------------+                                        |
              |                                                    |
              v                                                    v
+---------------------------+  +----------------------------------------+
|  Visual Query Tokens       |  |         InternLM Backbone            |
|  [B, K, 2048]            |  |   InternLM2Model (frozen decoder)     |
+---------------------------+  |   num_layers = 24                     |
                               |                                        |
                               |  At EACH layer:                        |
                               |   1. Self-attention (frozen)           |
                               |      text_hidden = layer.self_attn      |
                               |                    (pretrained QKV)   |
                               |   2. GatedCrossAttentionAdapter (TRAINABLE)
                               |      Q = text_hidden (current layer)
                               |      K,V = visual_query_tokens
                               |      output = gate * cross_attn_output
                               |      text_hidden = LN(text + output)   |
                               |   3. FFN (frozen, SwiGLU)              |
                               |      text_hidden = layer.ffn_norm(      |
                               |        text_hidden + layer.ffn)        |
                               +----------------------------------------+
                                             |
                                             v
                               +----------------------------------------+
                               |  Final Hidden States                   |
                               |  [B, L_text, 2048]                     |
                               +----------------------------------------+
                                             |
                                             v
                    +------------------------------------------------------------+
                    |        Aspect Representation Extraction                    |
                    |   lấy h_asp từ token aspect / <ASP> token / span aspect    |
                    +------------------------------------------------------------+
                                             |
                                             | q = h_asp
                                             | K,V = H
                                             v
                               +----------------------------------------+
                               |  Aspect-Guided Attention Pooling        |
                               |  aspect_queries attend over final_hidden|
                               |  output = classifier(concat(q, attn))   |
                               +----------------------------------------+
                                             |
                                             v
                               +----------------------------------------+
                               |  MLP / Linear Classifier               |
                               |  Linear(2048 -> 4)                    |
                               +----------------------------------------+
                                             |
                                             v
                               +----------------------------------------+
                               |   Softmax Output                       |
                               |   [B, 6, 4] — None/Neg/Neutral/Positive|
                               +----------------------------------------+

Ý nghĩa đúng của khối pooling này
H = [h1, h2, ..., hN] là toàn bộ hidden states cuối của InternLM
h_asp là biểu diễn của aspect token
dùng h_asp làm query
dùng toàn bộ H làm key, value
attention sẽ chọn ra những token trong câu liên quan nhất tới aspect
output z là vector đã aspect-aware, rồi mới đưa vào classifier
Công thức ngắn
q = h_asp
K = H
V = H

alpha = softmax(qK^T / sqrt(d))
z = alphaV
Luồng tư duy của kiến trúc này
Aspect ở input: để InternLM hiểu câu dưới điều kiện aspect
Aspect ở output pooling: để rút đúng phần thông tin liên quan aspect
Ảnh không fusion trước LLM, mà đi vào InternLM qua cross-attention adapter
```
