# ViMACSA — Encode-Once, Aspect-Loop Architecture

This document describes the `MultimodalACSAModel` architecture implemented in `src/aspect_model.py`.

## Overview

- **Task**: Multimodal Aspect-Based Sentiment Analysis (ACSA)
- **Aspects**: 6 fixed aspects (Facilities, Public_area, Location, Food, Room, Service)
- **Classes**: 4 sentiment classes (0=Irrelative, 1=Negative, 2=Neutral, 3=Positive)
- **Vision backbone**: SigLIP2-Large (frozen)
- **LLM backbone**: Qwen/Qwen3-4B-Instruct-2507 (LoRA r=64)
- **Max**: 7 images (256x256), 256 text tokens

## Design Principles

1. **Encode once**: comment, all images, and all ROIs are encoded only once per forward pass
2. **Aspect loop**: shared retrieval/fusion/decoder modules process each aspect independently
3. **Explicit evidence tokens**: each aspect has explicit `[TXT_EVI]`, `[IMG_EVI]`, `[ROI_EVI]`, `[FUSE]` tokens
4. **Gated fusion**: text/image/ROI evidence are combined via learnable gates

## Module Architecture

```
+--------------------------------------------------------------+
| Input                                                         |
|  comment ─────────────────────────────────────────────────────┤
|  images [B, M, 3, 256, 256] ─────────────────────────────────┤
|  roi_data per image ──────────────────────────────────────────┤
+--------------------------------------------------------------+
                           |
                           | output:
                           | - H_txt [B, L_txt, 2560]
                           | - H_patch [B, M, P', 2560]
                           | - H_img_sum [B, M, 2560]
                           | - H_roi [B, M, K', 2560]
                           v
+--------------------------------------------------------------+
| ENCODING (all frozen)                                         |
|  comment ──► llm.embed_tokens() ─────► H_txt [B, L_txt, 2560] |
|                                                               |
| images ───► SigLIPEncoder ──► patch_feats [B, M, P, 1152]     |
|               │              img_summaries [B, M, 1152]       |
|               │                                                   |
|               └─► encode_roi(pixel_values, roi_data)            |
|                                       └─► roi_seq [B, M, K, 1152]
+--------------------------------------------------------------+
                           |
                           | output:
                           | - H_txt [B, L_txt, 2560]
                           | - H_patch [B, M, P', 2560]
                           | - H_img_sum [B, M, 2560]
                           | - H_roi [B, M, K', 2560]
                           v
+--------------------------------------------------------------+
| PROJECTION (trainable)                                        |
|  patch_feats ──► MLPProjector ──► H_patch [B, M, P', 2560]    |
|  roi_seq ───────► RoIProjector ──► H_roi [B, M, K', 2560]    |
|  img_summaries ─► Linear(1152→2560) ──► H_img_sum [B, M, 2560]
+--------------------------------------------------------------+
                           |
                           | output:
                           | - H_txt [B, L_txt, 2560]
                           | - H_patch [B, M, P', 2560]
                           | - H_img_sum [B, M, 2560]
                           | - H_roi [B, M, K', 2560]
                           | - img_mask [B, M]
                           | - roi_mask [B, M*K']
                           v
+--------------------------------------------------------------+
| ASPECT LOOP (shared weights, 6 iterations: a = 0..5)          |
|                                                               |
|  For each aspect a:                                           |
|                                                               |
|  aspect_queries[a] ────────────────────────────► ASP_a [B, 2560]
|                                                               |
|  TextRetriever (cross-attention, 8 heads):                   |
|  H_txt ─┬─────────────────────────────► TXT_EVI_a [B, 2560]  |
|         │  attention_mask [B, L_txt]                           |
|                                                               |
|  ImageRetriever (cross-attention, 4 heads):                   |
|  H_img_sum ─┬────────────────────────► IMG_EVI_a [B, 2560]   |
|              │  img_mask [B, M]                               |
|                                                               |
|  RoiRetriever (cross-attention, 4 heads):                     |
|  H_roi ──────┬────────────────────────► ROI_EVI_a [B, 2560] |
|              │  roi_mask [B, M*K']                             |
|                                                               |
|  GatedFusion:                                                 |
|  [TXT_EVI] ─┤                                                  |
|  [IMG_EVI] ──┼─► GatedFusion ──────► FUSE_a [B, 2560]        |
|  [ROI_EVI] ─┘                                                  |
|                                                               |
|  5-token sequence:                                             |
|  [ASP_a | TXT_EVI_a | IMG_EVI_a | ROI_EVI_a | FUSE_a]         |
|                                                               |
|  Shared LLM decoder:                                          |
|  [ASP]──┐                                                     |
|  [TXT]──┤                                                     |
|  [IMG]──┼─► Qwen3-4B + LoRA ──► last_hidden ──► Classification │
|  [ROI]──┤       (r=64)       [B, 5, 2560]      Head ──►     │
|  [FUSE]─┘                                    logits_a [B, 4] |
|                                                               |
|  logits stacked:                                              |
|  └────────────────────────────────► logits [B, 6, 4]        |
+--------------------------------------------------------------+
                           |
                           | output:
                           | - logits [B, 6, 4]
                           | - targets [B, 6]
                           | - loss (CrossEntropy)
                           v
+--------------------------------------------------------------+
| CrossEntropyLoss                                              |
|  loss = CE(logits [B*6, 4], targets [B*6])                   |
|  classes: 0=Irrelative, 1=Negative, 2=Neutral, 3=Positive   |
+--------------------------------------------------------------+
```

## Tensor Shapes

| Variable | Shape | Description |
|----------|-------|-------------|
| `H_txt` | `[B, L_txt, 2560]` | Text token embeddings |
| `patch_tokens` | `[B, M, P, 1152]` | SigLIP patch features |
| `img_summaries` | `[B, M, 1152]` | Per-image CLS token |
| `roi_seq` | `[B, M, K, 1152]` | Per-ROI features (token 0 = pooled image) |
| `H_patch` | `[B, M, P', 2560]` | Projected patches (P' <= P) |
| `H_roi` | `[B, M, K', 2560]` | Projected ROI (K' <= K) |
| `H_img_sum` | `[B, M, 2560]` | Projected image summaries |
| `q_a` | `[B, 2560]` | Aspect query for aspect a |
| `h_txt_a` | `[B, 2560]` | Retrieved text evidence |
| `h_img_a` | `[B, 2560]` | Retrieved image evidence |
| `h_roi_a` | `[B, 2560]` | Retrieved ROI evidence |
| `h_fuse_a` | `[B, 2560]` | Gated fusion result |
| `seq_a` | `[B, 5, 2560]` | Aspect-specific 5-token sequence |
| `logits_a` | `[B, 4]` | Per-aspect class logits |
| `logits` | `[B, 6, 4]` | Final stacked logits |

## Component Details

### AspectQuery (`src/attention.py`)
Trainable embedding table `[6, 2560]`. Initialized with Gaussian noise. Each row is the query vector for one aspect.

### TextRetriever (`src/attention.py`)
Cross-attention (8 heads) from aspect query to text token embeddings. Returns weighted text evidence vector.

### ImageRetriever (`src/attention.py`)
Cross-attention (4 heads) from aspect query to per-image summary vectors. Returns weighted image evidence + per-image relevance weights.

### RoiRetriever (`src/attention.py`)
Cross-attention (4 heads) from aspect query to flattened ROI features. Returns weighted ROI evidence + per-ROI relevance weights.

### GatedFusion (`src/attention.py`)
Three independent sigmoid gates over the concatenation `[h_txt; h_img; h_roi]`. Output is a residual connection.

### build_aspect_sequence (`src/aspect_sequence.py`)
Stacks 5 pre-computed vectors `[B, D_h]` into `[B, 5, D_h]`. Returns attention mask (all 1s) and position ids `[0, 1, 2, 3, 4]`.

### MultimodalACSAModel (`src/aspect_model.py`)
Full model class integrating all components. Loops 6 aspects with shared weights. Forward returns `{"logits", "targets", "loss"}`.

## Data Pipeline (unchanged)

```
Dataset (per sample):
  comment
  pixel_values [M_max, 3, 256, 256]  (zero-padded)
  roi_data per image: [{"boxes": [...], "labels": [...]}]
  aspect_labels: {aspect_idx: class_id}

Collate:
  pixel_values [B, M_max, 3, 256, 256]
  roi_data List[List[dict]]  (nested list, unchanged)

Model internally:
  img_mask = (pixel_values.sum(...) != 0)  [B, M]
  roi_mask = (H_roi.sum(dim=-1) != 0)      [B, M*K]
```

## What Changed vs Previous Architecture

| Aspect | Old (MultimodalSentimentModel) | New (MultimodalACSAModel) |
|--------|-------------------------------|---------------------------|
| Sequence | All tokens concat: `[ASP]*6 + img_patches + roi + text` | Per-aspect 5-token sequence |
| Text retrieval | Not explicit | Cross-attention from aspect query |
| Image retrieval | Flat concat into LLM | Cross-attention + relevance weights |
| ROI retrieval | Flat concat into LLM | Cross-attention + relevance weights |
| Fusion | None (LLM handles) | Learnable gated fusion |
| Aspect handling | Single LLM forward (all in one seq) | Loop 6x with shared decoder |
| Encoding | Once per sample | Once per sample |

## Training

Same training loop as before (see `training.py`):
- Optimizer: AdamW with 2 groups (LoRA lr=2e-5, other lr=3e-5)
- Scheduler: warmup + cosine
- Loss: CrossEntropy on logits `[B*6, 4]` vs targets `[B*6]`
- Early stopping: patience=4 on dev macro-F1

New trainable parameters (vs old model):
- `aspect_queries.embed.weight`: 6 × 2560 = 15,360
- `text_retriever.*`: 4 × Linear layers
- `img_retriever.*`: 4 × Linear layers
- `roi_retriever.*`: 4 × Linear layers
- `gated_fusion.*`: 3 × Linear layers + norm + bias

## Inference

Same inference API (see `inference.py`):
- `predict_all_aspects(model, tokenizer, comment, image_paths, roi_data=None)`
- Returns dict: aspect_name → `{prediction, prediction_id, confidence, probabilities}`
