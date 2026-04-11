root@71d3261521137a80:~/a# USE_LORA=1 USE_MULTITASK=1 USE_WEIGHTED_SAMPLER=1 USE_VCE=1 USE_DUAL_ADAPTER=1 USE_CONTRASTIVE_LOSS=1 USE_ASPECT_ATTN=1 python multimodal_sentiment.py
The image processor of type `ViTImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`. 
Device: cuda
[INFO] Found existing checkpoint: output_model/best_model.safetensors
      Will resume from checkpoint after training.
Compute dtype: torch.bfloat16
Vision model: microsoft/swinv2-base-patch4-window8-256
LLM model:    Qwen/Qwen3-4B-Instruct-2507
Data dir:     datasets
Output dir:   output_model
Loaded 2876 samples from datasets/train.json
Loaded 1000 samples from datasets/dev.json
Loaded 1000 samples from datasets/test.json
[DATA] Class distribution: {'Negative': 830, 'Neutral': 1401, 'None': 8606, 'Positive': 6419}
[DATA] Class weights: [0.2122, 2.2, 1.3034, 0.2845]

Loading LLM: Qwen/Qwen3-4B-Instruct-2507
[INFO] Special tokens registered: <ASP>=151669, </ASP>=151670
Loading weights: 100%|████████████████████████████████████| 398/398 [00:00<00:00, 8584.28it/s]
[INFO] Tokenizer vocab size: 151671, embedding resized
[INFO] LLM base: Qwen3Model, num_layers=36, hidden_size=2560
Tokenizer: padding_side=right, pad_token=<|endoftext|> (id=151643)

[LoRA] Applying LoRA: r=64, alpha=128
[LoRA] Applying LoRA: r=64, alpha=128, targets=['q_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
[LoRA] Backbone frozen — only LoRA params train
[LoRA] Trainable params: 123,863,040 / 4,145,652,736 (2.99%)
[LoRA Summary] Trainable: 123,863,040 / 4,145,652,736 (2.99%)
trainable params: 123,863,040 || all params: 4,145,652,736 || trainable%: 2.9878
Loading Swin Transformer V2: microsoft/swinv2-base-patch4-window8-256 (torch_dtype=torch.float32)
Loading weights: 100%|███████████████████████████████████| 471/471 [00:00<00:00, 50856.41it/s]
Vision encoder: 471 params trainable (all stages)
Loaded: hidden_size=1024, num_patches=64
[Dual Adapter] Using DualGatedCrossAttentionAdapter (rank=64)
[VCE] MultiScaleVisualFusion enabled (64 tokens from 4 SwinV2 stages)
[MultimodalSentimentModel] Qwen backbone frozen — trainable params managed by PEFT (LoRA) or adapters
[Contrastive Loss] ArcFace head enabled
[Aspect Attention] Aspect-Guided Visual Attention enabled

[R8] Wrapping model with MultitaskSentimentModel
[VCE] MultiScaleVisualFusion enabled (64 tokens from 4 SwinV2 stages)
[MultimodalSentimentModel] Qwen backbone frozen — trainable params managed by PEFT (LoRA) or adapters
[Contrastive Loss] ArcFace head enabled
[Aspect Attention] Aspect-Guided Visual Attention enabled
[Multitask] Trainable params (723): LoRA + adapters + vision + projector + classifier
Total parameters: 4,338,498,871
Trainable parameters: 192,846,135
Prepared 17256 valid samples
[DATA] WeightedRandomSampler: 17256 samples (1 per aspect), minority_upsample_ratio=5.0
[DATA] Label distribution: {0: 8606, 1: 830, 2: 1401, 3: 6419}
Prepared 6000 valid samples
Prepared 6000 valid samples

Training config:
  LR: 5e-06
  Epochs: 15, Steps/Epoch: 1079, Total: 16185, Warmup: 1618
  Grad accumulation: 4
  Trainable params: 192,846,135 params across 723 tensors

[OK] Sanity forward: logits finite=True
     logits shape=torch.Size([4, 1, 1, 4])
     bad_batch=False

Epoch 1/15
Training:   0%|                                                      | 0/4314 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|██████████████████| 4314/4314 [38:49<00:00,  1.85it/s, loss=0.0697, cls=0.0697]
Validating: 100%|█████████████████████████████████████████| 1500/1500 [05:20<00:00,  4.68it/s]
Train Loss: 0.1939 (CLS=0.1939)
Val Loss:   0.0822 (CLS=0.0822)
Val Macro-F1: 0.1730
[SAVED] Best model macro-F1=0.1730

Epoch 2/15
Training:   0%|                                                      | 0/4314 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|██████████████████| 4314/4314 [39:37<00:00,  1.81it/s, loss=0.0577, cls=0.0577]
Validating: 100%|█████████████████████████████████████████| 1500/1500 [05:20<00:00,  4.67it/s]
Train Loss: 0.0883 (CLS=0.0883)
Val Loss:   0.0677 (CLS=0.0677)
Val Macro-F1: 0.2828
[SAVED] Best model macro-F1=0.2828

Epoch 3/15
Training:   0%|                                                      | 0/4314 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|██████████████████| 4314/4314 [39:02<00:00,  1.84it/s, loss=0.1380, cls=0.1380]
Validating: 100%|█████████████████████████████████████████| 1500/1500 [05:07<00:00,  4.88it/s]
Train Loss: 0.0762 (CLS=0.0762)
Val Loss:   0.0691 (CLS=0.0691)
Val Macro-F1: 0.2870
[SAVED] Best model macro-F1=0.2870

Epoch 4/15
Training:   0%|                                                      | 0/4314 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|██████████████████| 4314/4314 [38:11<00:00,  1.88it/s, loss=0.1574, cls=0.1574]
Validating: 100%|█████████████████████████████████████████| 1500/1500 [05:19<00:00,  4.70it/s]
Train Loss: 0.0681 (CLS=0.0681)
Val Loss:   0.0634 (CLS=0.0634)
Val Macro-F1: 0.3457
[SAVED] Best model macro-F1=0.3457

Epoch 5/15
Training:   0%|                                                      | 0/4314 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|██████████████████| 4314/4314 [36:48<00:00,  1.95it/s, loss=0.0314, cls=0.0314]
Validating: 100%|█████████████████████████████████████████| 1500/1500 [04:32<00:00,  5.50it/s]
Train Loss: 0.0603 (CLS=0.0603)
Val Loss:   0.0615 (CLS=0.0615)
Val Macro-F1: 0.3687
[SAVED] Best model macro-F1=0.3687

Epoch 6/15
Training:   0%|                                                      | 0/4314 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|██████████████████| 4314/4314 [36:16<00:00,  1.98it/s, loss=0.0702, cls=0.0702]
Validating: 100%|█████████████████████████████████████████| 1500/1500 [04:24<00:00,  5.66it/s]
Train Loss: 0.0549 (CLS=0.0549)
Val Loss:   0.0574 (CLS=0.0574)
Val Macro-F1: 0.4194
[SAVED] Best model macro-F1=0.4194

Epoch 7/15
Training:   0%|                                                      | 0/4314 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|██████████████████| 4314/4314 [36:00<00:00,  2.00it/s, loss=0.0451, cls=0.0451]
Validating: 100%|█████████████████████████████████████████| 1500/1500 [04:32<00:00,  5.50it/s]
Train Loss: 0.0483 (CLS=0.0483)
Val Loss:   0.0611 (CLS=0.0611)
Val Macro-F1: 0.4051

Epoch 8/15
Training:   0%|                                                      | 0/4314 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|██████████████████| 4314/4314 [36:29<00:00,  1.97it/s, loss=0.0393, cls=0.0393]
Validating: 100%|█████████████████████████████████████████| 1500/1500 [04:39<00:00,  5.37it/s]
Train Loss: 0.0449 (CLS=0.0449)
Val Loss:   0.0582 (CLS=0.0582)
Val Macro-F1: 0.4524
[SAVED] Best model macro-F1=0.4524

Epoch 9/15
Training:   0%|                                                      | 0/4314 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|██████████████████| 4314/4314 [37:24<00:00,  1.92it/s, loss=0.0152, cls=0.0152]
Validating: 100%|█████████████████████████████████████████| 1500/1500 [04:34<00:00,  5.47it/s]
Train Loss: 0.0414 (CLS=0.0414)
Val Loss:   0.0615 (CLS=0.0615)
Val Macro-F1: 0.4243

Epoch 10/15
Training:   0%|                                                      | 0/4314 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|██████████████████| 4314/4314 [37:25<00:00,  1.92it/s, loss=0.0258, cls=0.0258]
Validating: 100%|█████████████████████████████████████████| 1500/1500 [05:16<00:00,  4.73it/s]
Train Loss: 0.0389 (CLS=0.0389)
Val Loss:   0.0550 (CLS=0.0550)
Val Macro-F1: 0.4914
[SAVED] Best model macro-F1=0.4914

Epoch 11/15
Training:   0%|                                                      | 0/4314 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|██████████████████| 4314/4314 [39:11<00:00,  1.83it/s, loss=0.0091, cls=0.0091]
Validating: 100%|█████████████████████████████████████████| 1500/1500 [04:23<00:00,  5.68it/s]
Train Loss: 0.0370 (CLS=0.0370)
Val Loss:   0.0544 (CLS=0.0544)
Val Macro-F1: 0.4879

Epoch 12/15
Training:   0%|                                                      | 0/4314 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|██████████████████| 4314/4314 [36:22<00:00,  1.98it/s, loss=0.0170, cls=0.0170]
Validating: 100%|█████████████████████████████████████████| 1500/1500 [04:32<00:00,  5.50it/s]
Train Loss: 0.0351 (CLS=0.0351)
Val Loss:   0.0545 (CLS=0.0545)
Val Macro-F1: 0.4852

Epoch 13/15
Training:   0%|                                                      | 0/4314 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|██████████████████| 4314/4314 [38:10<00:00,  1.88it/s, loss=0.0240, cls=0.0240]
Validating: 100%|█████████████████████████████████████████| 1500/1500 [04:17<00:00,  5.83it/s]
Train Loss: 0.0344 (CLS=0.0344)
Val Loss:   0.0564 (CLS=0.0564)
Val Macro-F1: 0.4765
[EARLY STOP] triggered
Validating: 100%|█████████████████████████████████████████| 1500/1500 [04:23<00:00,  5.70it/s]

=== Test Set ===
Test Loss: 0.0618 (CLS=0.0618)
Test Macro-F1: 0.4767
Macro Precision: 0.5049
Macro Recall:   0.6002
Macro F1:       0.4767