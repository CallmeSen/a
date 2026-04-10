root@71d3261521137a80:~/a# USE_MULTITASK=1 USE_LORA=1 USE_WEIGHTED_SAMPLER=1 python multimodal_sentiment.py
The image processor of type `ViTImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`. 
Device: cuda
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
Loading weights: 100%|█████████████████████████████████| 398/398 [00:00<00:00, 8914.90it/s]
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
Loading weights: 100%|████████████████████████████████| 471/471 [00:00<00:00, 45469.59it/s]
Vision encoder: 471 params trainable (all stages)
Loaded: hidden_size=1024, num_patches=64
[MultimodalSentimentModel] Qwen backbone frozen — trainable params managed by PEFT (LoRA) or adapters

[R8] Wrapping model with MultitaskSentimentModel
[MultimodalSentimentModel] Qwen backbone frozen — trainable params managed by PEFT (LoRA) or adapters
[Multitask] Trainable params (599): LoRA + adapters + vision + projector + classifier
Total parameters: 4,785,920,782
Trainable parameters: 640,268,046
Prepared 17256 valid samples
[DATA] WeightedRandomSampler: 17256 samples (1 per aspect), minority_upsample_ratio=5.0
[DATA] Label distribution: {0: 8606, 1: 830, 2: 1401, 3: 6419}
Prepared 6000 valid samples
Prepared 6000 valid samples

Training config:
  LR: 2e-05
  Epochs: 15, Steps/Epoch: 1079, Total: 16185, Warmup: 809
  Grad accumulation: 2
  Trainable params: 640,268,046 params across 599 tensors

[OK] Sanity forward: logits finite=True
     logits shape=torch.Size([8, 1, 1, 4])
     bad_batch=False

Epoch 1/15
Training:   0%|                                                   | 0/2157 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|███████████████| 2157/2157 [34:02<00:00,  1.06it/s, loss=0.0938, cls=0.0938]
Validating: 100%|████████████████████████████████████████| 750/750 [03:35<00:00,  3.48it/s]
Train Loss: 0.1442 (CLS=0.1442)
Val Loss:   0.1193 (CLS=0.1193)
Val Macro-F1: 0.1173
[SAVED] Best model macro-F1=0.1173

Epoch 2/15
Training:   0%|                                                   | 0/2157 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|███████████████| 2157/2157 [33:56<00:00,  1.06it/s, loss=0.0639, cls=0.0639]
Validating: 100%|████████████████████████████████████████| 750/750 [03:29<00:00,  3.58it/s]
Train Loss: 0.0827 (CLS=0.0827)
Val Loss:   0.0877 (CLS=0.0877)
Val Macro-F1: 0.2371
[SAVED] Best model macro-F1=0.2371

Epoch 3/15
Training:   0%|                                                   | 0/2157 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|███████████████| 2157/2157 [33:29<00:00,  1.07it/s, loss=0.0296, cls=0.0296]
Validating: 100%|████████████████████████████████████████| 750/750 [02:53<00:00,  4.33it/s]
Train Loss: 0.0565 (CLS=0.0565)
Val Loss:   0.0947 (CLS=0.0947)
Val Macro-F1: 0.3121
[SAVED] Best model macro-F1=0.3121

Epoch 4/15
Training:   0%|                                                   | 0/2157 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|███████████████| 2157/2157 [32:10<00:00,  1.12it/s, loss=0.0297, cls=0.0297]
Validating: 100%|████████████████████████████████████████| 750/750 [03:01<00:00,  4.12it/s]
Train Loss: 0.0371 (CLS=0.0371)
Val Loss:   0.0941 (CLS=0.0941)
Val Macro-F1: 0.3788
[SAVED] Best model macro-F1=0.3788

Epoch 5/15
Training:   0%|                                                   | 0/2157 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|███████████████| 2157/2157 [32:37<00:00,  1.10it/s, loss=0.0173, cls=0.0173]
Validating: 100%|████████████████████████████████████████| 750/750 [02:59<00:00,  4.17it/s]
Train Loss: 0.0275 (CLS=0.0275)
Val Loss:   0.1095 (CLS=0.1095)
Val Macro-F1: 0.3907
[SAVED] Best model macro-F1=0.3907

Epoch 6/15
Training:   0%|                                                   | 0/2157 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|███████████████| 2157/2157 [33:07<00:00,  1.09it/s, loss=0.0077, cls=0.0077]
Validating: 100%|████████████████████████████████████████| 750/750 [03:22<00:00,  3.71it/s]
Train Loss: 0.0203 (CLS=0.0203)
Val Loss:   0.1207 (CLS=0.1207)
Val Macro-F1: 0.3946
[SAVED] Best model macro-F1=0.3946

Epoch 7/15
Training:   0%|                                                   | 0/2157 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|███████████████| 2157/2157 [33:04<00:00,  1.09it/s, loss=0.0108, cls=0.0108]
Validating: 100%|████████████████████████████████████████| 750/750 [03:03<00:00,  4.09it/s]
Train Loss: 0.0164 (CLS=0.0164)
Val Loss:   0.1337 (CLS=0.1337)
Val Macro-F1: 0.4225
[SAVED] Best model macro-F1=0.4225

Epoch 8/15
Training:   0%|                                                   | 0/2157 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|███████████████| 2157/2157 [32:59<00:00,  1.09it/s, loss=0.0051, cls=0.0051]
Validating: 100%|████████████████████████████████████████| 750/750 [03:21<00:00,  3.72it/s]
Train Loss: 0.0112 (CLS=0.0112)
Val Loss:   0.1491 (CLS=0.1491)
Val Macro-F1: 0.4224

Epoch 9/15
Training:   0%|                                                   | 0/2157 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|███████████████| 2157/2157 [33:56<00:00,  1.06it/s, loss=0.0067, cls=0.0067]
Validating: 100%|████████████████████████████████████████| 750/750 [03:14<00:00,  3.85it/s]
Train Loss: 0.0091 (CLS=0.0091)
Val Loss:   0.1529 (CLS=0.1529)
Val Macro-F1: 0.4420
[SAVED] Best model macro-F1=0.4420

Epoch 10/15
Training:   0%|                                                   | 0/2157 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|███████████████| 2157/2157 [34:31<00:00,  1.04it/s, loss=0.0029, cls=0.0029]
Validating: 100%|████████████████████████████████████████| 750/750 [03:28<00:00,  3.60it/s]
Train Loss: 0.0066 (CLS=0.0066)
Val Loss:   0.1536 (CLS=0.1536)
Val Macro-F1: 0.4321

Epoch 11/15
Training:   0%|                                                   | 0/2157 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|███████████████| 2157/2157 [34:26<00:00,  1.04it/s, loss=0.0044, cls=0.0044]
Validating: 100%|████████████████████████████████████████| 750/750 [03:35<00:00,  3.47it/s]
Train Loss: 0.0049 (CLS=0.0049)
Val Loss:   0.1753 (CLS=0.1753)
Val Macro-F1: 0.4440
[SAVED] Best model macro-F1=0.4440

Epoch 12/15
Training:   0%|                                                   | 0/2157 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|███████████████| 2157/2157 [33:00<00:00,  1.09it/s, loss=0.0037, cls=0.0037]
Validating: 100%|████████████████████████████████████████| 750/750 [03:30<00:00,  3.56it/s]
Train Loss: 0.0039 (CLS=0.0039)
Val Loss:   0.1845 (CLS=0.1845)
Val Macro-F1: 0.4344

Epoch 13/15
Training:   0%|                                                   | 0/2157 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|█████████████| 2157/2157 [1:19:54<00:00,  2.22s/it, loss=0.0072, cls=0.0072]
Validating: 100%|████████████████████████████████████████| 750/750 [03:18<00:00,  3.78it/s]
Train Loss: 0.0036 (CLS=0.0036)
Val Loss:   0.1925 (CLS=0.1925)
Val Macro-F1: 0.4427

Epoch 14/15
Training:   0%|                                                   | 0/2157 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|███████████████| 2157/2157 [32:46<00:00,  1.10it/s, loss=0.0012, cls=0.0012]
Validating: 100%|████████████████████████████████████████| 750/750 [03:20<00:00,  3.74it/s]
Train Loss: 0.0031 (CLS=0.0031)
Val Loss:   0.1987 (CLS=0.1987)
Val Macro-F1: 0.4362

Epoch 15/15
Training:   0%|                                                   | 0/2157 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training:  40%|██████▍         | 861/2157 [13:12<19:26,  1.11it/s, loss=0.0015, cls=0.0015]Training: 100%|█████████████| 2157/2157 [1:14:23<00:00,  2.07s/it, loss=0.0012, cls=0.0012]
Validating: 100%|████████████████████████████████████████| 750/750 [03:01<00:00,  4.14it/s]
Train Loss: 0.0031 (CLS=0.0031)
Val Loss:   0.1975 (CLS=0.1975)
Val Macro-F1: 0.4383
Validating: 100%|████████████████████████████████████████| 750/750 [02:58<00:00,  4.21it/s]

=== Test Set ===
Test Loss: 0.1884 (CLS=0.1884)
Test Macro-F1: 0.4460
Macro Precision: 0.4563
Macro Recall:   0.4395
Macro F1:       0.4460
Prepared 6000 valid samples