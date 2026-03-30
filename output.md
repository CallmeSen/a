root@71d3261521137a80:~/a# USE_LORA=1 USE_MULTITASK=1 python multimodal_sentiment.py
The image processor of type `ViTImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`. 
Device: cuda
[INFO] Found existing checkpoint: output_model/best_model.safetensors
      Will resume from checkpoint after training.
Compute dtype: torch.bfloat16
Vision model: microsoft/swinv2-base-patch4-window8-256
LLM model:    Qwen/Qwen2.5-7B-Instruct
Data dir:     datasets
Output dir:   output_model
Loaded 2876 samples from datasets/train.json
Loaded 1000 samples from datasets/dev.json
Loaded 1000 samples from datasets/test.json
[DATA] Class distribution: {'Negative': 830, 'Neutral': 1401, 'None': 8606, 'Positive': 6419}
[DATA] Class weights: [0.2122, 2.2, 1.3034, 0.2845]

Loading LLM: Qwen/Qwen2.5-7B-Instruct
[INFO] Special tokens registered: <ASP>=151665, </ASP>=151666
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 339/339 [00:00<00:00, 8061.40it/s]
[INFO] Tokenizer vocab size: 151667, embedding resized
[INFO] LLM base: Qwen2Model, num_layers=28, hidden_size=3584
Tokenizer: padding_side=right, pad_token=<|endoftext|> (id=151643)

[LoRA] Applying LoRA: r=16, alpha=32
[LoRA] Applying LoRA: r=16, alpha=32, targets=['q_proj', 'v_proj']
[LoRA] Backbone frozen — only LoRA params train
[LoRA] Trainable params: 5,046,272 / 7,617,817,088 (0.07%)
[LoRA Summary] Trainable: 5,046,272 / 7,617,817,088 (0.07%)
trainable params: 5,046,272 || all params: 7,617,817,088 || trainable%: 0.0662
Loading Swin Transformer V2: microsoft/swinv2-base-patch4-window8-256 (torch_dtype=torch.float32)
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 471/471 [00:00<00:00, 49263.54it/s]
Vision encoder: 88 params frozen (stages 0-1), 383 params trainable (stages 2-3)
Loaded: hidden_size=1024, num_patches=64
[MultimodalSentimentModel] Qwen backbone frozen — trainable params managed by PEFT (LoRA) or adapters

[R8] Wrapping model with MultitaskSentimentModel
[MultimodalSentimentModel] Qwen backbone frozen — trainable params managed by PEFT (LoRA) or adapters
[Multitask] Trainable params (427): LoRA + adapters + vision + projector + classifier
Total parameters: 8,068,342,016
Trainable parameters: 447,865,704
Prepared 2876 valid samples
Prepared 1000 valid samples
Prepared 1000 valid samples

Training config:
  LR: 2e-05
  Epochs: 10, Steps/Epoch: 180, Total: 1800, Warmup: 180
  Grad accumulation: 2
  Trainable params: 447,865,704 params across 427 tensors

[OK] Sanity forward: logits finite=True
     logits shape=torch.Size([48, 1, 1, 4])
     bad_batch=False

Epoch 1/10
Training:   0%|                                                                                                                                                                                                            | 0/360 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 360/360 [13:12<00:00,  2.20s/it, loss=0.1716, cls=0.1716]
Validating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 125/125 [03:18<00:00,  1.59s/it]
Train Loss: 0.8284 (CLS=0.8284)
Val Loss:   0.3791 (CLS=0.3791)
Val Macro-F1: 0.3946
[SAVED] Best model macro-F1=0.3946

Epoch 2/10
Training:   0%|                                                                                                                                                                                                            | 0/360 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 360/360 [13:15<00:00,  2.21s/it, loss=0.4824, cls=0.4824]
Validating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 125/125 [03:06<00:00,  1.49s/it]
Train Loss: 0.3223 (CLS=0.3223)
Val Loss:   0.2879 (CLS=0.2879)
Val Macro-F1: 0.3746

Epoch 3/10
Training:   0%|                                                                                                                                                                                                            | 0/360 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 360/360 [13:10<00:00,  2.20s/it, loss=0.1827, cls=0.1827]
Validating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 125/125 [03:06<00:00,  1.49s/it]
Train Loss: 0.2985 (CLS=0.2985)
Val Loss:   0.2789 (CLS=0.2789)
Val Macro-F1: 0.3254

Epoch 4/10
Training:   0%|                                                                                                                                                                                                            | 0/360 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 360/360 [13:18<00:00,  2.22s/it, loss=0.6650, cls=0.6650]
Validating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 125/125 [03:06<00:00,  1.49s/it]
Train Loss: 0.2716 (CLS=0.2716)
Val Loss:   0.2837 (CLS=0.2837)
Val Macro-F1: 0.4447
[SAVED] Best model macro-F1=0.4447

Epoch 5/10
Training:   0%|                                                                                                                                                                                                            | 0/360 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 360/360 [13:19<00:00,  2.22s/it, loss=0.3451, cls=0.3451]
Validating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 125/125 [03:05<00:00,  1.48s/it]
Train Loss: 0.2560 (CLS=0.2560)
Val Loss:   0.2698 (CLS=0.2698)
Val Macro-F1: 0.4338

Epoch 6/10
Training:   0%|                                                                                                                                                                                                            | 0/360 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 360/360 [13:08<00:00,  2.19s/it, loss=0.1859, cls=0.1859]
Validating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 125/125 [03:05<00:00,  1.49s/it]
Train Loss: 0.2441 (CLS=0.2441)
Val Loss:   0.2680 (CLS=0.2680)
Val Macro-F1: 0.4247

Epoch 7/10
Training:   0%|                                                                                                                                                                                                            | 0/360 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 360/360 [13:19<00:00,  2.22s/it, loss=0.1534, cls=0.1534]
Validating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 125/125 [03:06<00:00,  1.49s/it]
Train Loss: 0.2272 (CLS=0.2272)
Val Loss:   0.2680 (CLS=0.2680)
Val Macro-F1: 0.4427
[EARLY STOP] triggered
Validating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 125/125 [03:13<00:00,  1.55s/it]

=== Test Set ===
Test Loss: 0.3150 (CLS=0.3150)
Test Macro-F1: 0.4422
Macro Precision: 0.4414
Macro Recall:   0.4472
Macro F1:       0.4422
Prepared 1000 valid samples

=== Demo Inference ===
Comment: trải nghiệm dịch vụ chất lượng cao, tuyệt vời, hồ bơi rộng sạch, thích hợp để nghỉ dưỡng, khu resort rất gần vinwonder và grandworld
True labels: ['Location#Positive', 'Facilities#Positive', 'Service#Positive', 'Public_area#Neutral']

--- Aspect-level predictions ---

Aspect: Facilities
  Predicted: None
  Probabilities: {'None': 0.5678566694259644, 'Negative': 0.14610165357589722, 'Neutral': 0.08737168461084366, 'Positive': 0.19866998493671417}

Aspect: Public_area
  Predicted: Positive
  Probabilities: {'None': 0.08706483244895935, 'Negative': 0.027284976094961166, 'Neutral': 0.3830243647098541, 'Positive': 0.5026258230209351}

Aspect: Location
  Predicted: Negative
  Probabilities: {'None': 0.016445884481072426, 'Negative': 0.9557064175605774, 'Neutral': 0.0009690916049294174, 'Positive': 0.02687855251133442}

Aspect: Food
  Predicted: None
  Probabilities: {'None': 0.799465537071228, 'Negative': 0.0767006129026413, 'Neutral': 0.00013762805610895157, 'Positive': 0.1236962229013443}

Aspect: Room
  Predicted: Positive
  Probabilities: {'None': 0.28046715259552, 'Negative': 0.1746007651090622, 'Neutral': 0.11965563148260117, 'Positive': 0.42527642846107483}

Aspect: Service
  Predicted: Positive
  Probabilities: {'None': 0.12652577459812164, 'Negative': 0.03047863207757473, 'Neutral': 0.015357917174696922, 'Positive': 0.8276376724243164}