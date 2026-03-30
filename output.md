root@71d3261521137a80:~/a# USE_LORA=1 USE_MULTITASK=1 USE_WEIGHTED_SAMPLER=1 python multimodal_sentiment.py
The image processor of type `ViTImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`. 
Device: cuda
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
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 339/339 [00:00<00:00, 9332.91it/s]
[INFO] Tokenizer vocab size: 151667, embedding resized
[INFO] LLM base: Qwen2Model, num_layers=28, hidden_size=3584
Tokenizer: padding_side=right, pad_token=<|endoftext|> (id=151643)

[LoRA] Applying LoRA: r=32, alpha=64
[LoRA] Applying LoRA: r=32, alpha=64, targets=['q_proj', 'v_proj', 'o_proj']
[LoRA] Backbone frozen — only LoRA params train
[LoRA] Trainable params: 16,515,072 / 7,629,285,888 (0.22%)
[LoRA Summary] Trainable: 16,515,072 / 7,629,285,888 (0.22%)
trainable params: 16,515,072 || all params: 7,629,285,888 || trainable%: 0.2165
Loading Swin Transformer V2: microsoft/swinv2-base-patch4-window8-256 (torch_dtype=torch.float32)
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 471/471 [00:00<00:00, 47511.24it/s]
Vision encoder: 471 params trainable (all stages)
Loaded: hidden_size=1024, num_patches=64
[MultimodalSentimentModel] Qwen backbone frozen — trainable params managed by PEFT (LoRA) or adapters

[R8] Wrapping model with MultitaskSentimentModel
[MultimodalSentimentModel] Qwen backbone frozen — trainable params managed by PEFT (LoRA) or adapters
[Multitask] Trainable params (515): LoRA + adapters + vision + projector + classifier
Total parameters: 8,079,810,816
Trainable parameters: 450,524,928
Prepared 17256 valid samples
[DATA] WeightedRandomSampler: 17256 samples (1 per aspect), minority_upsample_ratio=2.0
[DATA] Label distribution: {0: 8606, 1: 830, 2: 1401, 3: 6419}
Prepared 6000 valid samples
Prepared 6000 valid samples

Training config:
  LR: 2e-05
  Epochs: 15, Steps/Epoch: 1079, Total: 16185, Warmup: 809
  Grad accumulation: 2
  Trainable params: 450,524,928 params across 515 tensors

[OK] Sanity forward: logits finite=True
     logits shape=torch.Size([8, 1, 1, 4])
     bad_batch=False

Epoch 1/15
Training:   0%|                                                                                                                                                                                             | 0/2157 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2157/2157 [19:16<00:00,  1.87it/s, loss=0.3543, cls=0.3543]
Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 750/750 [02:50<00:00,  4.41it/s]
Train Loss: 1.5115 (CLS=1.5115)
Val Loss:   0.8340 (CLS=0.8340)
Val Macro-F1: 0.3679
[SAVED] Best model macro-F1=0.3679

Epoch 2/15
Training:   0%|                                                                                                                                                                                             | 0/2157 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2157/2157 [19:24<00:00,  1.85it/s, loss=0.7549, cls=0.7549]
Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 750/750 [02:45<00:00,  4.52it/s]
Train Loss: 0.6899 (CLS=0.6899)
Val Loss:   0.5832 (CLS=0.5832)
Val Macro-F1: 0.4488
[SAVED] Best model macro-F1=0.4488

Epoch 3/15
Training:   0%|                                                                                                                                                                                             | 0/2157 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2157/2157 [21:29<00:00,  1.67it/s, loss=0.4292, cls=0.4292]
Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 750/750 [03:10<00:00,  3.94it/s]
Train Loss: 0.6472 (CLS=0.6472)
Val Loss:   0.5816 (CLS=0.5816)
Val Macro-F1: 0.4501
[SAVED] Best model macro-F1=0.4501

Epoch 4/15
Training:   0%|                                                                                                                                                                                             | 0/2157 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2157/2157 [20:10<00:00,  1.78it/s, loss=0.8498, cls=0.8498]
Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 750/750 [02:56<00:00,  4.24it/s]
Train Loss: 0.6099 (CLS=0.6099)
Val Loss:   0.6595 (CLS=0.6595)
Val Macro-F1: 0.3850

Epoch 5/15
Training:   0%|                                                                                                                                                                                             | 0/2157 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2157/2157 [20:05<00:00,  1.79it/s, loss=0.2767, cls=0.2767]
Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 750/750 [02:54<00:00,  4.30it/s]
Train Loss: 0.5765 (CLS=0.5765)
Val Loss:   0.6400 (CLS=0.6400)
Val Macro-F1: 0.4204

Epoch 6/15
Training:   0%|                                                                                                                                                                                             | 0/2157 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2157/2157 [19:40<00:00,  1.83it/s, loss=0.4030, cls=0.4030]
Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 750/750 [02:50<00:00,  4.39it/s]
Train Loss: 0.5498 (CLS=0.5498)
Val Loss:   0.6052 (CLS=0.6052)
Val Macro-F1: 0.4571
[SAVED] Best model macro-F1=0.4571

Epoch 7/15
Training:   0%|                                                                                                                                                                                             | 0/2157 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2157/2157 [19:17<00:00,  1.86it/s, loss=0.5463, cls=0.5463]
Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 750/750 [02:49<00:00,  4.41it/s]
Train Loss: 0.5259 (CLS=0.5259)
Val Loss:   0.5941 (CLS=0.5941)
Val Macro-F1: 0.4388

Epoch 8/15
Training:   0%|                                                                                                                                                                                             | 0/2157 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2157/2157 [19:30<00:00,  1.84it/s, loss=0.7644, cls=0.7644]
Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 750/750 [02:48<00:00,  4.44it/s]
Train Loss: 0.5067 (CLS=0.5067)
Val Loss:   0.5565 (CLS=0.5565)
Val Macro-F1: 0.4359

Epoch 9/15
Training:   0%|                                                                                                                                                                                             | 0/2157 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2157/2157 [19:09<00:00,  1.88it/s, loss=0.6496, cls=0.6496]
Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 750/750 [02:50<00:00,  4.41it/s]
Train Loss: 0.4622 (CLS=0.4622)
Val Loss:   0.6130 (CLS=0.6130)
Val Macro-F1: 0.4390

Epoch 10/15
Training:   0%|                                                                                                                                                                                             | 0/2157 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2157/2157 [19:41<00:00,  1.83it/s, loss=0.3253, cls=0.3253]
Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 750/750 [02:53<00:00,  4.32it/s]
Train Loss: 0.4346 (CLS=0.4346)
Val Loss:   0.6015 (CLS=0.6015)
Val Macro-F1: 0.4445

Epoch 11/15
Training:   0%|                                                                                                                                                                                             | 0/2157 [00:00<?, ?it/s][DIAG] tokenizer.padding_side=right, pad_token_id=151643
Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2157/2157 [19:54<00:00,  1.81it/s, loss=0.1140, cls=0.1140]
Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 750/750 [02:48<00:00,  4.45it/s]
Train Loss: 0.4119 (CLS=0.4119)
Val Loss:   0.6090 (CLS=0.6090)
Val Macro-F1: 0.4396
[EARLY STOP] triggered
Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 750/750 [02:55<00:00,  4.29it/s]

=== Test Set ===
Test Loss: 0.6244 (CLS=0.6244)
Test Macro-F1: 0.4626
Macro Precision: 0.4776
Macro Recall:   0.4660
Macro F1:       0.4626
Prepared 6000 valid samples
Traceback (most recent call last):
  File "/root/a/multimodal_sentiment.py", line 482, in <module>
    main()
  File "/root/a/multimodal_sentiment.py", line 386, in main
    demo_true_labels = demo_sample["raw_labels"]
                       ~~~~~~~~~~~^^^^^^^^^^^^^^
KeyError: 'raw_labels'