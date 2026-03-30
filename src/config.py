import os
import warnings
import logging
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_DTYPE = torch.bfloat16

VISION_MODEL_NAME = "microsoft/swinv2-base-patch4-window8-256"
VISION_HIDDEN_SIZE = 1024
LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LLM_HIDDEN_SIZE = 3584

IMAGE_SIZE = 256
MAX_TEXT_LENGTH = 256
MAX_IMAGES = 5
TOP_K_IMAGES = 3

DATA_DIR = "datasets"
IMAGE_DIR = os.path.join(DATA_DIR, "image")
OUTPUT_DIR = "output_model"
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.safetensors")
TRAINING_INFO_PATH = os.path.join(OUTPUT_DIR, "training_info.pt")

BATCH_SIZE = 8
NUM_WORKERS = 0

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
LORA_LR = 1e-4
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
USE_LORA = os.environ.get("USE_LORA", "0") == "1"
USE_MULTITASK = os.environ.get("USE_MULTITASK", "0") == "1"
USE_WEIGHTED_SAMPLER = os.environ.get("USE_WEIGHTED_SAMPLER", "0") == "1"
NUM_EPOCHS = 15
WARMUP_PROPORTION = 0.05
GRADIENT_ACCUMULATION_STEPS = 2
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 1e-4

ASPECT_START = "<ASP>"
ASPECT_END = "</ASP>"
ASPECT_LABELS = ["Facilities", "Public_area", "Location", "Food", "Room", "Service"]
ASPECT2ID = {label: idx for idx, label in enumerate(ASPECT_LABELS)}
ID2ASPECT = {idx: label for idx, label in enumerate(ASPECT_LABELS)}
NUM_ASPECTS = len(ASPECT_LABELS)

CLASS_LABELS = ["None", "Negative", "Neutral", "Positive"]
CLASS2ID = {label: idx for idx, label in enumerate(CLASS_LABELS)}
ID2CLASS = {idx: label for idx, label in enumerate(CLASS_LABELS)}
NUM_CLASSES = len(CLASS_LABELS)

_SENTIMENT_TO_CLASS = {"Negative": 1, "Neutral": 2, "Positive": 3}


# Dynamic: populated after tokenizer.add_special_tokens() in language_model.py
ASPECT_START_ID = None
ASPECT_END_ID = None

def _set_special_token_ids(start_id: int, end_id: int):
    global ASPECT_START_ID, ASPECT_END_ID
    ASPECT_START_ID = start_id
    ASPECT_END_ID = end_id

def setup_runtime() -> None:
    warnings.filterwarnings("ignore", message=".*AttentionMaskConverter.*")
    logging.getLogger("transformers").setLevel(logging.ERROR)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = False


def print_runtime_summary() -> None:
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Compute dtype: {COMPUTE_DTYPE}")
    print(f"Vision model: {VISION_MODEL_NAME}")
    print(f"LLM model:    {LLM_MODEL_NAME}")
    print(f"Data dir:     {DATA_DIR}")
