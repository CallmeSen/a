import os
import json
from typing import Dict

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoImageProcessor

from .config import (
    VISION_MODEL_NAME,
    IMAGE_SIZE,
    MAX_IMAGES,
    MAX_TEXT_LENGTH,
    DATA_DIR,
    IMAGE_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    _SENTIMENT_TO_CLASS,
    ASPECT_START,
    ASPECT_END,
    ASPECT2ID,
    ID2ASPECT,
    ASPECT_LABELS,
    CLASS_LABELS,
)


def compute_class_weights(dataset_splits, num_classes=4):
    """Đếm tần suất mỗi class trong training set và compute inverse-frequency weights.

    Mỗi sample trong training set được expand thành 6 aspects.
    - Aspect có trong parsed_labels → sentiment class (1/2/3)
    - Aspect không có trong parsed_labels → class "None" (id=0)
    """
    from collections import Counter

    all_labels = []
    for item in dataset_splits.get("train", []):
        raw_labels = item.get("text_img_label", [])
        parsed = []
        for label in raw_labels:
            parsed_l = _parse_label_standalone(label)
            if parsed_l is not None:
                parsed.append(parsed_l)

        mentioned_aspects = {p[0] for p in parsed}
        for aspect in ASPECT_LABELS:
            if aspect in mentioned_aspects:
                sentiment = next(s for a, s in parsed if a == aspect)
                all_labels.append(sentiment)
            else:
                all_labels.append("None")

    counter = Counter(all_labels)
    total = len(all_labels)
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for i, cls_name in enumerate(CLASS_LABELS):
        freq = counter.get(cls_name, 0)
        weights[i] = total / (freq + 1e-6)
    weights = weights / weights.sum() * num_classes
    print(f"[DATA] Class distribution: {dict(sorted(counter.items()))}")
    print(f"[DATA] Class weights: {[round(w.item(), 4) for w in weights]}")
    return weights


def _parse_label_standalone(label: str):
    """Standalone label parser (equivalent to SentimentDataset._parse_label)."""
    if "#" not in label:
        return None
    parts = label.split("#")
    if len(parts) != 2:
        return None
    aspect, sentiment = parts[0], parts[1]
    if aspect in ASPECT2ID and sentiment in _SENTIMENT_TO_CLASS:
        return (aspect, sentiment)
    return None


swin_image_processor = AutoImageProcessor.from_pretrained(VISION_MODEL_NAME)


def _ensure_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if img.mode != "RGB" else img


def build_transform(input_size: int = IMAGE_SIZE):
    def _transform(img: Image.Image) -> torch.Tensor:
        img = _ensure_rgb(img)
        proc_inputs = swin_image_processor(
            images=img,
            return_tensors="pt",
            do_resize=True,
            size={"height": input_size, "width": input_size},
            do_center_crop=False,
        )
        return proc_inputs["pixel_values"].squeeze(0)

    return _transform


def build_train_transform(input_size: int = IMAGE_SIZE):
    """Image augmentation for training — applied to ALL classes uniformly.

    Additions vs basic (R7):
    - RandomRotation(±15°): helps model see rotated hotel views
    - RandomGrayscale(p=0.1): simulates low-light / monochrome images
    - RandomVerticalFlip(p=0.2): hotel photos can be taken from various angles
    """
    train_aug = T.Compose(
        [
            T.RandomResizedCrop((input_size, input_size), scale=(0.85, 1.0), interpolation=InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.RandomGrayscale(p=0.1),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        ]
    )

    def _transform(img: Image.Image) -> torch.Tensor:
        img = _ensure_rgb(img)
        img = train_aug(img)
        proc_inputs = swin_image_processor(
            images=img,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )
        return proc_inputs["pixel_values"].squeeze(0)

    return _transform


def load_dataset_json(json_path: str) -> list:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {json_path}")
    return data


def load_all_splits(data_dir: str = DATA_DIR) -> Dict[str, list]:
    splits = {}
    for split in ["train", "dev", "test"]:
        json_path = os.path.join(data_dir, f"{split}.json")
        if os.path.exists(json_path):
            splits[split] = load_dataset_json(json_path)
        else:
            print(f"[WARNING] {json_path} not found!")
            splits[split] = []
    return splits


class SentimentDataset(Dataset):
    def __init__(self, data: list, image_dir: str, aspect2id: dict, transform=None):
        self.data = data
        self.image_dir = image_dir
        self.aspect2id = aspect2id
        self.num_aspects = len(aspect2id)
        self.transform = transform if transform else build_transform(IMAGE_SIZE)

        self.samples = self._prepare_samples()
        print(f"Prepared {len(self.samples)} valid samples")

    def _parse_label(self, label: str):
        if "#" not in label:
            return None
        parts = label.split("#")
        if len(parts) != 2:
            return None
        aspect, sentiment = parts[0], parts[1]
        if aspect in self.aspect2id and sentiment in _SENTIMENT_TO_CLASS:
            return (aspect, sentiment)
        return None

    def _prepare_samples(self) -> list:
        """Prepare flattened samples: each entry is one (raw_sample, aspect) pair.

        This ensures WeightedRandomSampler weights each aspect independently,
        rather than up/down-sampling entire samples based on just one aspect.
        """
        valid_samples = []
        for item in self.data:
            if not item.get("list_img") or len(item["list_img"]) == 0:
                continue

            raw_labels = item.get("text_img_label", [])
            if not raw_labels:
                continue

            parsed_labels = []
            for label in raw_labels:
                parsed = self._parse_label(label)
                if parsed is not None:
                    parsed_labels.append(parsed)

            if not parsed_labels:
                continue

            valid_img_paths = []
            for img_name in item["list_img"][:MAX_IMAGES]:
                img_path = os.path.join(self.image_dir, img_name)
                if os.path.exists(img_path):
                    valid_img_paths.append(img_path)

            if not valid_img_paths:
                continue

            # Track which aspects are mentioned in this sample
            mentioned_aspects = {a for a, _ in parsed_labels}

            # Create one entry PER ASPECT (flattened, not per-sample)
            for aspect_name in ASPECT_LABELS:
                aspect_sentiment = "None"
                aspect_present = 0
                for a, s in parsed_labels:
                    if a == aspect_name:
                        aspect_sentiment = s
                        aspect_present = 1
                        break

                valid_samples.append(
                    {
                        "comment": item.get("comment", ""),
                        "image_paths": valid_img_paths,
                        "aspect_name": aspect_name,
                        "aspect_id": ASPECT2ID[aspect_name],
                        "sentiment": aspect_sentiment,
                        "label": _SENTIMENT_TO_CLASS.get(aspect_sentiment, 0),
                        "aspect_present": aspect_present,
                    }
                )

        return valid_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        image_tensors = []
        for img_path in sample["image_paths"]:
            image = Image.open(img_path).convert("RGB")
            image_tensors.append(self.transform(image))

        pixel_values = torch.stack(image_tensors)
        label = sample["label"]

        return {
            "pixel_values": pixel_values,
            "num_images": len(image_tensors),
            "label": label,
            "aspect_name": sample["aspect_name"],
            "aspect_present": sample["aspect_present"],
            "comment": sample["comment"],
            "image_paths": sample["image_paths"],
        }


def _build_aspect_text(comment: str, aspect_name: str) -> str:
    """Wrap comment with aspect instruction tokens per diagram."""
    return f"{ASPECT_START}{aspect_name}{ASPECT_END} {comment}"


def make_collate_fn(tokenizer_ref):
    def collate_fn(batch):
        # Each item in batch is already one (raw_sample, aspect) pair — no expansion needed.
        # This ensures WeightedRandomSampler weights each aspect independently.
        padded_pixels = []
        image_counts = []
        aspect_texts = []
        aspect_labels = []
        aspect_present_labels = []

        for item in batch:
            pvs = item["pixel_values"]
            padded_pixels.append(pvs)
            image_counts.append(item["num_images"])
            aspect_texts.append(_build_aspect_text(item["comment"], item["aspect_name"]))
            aspect_labels.append(item["label"])
            aspect_present_labels.append(item["aspect_present"])

        # Pad pixel_values
        max_imgs = max(i.shape[0] for i in padded_pixels)
        padded = []
        for pvs in padded_pixels:
            n = pvs.shape[0]
            if n < max_imgs:
                pad = torch.zeros(max_imgs - n, *pvs.shape[1:], dtype=pvs.dtype)
                padded.append(torch.cat([pvs, pad], dim=0))
            else:
                padded.append(pvs)
        pixel_values_batch = torch.stack(padded)
        image_counts_tensor = torch.tensor(image_counts, dtype=torch.long)

        # Tokenize
        text_inputs = tokenizer_ref(
            aspect_texts,
            padding=True,
            truncation=True,
            max_length=MAX_TEXT_LENGTH,
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values_batch,
            "image_counts": image_counts_tensor,
            "input_ids": text_inputs.input_ids,
            "attention_mask": text_inputs.attention_mask,
            "labels": torch.tensor(aspect_labels, dtype=torch.long),
            "aspect_present_labels": torch.tensor(aspect_present_labels, dtype=torch.long),
        }

    return collate_fn


def build_dataloaders(dataset_splits: Dict[str, list], aspect2id: dict, tokenizer):
    train_dataset = SentimentDataset(dataset_splits["train"], IMAGE_DIR, aspect2id, transform=build_train_transform(IMAGE_SIZE))
    dev_dataset = SentimentDataset(dataset_splits["dev"], IMAGE_DIR, aspect2id, transform=build_transform(IMAGE_SIZE))
    test_dataset = SentimentDataset(dataset_splits["test"], IMAGE_DIR, aspect2id, transform=build_transform(IMAGE_SIZE))

    collate_fn = make_collate_fn(tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)

    return train_dataset, dev_dataset, test_dataset, train_loader, dev_loader, test_loader


def build_weighted_sampler(dataset: SentimentDataset, minority_upsample_ratio: float = 4.0) -> "torch.utils.data.WeightedRandomSampler":
    """
    WeightedRandomSampler that oversamples minority sentiment aspects.

    R8 fix: Dataset now returns one (raw_sample, aspect) pair per index.
    Sampler weights each aspect independently by its sentiment label.
    - "None" (label=0): base weight = 1.0
    - Negative/Neutral/Positive (label=1/2/3): upsampled by minority_upsample_ratio

    This is correct because each dataset index is exactly one aspect.
    """
    import torch.utils.data
    from collections import Counter

    # Count label distribution
    label_counts = Counter()
    for sample in dataset.samples:
        label_counts[sample["label"]] += 1

    total = len(dataset.samples)
    weights = []
    # Per-class weights based on inverse frequency.
    # Negative (830) vs None (8606): 10.4x → weight 10.0
    # Neutral (1401) vs None: 6.1x → weight 6.0
    # Positive (6419) vs None: 1.3x → weight 1.5
    # None: baseline weight 1.0
    class_ratio = {0: 1.0, 1: 10.0, 2: 6.0, 3: 1.5}
    for sample in dataset.samples:
        label = sample["label"]
        weights.append(class_ratio.get(label, 1.0))

    probabilities = [w / sum(weights) for w in weights]

    num_samples = len(weights)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=probabilities,
        num_samples=num_samples,
        replacement=True,
    )
    print(f"[DATA] WeightedRandomSampler: {num_samples} samples (1 per aspect), "
          f"minority_upsample_ratio={minority_upsample_ratio}")
    print(f"[DATA] Label distribution: {dict(sorted(label_counts.items()))}")
    return sampler
