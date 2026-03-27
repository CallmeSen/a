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

            valid_samples.append(
                {
                    "comment": item.get("comment", ""),
                    "image_paths": valid_img_paths,
                    "parsed_labels": parsed_labels,
                    "raw_labels": raw_labels,
                    "aspect_present": {aspect: any(a == aspect for a, _ in parsed_labels) for aspect in ASPECT_LABELS},
                }
            )

        return valid_samples

    def __len__(self):
        return len(self.samples)

    def _labels_to_tensor(self, parsed_labels: list) -> torch.Tensor:
        labels = torch.zeros(self.num_aspects, dtype=torch.long)
        for aspect, sentiment in parsed_labels:
            a_id = self.aspect2id[aspect]
            labels[a_id] = _SENTIMENT_TO_CLASS[sentiment]
        return labels

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        image_tensors = []
        for img_path in sample["image_paths"]:
            image = Image.open(img_path).convert("RGB")
            image_tensors.append(self.transform(image))

        pixel_values = torch.stack(image_tensors)
        labels = self._labels_to_tensor(sample["parsed_labels"])

        return {
            "pixel_values": pixel_values,
            "num_images": len(image_tensors),
            "labels": labels,
            "comment": sample["comment"],
            "image_paths": sample["image_paths"],
            "aspect_present": sample["aspect_present"],
        }


def _build_aspect_text(comment: str, aspect_name: str) -> str:
    """Wrap comment with aspect instruction tokens per diagram."""
    return f"{ASPECT_START}{aspect_name}{ASPECT_END} {comment}"


def make_collate_fn(tokenizer_ref):
    def collate_fn(batch):
        # Expand each sample into 6 individual samples (1 per aspect).
        # Each raw sample contains text+images+all aspect labels.
        # After expansion: N raw samples → N × 6 samples (1 aspect each).
        expanded_items = []
        for item in batch:
            pixel_values = item["pixel_values"]   # [M, C, H, W]
            num_images = item["num_images"]
            labels = item["labels"]               # [6]
            comment = item["comment"]
            image_paths = item["image_paths"]

            for aid in range(labels.size(0)):
                aspect_name = ID2ASPECT[aid]
                aspect_texts = _build_aspect_text(comment, aspect_name)
                # R8: aspect_present = 1 if mentioned in dataset, 0 if "None"
                aspect_present_val = 1 if item["aspect_present"].get(aspect_name, False) else 0
                expanded_items.append({
                    "pixel_values": pixel_values,
                    "num_images": num_images,
                    "label": labels[aid].item(),   # scalar label for this aspect
                    "aspect_present": aspect_present_val,  # R8: binary aux label
                    "aspect_text": aspect_texts,
                    "comment": comment,
                    "image_paths": image_paths,
                })

        # Determine max images across all expanded items
        image_counts = [item["num_images"] for item in expanded_items]
        max_imgs = max(image_counts)

        # Pad pixel_values to [max_imgs, C, H, W]
        padded_pixels = []
        for item in expanded_items:
            pvs = item["pixel_values"]
            n = pvs.shape[0]
            if n < max_imgs:
                pad = torch.zeros(max_imgs - n, *pvs.shape[1:], dtype=pvs.dtype)
                padded_pixels.append(torch.cat([pvs, pad], dim=0))
            else:
                padded_pixels.append(pvs)
        pixel_values_batch = torch.stack(padded_pixels)  # [B, max_imgs, C, H, W]
        image_counts_tensor = torch.tensor(image_counts, dtype=torch.long)

        # Tokenize aspect-prompted texts
        aspect_texts = [item["aspect_text"] for item in expanded_items]
        text_inputs = tokenizer_ref(
            aspect_texts,
            padding=True,
            truncation=True,
            max_length=MAX_TEXT_LENGTH,
            return_tensors="pt",
        )

        # Labels: 1 aspect per sample → [B]
        aspect_labels = [item["label"] for item in expanded_items]
        aspect_labels_tensor = torch.tensor(aspect_labels, dtype=torch.long)

        # R8: Aspect presence labels for auxiliary task [B]
        aspect_present_labels = torch.tensor(
            [item["aspect_present"] for item in expanded_items], dtype=torch.long
        )

        return {
            "pixel_values": pixel_values_batch,
            "image_counts": image_counts_tensor,
            "input_ids": text_inputs.input_ids,
            "attention_mask": text_inputs.attention_mask,
            "labels": aspect_labels_tensor,          # [B] — sentiment class
            "aspect_present_labels": aspect_present_labels,  # [B] — R8: binary
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
    WeightedRandomSampler that oversamples samples with minority sentiment labels.

    R7: For class imbalance — samples containing Positive/Negative/Neutral aspects
    (not "None" only) are upsampled by `minority_upsample_ratio`.

    The sampler operates on raw dataset items (before aspect expansion), so each
    item contributes weight based on whether it has any sentiment aspect.

    Args:
        dataset: SentimentDataset
        minority_upsample_ratio: multiplier for samples with non-None sentiment
    Returns:
        WeightedRandomSampler instance (or None if all samples are equal weight)
    """
    import torch.utils.data
    weights = []
    for sample in dataset.samples:
        labels = sample["parsed_labels"]
        # If any aspect has sentiment (not "None") → minority → upsampled
        has_sentiment = any(sent != "None" for _, sent in labels)
        weight = minority_upsample_ratio if has_sentiment else 1.0
        weights.append(weight)

    total = sum(weights)
    # Normalize to valid probability distribution
    probabilities = [w / total for w in weights]

    num_samples = len(weights)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=probabilities,
        num_samples=num_samples,
        replacement=True,
    )
    print(f"[DATA] WeightedRandomSampler: {num_samples} samples, "
          f"minority_upsample_ratio={minority_upsample_ratio}")
    return sampler
