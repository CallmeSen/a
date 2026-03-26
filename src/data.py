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
)


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
    train_aug = T.Compose(
        [
            T.RandomResizedCrop((input_size, input_size), scale=(0.85, 1.0), interpolation=InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
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
        }


def _build_aspect_text(comment: str, aspect_name: str) -> str:
    """Wrap comment with aspect instruction tokens per diagram."""
    return f"{ASPECT_START}{aspect_name}{ASPECT_END} {comment}"


def make_collate_fn(tokenizer_ref):
    def collate_fn(batch):
        image_counts = [item["num_images"] for item in batch]
        max_imgs = max(image_counts)

        padded_pixels = []
        for item in batch:
            pvs = item["pixel_values"]
            n = pvs.shape[0]
            if n < max_imgs:
                pad = torch.zeros(max_imgs - n, *pvs.shape[1:], dtype=pvs.dtype)
                padded_pixels.append(torch.cat([pvs, pad], dim=0))
            else:
                padded_pixels.append(pvs)

        pixel_values = torch.stack(padded_pixels)
        image_counts_tensor = torch.tensor(image_counts)

        # Build 6 aspect texts per sample (one per aspect)
        aspect_texts_by_aspect = [[] for _ in range(6)]
        multi_labels = []

        for item in batch:
            comment = item["comment"]
            labels = item["labels"]
            multi_labels.append(labels)
            for a in range(6):
                aspect_texts_by_aspect[a].append(_build_aspect_text(comment, ID2ASPECT[a]))

        # Tokenize all B*6 texts in one batch to ensure uniform length,
        # then reshape to [B, 6, L] for multi-aspect forward pass.
        all_texts = []
        for a in range(6):
            all_texts.extend(aspect_texts_by_aspect[a])
        text_inputs = tokenizer_ref(
            all_texts,
            padding=True,
            truncation=True,
            max_length=MAX_TEXT_LENGTH,
            return_tensors="pt",
        )
        B = len(batch)
        L = text_inputs.input_ids.size(1)
        input_ids = text_inputs.input_ids.view(B, 6, L)           # [B, 6, L]
        attention_mask = text_inputs.attention_mask.view(B, 6, L) # [B, 6, L]

        # Multi-label tensor: [B, 6] — used as labels for 6-aspect loss
        multi_labels_tensor = torch.stack(multi_labels)

        return {
            "pixel_values": pixel_values,
            "image_counts": image_counts_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": multi_labels_tensor,        # [B, 6] — all 6 aspect labels
            "comments": [item["comment"] for item in batch],
            "image_paths": [item["image_paths"] for item in batch],
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
