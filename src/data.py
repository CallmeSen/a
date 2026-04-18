import json
import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple

from src.config import (
    DATA_DIR, IMAGE_SIZE, MAX_IMAGES, MAX_TEXT_LEN,
    ASPECT_LABELS, NUM_ASPECTS, VIT_NAME,
)

SENTIMENT_TO_CLASS = {
    "Irrelative": 0,
    "Negative": 1,
    "Neutral": 2,
    "Positive": 3,
}

_siglip_processor = None

def _get_siglip_processor():
    global _siglip_processor
    if _siglip_processor is None:
        from transformers import AutoProcessor
        _siglip_processor = AutoProcessor.from_pretrained(VIT_NAME)
    return _siglip_processor


class MultimodalSentimentDataset(Dataset):
    """
    Per-aspect dataset: each instance is fixed to one aspect.
    Loads text and images for each sample.
    """

    def __init__(
        self,
        split: str,
        data_dir: str = DATA_DIR,
    ):
        self.split = split
        self.data_dir = data_dir

        split_file = os.path.join(data_dir, f"{split}.json")
        with open(split_file, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

        self.image_dir = os.path.join(data_dir, "image")

        # Load roi_data.csv: per image_name → list of {boxes, labels}
        roi_csv_path = os.path.join(data_dir, "roi_data.csv")
        if os.path.exists(roi_csv_path):
            self.roi_df = pd.read_csv(roi_csv_path)
            self._build_roi_dict()
        else:
            self.roi_df = None
            self._roi_dict = {}

    def _build_roi_dict(self):
        """Group roi_data.csv by file_name → list of {boxes, labels} per image."""
        self._roi_dict = {}
        if self.roi_df is None:
            return
        for _, row in self.roi_df.iterrows():
            file_name = str(row["file_name"])
            box = [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])]
            label = str(row.get("label", "object"))
            if file_name not in self._roi_dict:
                self._roi_dict[file_name] = {"boxes": [], "labels": []}
            self._roi_dict[file_name]["boxes"].append(box)
            self._roi_dict[file_name]["labels"].append(label)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, image_name: str) -> Optional[Image.Image]:
        """Load and preprocess a single image."""
        image_path = os.path.join(self.image_dir, image_name)
        try:
            img = Image.open(image_path).convert("RGB")
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
            return img
        except (FileNotFoundError, IOError):
            return None

    def _images_to_tensor(self, images: List[Image.Image]) -> torch.Tensor:
        """Convert list of PIL images to [M, 3, H, W] tensor using SigLIP processor."""
        processor = _get_siglip_processor()
        M = len(images)
        tensor = torch.zeros((MAX_IMAGES, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)

        for i, img in enumerate(images[:MAX_IMAGES]):
            img = img.convert("RGB")
            inputs = processor(images=img, return_tensors="pt")
            tensor[i] = inputs["pixel_values"][0]

        return tensor

    def _get_aspect_label(self, sample: Dict) -> Dict[int, int]:
        """
        Extract all aspect-sentiment labels for this sample.
        Returns dict: aspect_idx → class_id (0-3).
        """
        labels = {}
        text_img_labels = sample.get("text_img_label", [])
        for lbl_str in text_img_labels:
            for idx, aspect_name in enumerate(ASPECT_LABELS):
                prefix = f"{aspect_name}#"
                if lbl_str.startswith(prefix):
                    sentiment = lbl_str[len(prefix):]
                    class_id = SENTIMENT_TO_CLASS.get(sentiment, 0)
                    labels[idx] = class_id
        return labels

    def _get_image_names(self, sample: Dict) -> List[str]:
        """Get list of image filenames for this sample."""
        return sample.get("list_img", [])

    def _get_roi_for_image(self, image_name: str) -> Dict[str, Any]:
        """Get RoI data for a single image name. Returns empty dict if not found."""
        roi = self._roi_dict.get(image_name, None)
        if roi is None:
            return {"boxes": [], "labels": []}
        return {"boxes": roi["boxes"][:], "labels": roi["labels"][:]}

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        comment = sample.get("comment", "")

        image_names = self._get_image_names(sample)
        images = []
        for name in image_names[:MAX_IMAGES]:
            img = self._load_image(name)
            if img is not None:
                images.append(img)

        black_img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
        while len(images) < MAX_IMAGES:
            images.append(black_img)

        images_tensor = self._images_to_tensor(images)

        aspect_labels = self._get_aspect_label(sample)

        # Build roi_data per image
        roi_data_per_img = [self._get_roi_for_image(name) for name in image_names]

        return {
            "comment": comment,
            "pixel_values": images_tensor,
            "image_names": image_names,
            "aspect_labels": aspect_labels,
            "roi_data": roi_data_per_img,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    comments = [b["comment"] for b in batch]
    image_names = [b["image_names"] for b in batch]
    aspect_labels = [b["aspect_labels"] for b in batch]
    roi_data = [b["roi_data"] for b in batch]

    return {
        "comments": comments,
        "pixel_values": pixel_values,
        "image_names": image_names,
        "aspect_labels": aspect_labels,
        "roi_data": roi_data,
    }


def build_dataloaders(
    batch_size: int = 4,
    data_dir: str = DATA_DIR,
    num_workers: int = 4,
) -> Dict[str, DataLoader]:
    dataloaders = {}
    for split in ["train", "dev", "test"]:
        dataset = MultimodalSentimentDataset(
            split=split,
            data_dir=data_dir,
        )
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
    return dataloaders


def build_dataloader(
    split: str,
    tokenizer,
    batch_size: int = 4,
    data_dir: str = DATA_DIR,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    dataset = MultimodalSentimentDataset(
        split=split,
        data_dir=data_dir,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
