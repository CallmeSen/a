"""Contrastive Loss modules for sentiment classification.

Triển khai hai loại loss:
1. SupervisedAngularMarginLoss (ArcFace-style): tăng khoảng cách góc giữa các lớp
2. CombinedSentimentLoss: kết hợp Focal Loss + ArcFace Loss

ArcFace formula:
    L = -log exp(s * cos(theta_y + m)) / sum(exp(s * cos(theta_j)))

Ưu điểm:
- Tăng inter-class separability (Positive vs Negative)
- Giảm intra-class variance
- Hoạt động tốt với class imbalance
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SupervisedAngularMarginLoss(nn.Module):
    """
    ArcFace-style Angular Margin Loss cho 4-class sentiment classification.

    Hoạt động trong không gian góc (hyper-sphere), tăng khoảng cách
    giữa các center của các lớp sentiment.

    Args:
        embedding_dim: Kích thước embedding vector (hidden_size = 2560)
        num_classes: Số lớp (4: None, Negative, Neutral, Positive)
        scale: Scaling factor cho logits (default: 30.0)
        margin: Angular margin trong radians (default: 0.5 ≈ 28.6°)
        class_weights: Tensor shape [num_classes] cho class balancing
    """

    def __init__(
        self,
        embedding_dim: int = 2560,
        num_classes: int = 4,
        scale: float = 30.0,
        margin: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.class_weights = class_weights

        # Class centers trong không gian embedding
        # Shape: [num_classes, embedding_dim]
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # Pre-compute sin(margin) và cos(margin)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

        # Threshold để tránh numerical instability khi cos(theta) gần -1
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: [B, D] - L2-normalized embeddings (z_a sau khi normalize)
            labels: [B] - Ground truth class indices (0-3)

        Returns:
            loss: Scalar loss value
        """
        # L2 normalize embeddings và weights
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)  # [B, D]
        weight_norm = F.normalize(self.weight, p=2, dim=1)      # [num_classes, D]

        # Cosine similarity: cos(theta_j) = <e, W_j>
        cos_theta = F.linear(embeddings_norm, weight_norm)  # [B, num_classes]
        cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Compute sin(theta) từ cos(theta)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2).clamp(0.0, 1.0))

        # cos(theta_y + m) = cos(theta_y) * cos(m) - sin(theta_y) * sin(m)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

        # Hard margin: chỉ áp dụng margin cho correct class
        cond = F.one_hot(labels, num_classes=self.num_classes).bool()
        cos_theta_new = torch.where(cond, cos_theta_m, cos_theta)

        # Scale logits
        logits = self.scale * cos_theta_new  # [B, num_classes]

        # Cross-entropy loss với optional class weights
        if self.class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        else:
            loss = F.cross_entropy(logits, labels)

        return loss

    def get_cosine_similarity(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Trả về cosine similarity giữa embeddings và class centers."""
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        return F.linear(embeddings_norm, weight_norm)


class CombinedSentimentLoss(nn.Module):
    """
    Kết hợp Focal Loss và ArcFace Loss cho sentiment classification.

    Focal Loss: Xử lý class imbalance tốt
    ArcFace Loss: Tăng inter-class separability

    Total Loss = focal_weight * Focal_Loss + arc_weight * ArcFace_Loss

    Args:
        embedding_dim: Kích thước embedding (hidden_size = 2560)
        num_classes: Số lớp sentiment (4)
        arc_scale: ArcFace scale factor (default: 30.0)
        arc_margin: ArcFace margin radians (default: 0.5)
        focal_gamma: Focal loss gamma (default: 2.0)
        focal_alpha: Focal loss alpha (default: 0.25)
        arc_weight: Trọng số cho ArcFace loss (default: 0.5)
        focal_weight: Trọng số cho Focal loss (default: 0.5)
        class_weights: Optional class weights cho class balancing
    """

    def __init__(
        self,
        embedding_dim: int = 2560,
        num_classes: int = 4,
        arc_scale: float = 30.0,
        arc_margin: float = 0.5,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        arc_weight: float = 0.5,
        focal_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.arc_loss = SupervisedAngularMarginLoss(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            scale=arc_scale,
            margin=arc_margin,
            class_weights=class_weights,
        )
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.arc_weight = arc_weight
        self.focal_weight = focal_weight
        # Store class_weights for focal loss (ArcFace already receives it)
        self._class_weights = class_weights

    def _focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss từ logits với class_weights hỗ trợ."""
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        probs = torch.exp(log_probs)

        # One-hot targets
        targets_onehot = torch.zeros_like(log_probs).scatter_(
            1, targets.unsqueeze(1), 1.0
        )

        # p_t = probability của correct class
        p_t = (targets_onehot * probs).sum(dim=-1).clamp(min=1e-7, max=1.0)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.focal_gamma

        # Cross-entropy
        ce = -(targets_onehot * log_probs).sum(dim=-1)

        # Focal loss = alpha * (1 - p_t)^gamma * CE
        focal = self.focal_alpha * focal_weight * ce

        # Apply class weights nếu có (tương tự ArcFace)
        if self._class_weights is not None:
            class_weight_per_sample = self._class_weights[targets]
            focal = focal * class_weight_per_sample

        return focal.mean()

    def forward(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple:
        """
        Args:
            logits: [B, num_classes] - Classifier logits từ model
            embeddings: [B, D] - Normalized embeddings (z_a)
            labels: [B] - Ground truth labels

        Returns:
            tuple: (total_loss, focal_loss_value, arc_loss_value)
        """
        # Focal loss từ logits
        focal = self._focal_loss(logits, labels)

        # ArcFace loss từ embeddings
        arc = self.arc_loss(embeddings, labels)

        # Combined loss
        total = self.focal_weight * focal + self.arc_weight * arc

        return total, focal.detach(), arc.detach()
