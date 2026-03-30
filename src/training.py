import math
from typing import List

import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score

from .config import COMPUTE_DTYPE


# --- Focal Loss with optional label smoothing ---
def focal_loss_with_smoothing(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor = None,
    alpha: float = 0.25,
    gamma: float = 2.0,
    label_smoothing: float = 0.1,
) -> torch.Tensor:
    """Focal loss + label smoothing for extreme class imbalance.

    - gamma: focusing parameter — down-weights easy examples (high p_t)
    - alpha: class balancing weight for the rare (positive) classes
    - label_smoothing: soft targets to prevent overconfidence
    """
    num_classes = logits.size(-1)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = torch.exp(log_probs)

    # One-hot with smoothing
    if label_smoothing > 0:
        targets_smooth = torch.zeros_like(log_probs).scatter_(
            1, targets.unsqueeze(1), 1.0
        )
        targets_smooth = targets_smooth * (1 - label_smoothing) + label_smoothing / num_classes
    else:
        targets_onehot = torch.zeros_like(log_probs).scatter_(
            1, targets.unsqueeze(1), 1.0
        )
        targets_smooth = targets_onehot

    # Focal weight: (1 - p_t)^gamma
    p_t = (targets_smooth * probs).sum(dim=-1).clamp(min=1e-7, max=1.0)
    focal_weight = (1 - p_t) ** gamma

    # CE per sample
    ce = -(targets_smooth * log_probs).sum(dim=-1)

    # Alpha balancing (up-weight rare classes; down-weight "None")
    if class_weights is not None:
        sample_weights = class_weights[targets]
        loss = alpha * focal_weight * ce * sample_weights
    else:
        loss = alpha * focal_weight * ce

    return loss.mean()


class LazyLambdaScheduler:
    def __init__(self, optimizer, lr_lambda):
        self.optimizer = optimizer
        self.lr_lambda = lr_lambda
        self.base_lrs = []
        for group in optimizer.param_groups:
            base_lr = group["lr"]
            group.setdefault("initial_lr", base_lr)
            self.base_lrs.append(base_lr)
        self._last_lr = list(self.base_lrs)
        self.last_epoch = -1
        self.started = False

    def _apply(self, step_idx):
        scale = float(self.lr_lambda(step_idx))
        self._last_lr = []
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            lr = base_lr * scale
            group["lr"] = lr
            self._last_lr.append(lr)
        self.last_epoch = step_idx

    def start(self):
        if self.started:
            return
        self.started = True
        self._apply(0)

    def step(self):
        if not self.started:
            self.start()
            return
        self._apply(self.last_epoch + 1)

    def get_last_lr(self):
        return list(self._last_lr)


def setup_optimizer(sentiment_model, learning_rate, weight_decay, vision_lr_ratio=0.1, lora_lr=None):
    """Setup AdamW optimizer with separate LR for vision encoder and LoRA params.

    vision_lr_ratio: multiplier applied to learning_rate for vision_encoder params.
    Default 0.1 means vision gets LR = learning_rate * 0.1.
    lora_lr: separate learning rate for LoRA parameters (if None, uses learning_rate).
    """
    trainable_params = [
        param for _, param in sentiment_model.named_parameters()
        if param.requires_grad
    ]

    vision_params = [
        p for n, p in sentiment_model.named_parameters()
        if p.requires_grad and "vision_encoder" in n
    ]
    lora_params = [
        p for n, p in sentiment_model.named_parameters()
        if p.requires_grad and ("lora_" in n or "peft" in n.lower())
    ]
    other_params = [
        p for n, p in sentiment_model.named_parameters()
        if p.requires_grad and "vision_encoder" not in n
        and "lora_" not in n and "peft" not in n.lower()
    ]

    param_groups = [{"params": other_params, "lr": learning_rate}]
    if vision_params:
        param_groups.append({"params": vision_params, "lr": learning_rate * vision_lr_ratio})
    if lora_params:
        param_groups.append({"params": lora_params, "lr": lora_lr or learning_rate * 5})

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=weight_decay,
        eps=1e-5,
    )
    return optimizer, trainable_params


def compute_loss(outputs, labels, run_device, class_weights=None):
    """Single-head 4-class Focal loss.

    NOTE: class_weights are intentionally NOT applied in the loss function.
    WeightedRandomSampler already handles class imbalance by oversampling minority
    samples at the dataset level. Applying class_weights in the loss on top of
    sampler oversampling causes double-weighting which destabilizes training.
    """
    logits = outputs["logits"]
    while logits.dim() > 2:
        logits = logits.squeeze(-2)
    loss = focal_loss_with_smoothing(
        logits.float(),
        labels.reshape(-1).long(),
        class_weights=None,
        alpha=1.0,
        gamma=1.0,
        label_smoothing=0.0,
    )
    return loss, loss.detach()


def _sanitize_grads(model):
    count = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            bad = torch.isnan(p.grad) | torch.isinf(p.grad)
            if bad.any():
                pct = bad.float().mean().item() * 100
                print(f"[WARN-GRAD] {name}: {pct:.1f}% NaN/Inf grads -> zeroed")
                p.grad[bad] = 0.0
                count += 1
    return count


def _optimizer_step(
    model, optimizer, scheduler, trainable_params: List[torch.nn.Parameter]
):
    n_fixed = _sanitize_grads(model)

    if n_fixed > 0:
        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state.get(p, {})
                if "exp_avg" in state:
                    state["exp_avg"].zero_()
                if "exp_avg_sq" in state:
                    state["exp_avg_sq"].zero_()
        optimizer.zero_grad(set_to_none=True)
        return True

    if trainable_params:
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return False


def multi_task_compute_loss(outputs, labels, aspect_present_labels, run_device, class_weights=None):
    """Single-task Focal loss (aspect detection auxiliary head removed).

    NOTE: class_weights intentionally omitted — sampler handles imbalance.
    See compute_loss() for details.
    """
    logits = outputs["logits"]
    while logits.dim() > 2:
        logits = logits.squeeze(-2)
    loss = focal_loss_with_smoothing(
        logits.float(),
        labels.reshape(-1).long(),
        class_weights=None,
        alpha=1.0,
        gamma=1.0,
        label_smoothing=0.0,
    )
    return loss, loss, torch.tensor(0.0, device=run_device)


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    run_device,
    trainable_params,
    gradient_accumulation_steps=1,
    tokenizer=None,
    class_weights=None,
    use_multitask=False,
):
    """train_epoch: supports both single-task and R8 multitask mode.

    use_multitask=True: uses multi_task_compute_loss (aspect_present_labels from batch).
    use_multitask=False: uses compute_loss (backward compatible).
    """
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    num_batches = 0
    skipped_batches = 0
    sanitized_steps = 0
    optimizer_steps = 0
    valid_micro_steps = 0

    optimizer.zero_grad(set_to_none=True)
    scheduler.start()
    progress_bar = tqdm(dataloader, desc="Training")

    for step, batch in enumerate(progress_bar):
        pixel_values = batch["pixel_values"].to(run_device, dtype=COMPUTE_DTYPE)
        input_ids = batch["input_ids"].to(run_device)
        labels = batch["labels"].to(run_device)

        # R8: Multitask — extract aspect_present_labels
        aspect_present_labels = None
        if use_multitask:
            asp_labels = batch.get("aspect_present_labels")
            if asp_labels is not None:
                aspect_present_labels = asp_labels.to(run_device)

        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(run_device)

        image_counts = batch.get("image_counts")
        if image_counts is not None:
            image_counts = image_counts.to(run_device)

        try:
            outputs = model(
                pixel_values,
                input_ids,
                attention_mask=attention_mask,
                image_counts=image_counts,
            )
        except RuntimeError as e:
            if "NaN" in str(e) or "corrupted" in str(e):
                skipped_batches += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            raise

        if outputs.get("bad_batch", False):
            skipped_batches += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        logits = outputs["logits"]
        if not torch.isfinite(logits).all():
            skipped_batches += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        # R8: Multitask vs single-task loss
        if use_multitask and aspect_present_labels is not None:
            loss, l_cls, asp_loss = multi_task_compute_loss(
                outputs, labels, aspect_present_labels, run_device, class_weights
            )
        else:
            loss, l_cls = compute_loss(outputs, labels, run_device, class_weights)
            asp_loss = None
        if not torch.isfinite(loss):
            skipped_batches += 1
            optimizer.zero_grad(set_to_none=True)
            if torch.isnan(loss):
                raise RuntimeError(
                    f"[FATAL-NaN] Loss is NaN at step {step} (batch skipped) — stopping immediately."
                )
            continue

        if step == 0 and tokenizer is not None:
            print(
                f"[DIAG] tokenizer.padding_side={tokenizer.padding_side}, "
                f"pad_token_id={tokenizer.pad_token_id}"
            )

        scaled_loss = loss / gradient_accumulation_steps
        scaled_loss.backward()
        valid_micro_steps += 1

        total_loss += loss.item()
        total_cls_loss += l_cls.item()
        num_batches += 1

        if valid_micro_steps % gradient_accumulation_steps == 0:
            skipped_opt = _optimizer_step(
                model, optimizer, scheduler, trainable_params
            )
            optimizer_steps += 1
            if skipped_opt:
                sanitized_steps += 1
                print(
                    f"\n[WARN] Optimizer step {optimizer_steps} skipped "
                    f"(NaN gradients detected)"
                )

        progress_bar.set_postfix(
            {"loss": f"{loss.item():.4f}", "cls": f"{l_cls.item():.4f}"}
        )

    if valid_micro_steps % gradient_accumulation_steps != 0 and valid_micro_steps > 0:
        _optimizer_step(model, optimizer, scheduler, trainable_params)

    if skipped_batches > 0:
        print(f"[WARN] {skipped_batches} batches skipped (bad batch or non-finite outputs)")
    if sanitized_steps > 0:
        print(
            f"[INFO] {sanitized_steps}/{optimizer_steps} optimizer steps had "
            f"NaN grads (sanitized to 0)"
        )

    if num_batches == 0:
        return float("nan"), float("nan")

    return total_loss / num_batches, total_cls_loss / num_batches


@torch.no_grad()
def validate(model, dataloader, run_device, class_weights=None, use_multitask=False):
    """Validate — supports both single-task and R8 multitask mode.

    use_multitask=True: computes sentiment loss only (aspect detection is auxiliary, not metric-tracked).
    """
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    num_batches = 0

    all_true_labels = []
    all_pred_labels = []

    for batch in tqdm(dataloader, desc="Validating"):
        pixel_values = batch["pixel_values"].to(run_device, dtype=COMPUTE_DTYPE)
        input_ids = batch["input_ids"].to(run_device)
        labels = batch["labels"].to(run_device)

        # R8: Multitask — extract aspect_present_labels for loss
        aspect_present_labels = None
        if use_multitask:
            asp_labels = batch.get("aspect_present_labels")
            if asp_labels is not None:
                aspect_present_labels = asp_labels.to(run_device)

        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(run_device)

        image_counts = batch.get("image_counts")
        if image_counts is not None:
            image_counts = image_counts.to(run_device)

        try:
            outputs = model(
                pixel_values,
                input_ids,
                attention_mask=attention_mask,
                image_counts=image_counts,
            )
        except RuntimeError:
            continue

        if outputs.get("bad_batch", False):
            continue

        # R8: use multitask loss if aspect_present_labels available
        if use_multitask and aspect_present_labels is not None:
            loss, l_cls, _ = multi_task_compute_loss(
                outputs, labels, aspect_present_labels, run_device, class_weights
            )
        else:
            loss, l_cls = compute_loss(outputs, labels, run_device, class_weights)
        if not torch.isfinite(loss):
            continue

        total_loss += loss.item()
        total_cls_loss += l_cls.item()
        num_batches += 1

        logits = outputs["logits"]
        # Handle [B, 1, 1, 4] -> [B, 4]
        if logits.dim() == 4:
            logits = logits.squeeze(1).squeeze(1)
        elif logits.dim() == 3:
            logits = logits.squeeze(1)
        pred_labels = logits.argmax(dim=-1)  # [B]

        all_true_labels.append(labels.cpu())
        all_pred_labels.append(pred_labels.cpu())

    if len(all_true_labels) == 0:
        print("[WARN] No valid validation batches")
        return float("nan"), float("nan"), 0.0, None, None

    avg_loss = total_loss / num_batches
    avg_cls = total_cls_loss / num_batches

    all_true = torch.cat(all_true_labels, dim=0)
    all_pred = torch.cat(all_pred_labels, dim=0)

    true_flat = all_true.numpy().reshape(-1)
    pred_flat = all_pred.numpy().reshape(-1)
    macro_f1 = f1_score(true_flat, pred_flat, average="macro", zero_division=0)

    return avg_loss, avg_cls, macro_f1, all_pred, all_true