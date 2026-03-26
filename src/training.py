import math
from typing import List

import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score

from .config import COMPUTE_DTYPE


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


def setup_optimizer(sentiment_model, learning_rate, weight_decay):
    trainable_params = [
        param for _, param in sentiment_model.named_parameters()
        if param.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [{"params": trainable_params, "lr": learning_rate}],
        weight_decay=weight_decay,
        eps=1e-5,
    )
    return optimizer, trainable_params


def compute_loss(outputs, labels, run_device):
    """6-aspect 4-class CrossEntropy loss.

    Model output shape: logits [B, 6, 4].
    Labels shape: [B, 6].
    We flatten to [B*6, 4] and [B*6] for cross_entropy.
    """
    logits = outputs["logits"]   # [B, 6, 4]
    logits_flat = logits.view(-1, logits.size(-1))     # [B*6, 4]
    labels_flat = labels.view(-1).long()               # [B*6]
    loss = F.cross_entropy(
        logits_flat.float(),
        labels_flat,
        reduction="mean",
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
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return False


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    run_device,
    trainable_params,
    gradient_accumulation_steps=1,
    tokenizer=None,
):
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

        loss, l_cls = compute_loss(outputs, labels, run_device)
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

        # Check gradients for NaN/Inf IMMEDIATELY after backward — halt if found
        grad_bad_params = [
            (name, p) for name, p in model.named_parameters()
            if p.grad is not None and not torch.isfinite(p.grad).all()
        ]
        if grad_bad_params:
            optimizer.zero_grad(set_to_none=True)
            bad_names = [f"{n} ({p.grad.numel()}elems)" for n, p in grad_bad_params]
            raise RuntimeError(
                f"[FATAL-NaN] NaN/Inf gradients detected in {len(grad_bad_params)} param(s) "
                f"at step {step} after backward() — stopping immediately.\n"
                f"  Bad params: {bad_names[:5]}"
            )

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
def validate(model, dataloader, run_device):
    """Validate 6-aspect model.

    Labels: [B, 6] — all 6 aspects per sample.
    Logits: [B, 6, 4] — predict all 6 aspects.
    F1 is computed over all B*6 flattened predictions.
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

        loss, l_cls = compute_loss(outputs, labels, run_device)
        if not torch.isfinite(loss):
            continue

        total_loss += loss.item()
        total_cls_loss += l_cls.item()
        num_batches += 1

        logits = outputs["logits"]   # [B, 6, 4]
        pred_labels = logits.argmax(dim=-1)  # [B, 6]

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