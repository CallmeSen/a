import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType

from .config import TOP_K_IMAGES
from .vit_transformer import VisionEncoder
from .projector_layer import MLPProjector


class MultimodalSentimentModel(nn.Module):
    def __init__(
        self,
        vision_encoder: VisionEncoder,
        projector: MLPProjector,
        llm_model,
        tokenizer,
        num_aspects: int = 6,
        num_sentiment_classes: int = 3,
        top_k_images: int = TOP_K_IMAGES,
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.projector = projector
        self.tokenizer = tokenizer
        self.num_aspects = num_aspects
        self.num_sentiment_classes = num_sentiment_classes
        self.top_k_images = top_k_images

        self.llm_dtype = next(llm_model.parameters()).dtype
        self.llm_hidden_size = llm_model.config.hidden_size

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["wqkv", "wo"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.llm = get_peft_model(llm_model, lora_config)
        self.llm.print_trainable_parameters()
        self.llm.enable_input_require_grads()

        self.text_embeddings = self.llm.get_input_embeddings()
        for param in self.text_embeddings.parameters():
            param.requires_grad = False

        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        self.aspect_token_embeddings = nn.Embedding(num_aspects, self.llm_hidden_size)
        nn.init.normal_(self.aspect_token_embeddings.weight, std=0.02)

        # Explicit fusion blocks to align with architecture diagram:
        # 1) Cross-modal attention (image <-> text)
        # 2) Aspect-guided attention over fused context
        self.num_fusion_heads = self._infer_num_fusion_heads(self.llm_hidden_size)
        self.cross_modal_attn = nn.MultiheadAttention(
            embed_dim=self.llm_hidden_size,
            num_heads=self.num_fusion_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.aspect_guided_attn = nn.MultiheadAttention(
            embed_dim=self.llm_hidden_size,
            num_heads=self.num_fusion_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.fusion_norm = nn.LayerNorm(self.llm_hidden_size)

        self.acd_head = nn.Linear(self.llm_hidden_size, 1)
        nn.init.zeros_(self.acd_head.weight)
        nn.init.zeros_(self.acd_head.bias)

        self.asc_head = nn.Linear(self.llm_hidden_size, num_sentiment_classes)
        nn.init.xavier_uniform_(self.asc_head.weight)
        nn.init.zeros_(self.asc_head.bias)

        self._grad_safety_handles = []
        self._register_grad_safety_hooks()

    @staticmethod
    def _infer_num_fusion_heads(hidden_size: int) -> int:
        for h in (8, 4, 2):
            if hidden_size % h == 0:
                return h
        return 1

    def train(self, mode: bool = True):
        super().train(mode)
        self.vision_encoder.eval()
        self.llm.train(mode)
        return self

    def _make_grad_safety_hook(self):
        def hook(grad: torch.Tensor) -> torch.Tensor:
            if grad is None:
                return grad
            if torch.isfinite(grad).all():
                return grad
            return torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

        return hook

    def _register_grad_safety_hooks(self):
        grad_hook = self._make_grad_safety_hook()
        for _, param in self.named_parameters():
            if param.requires_grad:
                self._grad_safety_handles.append(param.register_hook(grad_hook))

    def _check_nan(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        if has_nan or has_inf:
            finite_vals = tensor[torch.isfinite(tensor)]
            if finite_vals.numel() > 0:
                print(
                    f"[NaN-DETECT] {name}: nan={has_nan}, inf={has_inf}, "
                    f"range=[{finite_vals.min().item():.4f}, {finite_vals.max().item():.4f}]"
                )
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e4, neginf=-1e4)
        return tensor

    def _masked_text_summary(self, text_lang_tokens: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attention_mask is None:
            return text_lang_tokens.float().mean(dim=1)
        mask = attention_mask.to(device=text_lang_tokens.device, dtype=torch.float32).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (text_lang_tokens.float() * mask).sum(dim=1) / denom

    def _get_combined_embeds(self, pixel_values, input_ids, attention_mask, image_counts=None):
        B = pixel_values.shape[0]

        with torch.no_grad():
            text_lang_tokens = self.text_embeddings(input_ids.to(pixel_values.device))

        if pixel_values.dim() == 5:
            M = pixel_values.shape[1]
            P = self.vision_encoder.num_patches

            if image_counts is None:
                image_counts = torch.full((B,), M, device=pixel_values.device, dtype=torch.long)
            else:
                image_counts = image_counts.to(device=pixel_values.device, dtype=torch.long).clamp(min=0, max=M)

            valid_mask_2d = torch.arange(M, device=pixel_values.device).unsqueeze(0) < image_counts.unsqueeze(1)
            valid_indices = valid_mask_2d.reshape(-1).nonzero(as_tuple=True)[0]
            pixel_flat_all = pixel_values.reshape(B * M, *pixel_values.shape[2:])
            pixel_flat_valid = pixel_flat_all[valid_indices] if valid_indices.numel() > 0 else pixel_flat_all[:0]

            with torch.no_grad():
                if pixel_flat_valid.numel() > 0:
                    img_tokens_valid = self.vision_encoder(pixel_flat_valid)
                else:
                    img_tokens_valid = torch.zeros(
                        0,
                        P,
                        self.vision_encoder.hidden_size,
                        dtype=self.vision_encoder.torch_dtype,
                        device=pixel_values.device,
                    )

            img_tokens_bmpv = torch.zeros(
                B * M,
                P,
                self.vision_encoder.hidden_size,
                dtype=img_tokens_valid.dtype,
                device=img_tokens_valid.device,
            )
            if valid_indices.numel() > 0:
                img_tokens_bmpv[valid_indices] = img_tokens_valid

            img_tokens_bmpv = img_tokens_bmpv.reshape(B, M, P, self.vision_encoder.hidden_size)
            img_tokens_f32 = self._check_nan(img_tokens_bmpv.float(), "vision_output")
            img_lang_tokens_bmph = self.projector(img_tokens_f32.reshape(B * M, P, self.vision_encoder.hidden_size)).reshape(
                B, M, P, self.llm_hidden_size
            )
            img_lang_tokens_bmph = self._check_nan(img_lang_tokens_bmph, "projector_output")

            text_summary = self._check_nan(self._masked_text_summary(text_lang_tokens, attention_mask), "text_summary")
            image_summary = self._check_nan(img_lang_tokens_bmph.float().mean(dim=2), "image_summary")
            text_unit = torch.nn.functional.normalize(text_summary, p=2, dim=-1, eps=1e-6)
            image_unit = torch.nn.functional.normalize(image_summary, p=2, dim=-1, eps=1e-6)
            image_scores = (image_unit * text_unit.unsqueeze(1)).sum(dim=-1)
            image_scores = image_scores.masked_fill(~valid_mask_2d, -1e4)

            K = min(self.top_k_images, M)
            _, topk_indices = image_scores.topk(K, dim=1)
            topk_exp = topk_indices[:, :, None, None].expand(-1, -1, P, self.llm_hidden_size)
            selected_img_tokens = torch.gather(img_lang_tokens_bmph, 1, topk_exp)
            img_lang_tokens = selected_img_tokens.reshape(B, K * P, self.llm_hidden_size)
            img_lang_tokens = self._check_nan(img_lang_tokens, "topk_img_tokens")

            img_attn = torch.ones(B, K * P, device=pixel_values.device, dtype=torch.long)
        else:
            with torch.no_grad():
                img_tokens = self.vision_encoder(pixel_values)
            img_tokens_f32 = self._check_nan(img_tokens.float(), "vision_output")
            img_lang_tokens = self.projector(img_tokens_f32)
            img_lang_tokens = self._check_nan(img_lang_tokens, "projector_output")
            img_attn = torch.ones(B, img_lang_tokens.size(1), device=img_lang_tokens.device, dtype=torch.long)

        img_lang_tokens = img_lang_tokens.to(dtype=self.llm_dtype)
        text_lang_tokens = text_lang_tokens.to(device=img_lang_tokens.device, dtype=self.llm_dtype)

        aspect_ids = torch.arange(self.num_aspects, device=text_lang_tokens.device).unsqueeze(0).expand(B, -1)
        aspect_lang_tokens = self.aspect_token_embeddings(aspect_ids).to(dtype=self.llm_dtype)
        aspect_attn = torch.ones(B, self.num_aspects, device=img_lang_tokens.device, dtype=torch.long)

        combined_embeds = torch.cat([img_lang_tokens, text_lang_tokens, aspect_lang_tokens], dim=1)

        if attention_mask is not None:
            text_attn = attention_mask.to(device=img_lang_tokens.device, dtype=torch.long)
        else:
            text_attn = torch.ones(B, text_lang_tokens.size(1), device=img_lang_tokens.device, dtype=torch.long)

        full_attention_mask = torch.cat([img_attn, text_attn, aspect_attn], dim=1)
        return combined_embeds, full_attention_mask

    def _llm_hidden_states(self, combined_embeds: torch.Tensor, full_attention_mask: torch.Tensor) -> torch.Tensor:
        combined_embeds = combined_embeds.to(dtype=self.llm_dtype)

        outputs = self.llm(
            inputs_embeds=combined_embeds,
            attention_mask=full_attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1]

        if not torch.isfinite(hidden_states).all():
            print("[NaN-TRACE] Non-finite hidden_states detected, retry with explicit position_ids...")
            safe_mask = full_attention_mask > 0
            position_ids = safe_mask.long().cumsum(dim=-1) - 1
            position_ids = position_ids.clamp(min=0)

            outputs_retry = self.llm(
                inputs_embeds=combined_embeds,
                attention_mask=safe_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden_states = outputs_retry.hidden_states[-1]

        if not torch.isfinite(hidden_states).all():
            raise RuntimeError("NaN in LLM output hidden_states (including fallback).")

        return self._check_nan(hidden_states, "llm_hidden_states")

    def _cross_modal_fuse(
        self,
        hidden_states: torch.Tensor,
        full_attention_mask: torch.Tensor,
        text_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build explicit image<->text fusion from LLM hidden states.
        Returns:
            fused_context: [B, S, H]
            fused_mask:    [B, S] bool
            aspect_tokens: [B, A, H]
        """
        B, L, _ = hidden_states.shape
        A = self.num_aspects
        img_len = max(L - text_len - A, 0)
        txt_end = min(img_len + text_len, L)

        img_tokens = hidden_states[:, :img_len, :]
        txt_tokens = hidden_states[:, img_len:txt_end, :]

        # Keep aspect tokens from the tail as currently modeled.
        if L >= A:
            aspect_tokens = hidden_states[:, L - A : L, :]
        else:
            aspect_tokens = hidden_states[:, :0, :]

        img_mask = (full_attention_mask[:, :img_len] > 0) if img_len > 0 else full_attention_mask[:, :0].bool()
        txt_mask = (full_attention_mask[:, img_len:txt_end] > 0) if txt_end > img_len else full_attention_mask[:, :0].bool()

        # Fallback when one modality segment is empty.
        if img_tokens.size(1) == 0 or txt_tokens.size(1) == 0:
            fused_context = hidden_states[:, :txt_end, :]
            fused_mask = full_attention_mask[:, :txt_end] > 0
            return fused_context, fused_mask, aspect_tokens

        img_from_txt, _ = self.cross_modal_attn(
            query=img_tokens,
            key=txt_tokens,
            value=txt_tokens,
            key_padding_mask=~txt_mask,
            need_weights=False,
        )
        txt_from_img, _ = self.cross_modal_attn(
            query=txt_tokens,
            key=img_tokens,
            value=img_tokens,
            key_padding_mask=~img_mask,
            need_weights=False,
        )

        fused_context = torch.cat([img_from_txt, txt_from_img], dim=1)
        fused_mask = torch.cat([img_mask, txt_mask], dim=1)
        return fused_context, fused_mask, aspect_tokens

    def _aspect_guided_fuse_pool(
        self,
        aspect_tokens: torch.Tensor,
        fused_context: torch.Tensor,
        fused_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aspect-guided fusion + attention-style pooling.
        Output shape: [B, A, H]
        """
        if aspect_tokens.size(1) == 0:
            return aspect_tokens

        if fused_context.size(1) == 0:
            z = self.fusion_norm(aspect_tokens.to(dtype=self.fusion_norm.weight.dtype)).float()
            return z

        aspect_ctx, _ = self.aspect_guided_attn(
            query=aspect_tokens,
            key=fused_context,
            value=fused_context,
            key_padding_mask=~fused_mask,
            need_weights=False,
        )

        score = torch.einsum("bah,bsh->bas", aspect_tokens.float(), fused_context.float()) / math.sqrt(
            float(aspect_tokens.size(-1))
        )
        score = score.masked_fill(~fused_mask.unsqueeze(1), -1e4)
        weight = torch.softmax(score, dim=-1)
        pooled = torch.einsum("bas,bsh->bah", weight, fused_context.float())

        z = (aspect_ctx.float() + pooled).to(dtype=self.fusion_norm.weight.dtype)
        z = self.fusion_norm(z).float()
        return z

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        image_counts: torch.Tensor = None,
    ) -> dict:
        combined_embeds, full_attention_mask = self._get_combined_embeds(pixel_values, input_ids, attention_mask, image_counts)
        hidden_states = self._llm_hidden_states(combined_embeds, full_attention_mask)

        fused_context, fused_mask, aspect_tokens = self._cross_modal_fuse(
            hidden_states=hidden_states,
            full_attention_mask=full_attention_mask,
            text_len=input_ids.shape[1],
        )

        Z = self._aspect_guided_fuse_pool(
            aspect_tokens=aspect_tokens,
            fused_context=fused_context,
            fused_mask=fused_mask,
        )
        Z = self._check_nan(Z, "aspect_aware_pooled_Z").float()
        Z = self._check_nan(Z, "aspect_Z")
        Z = torch.clamp(Z, -1e3, 1e3)

        acd_logits = self.acd_head(Z).squeeze(-1)
        asc_logits = self.asc_head(Z)

        return {"acd_logits": acd_logits, "asc_logits": asc_logits}

    def get_trainable_params(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
