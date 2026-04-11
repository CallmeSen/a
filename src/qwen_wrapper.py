"""Hook-based LLM wrapper for Qwen2.5-7B-Instruct.

Uses forward hooks to inject visual cross-attention at the correct architectural position:
1. Hook decoder layer after self-attention to inject visual info via gated residual.
2. Pre-compute position embeddings in the wrapper to avoid rotary_emb hook conflicts.
3. No layer replacement — only hook injection points.

Supports two adapter types:
- GatedCrossAttentionAdapter (original): full-rank projections, single gate
- DualGatedCrossAttentionAdapter: low-rank dual-branch, ReLU gating (USE_DUAL_ADAPTER=1)
"""
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── GatedCrossAttentionAdapter ──────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * x).to(input_dtype)


class GatedCrossAttentionAdapter(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rms_norm = RMSNorm(hidden_size)

        # sigmoid(-2) ~ 0.12: visual injection ~12% from epoch 1, grows as gate trains
        self.gate = nn.Parameter(torch.tensor(-2.0))

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)

    def forward(
        self,
        text_hidden: torch.Tensor,
        visual_tokens: torch.Tensor,
        visual_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, q_len, hidden = text_hidden.shape
        _, kv_len, _ = visual_tokens.shape

        x_norm = self.rms_norm(text_hidden)
        q = self.q_proj(x_norm)
        k = self.k_proj(visual_tokens)
        v = self.v_proj(visual_tokens)

        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        if visual_mask is not None:
            invalid = ~visual_mask.bool()
            attn_scores = attn_scores.masked_fill(
                invalid[:, None, None, :],
                torch.finfo(attn_scores.dtype).min,
            )

        attn_probs = F.softmax(attn_scores.float(), dim=-1).to(dtype=attn_scores.dtype)
        attn_out = torch.matmul(attn_probs, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, hidden)
        attn_out = self.o_proj(attn_out)

        gate = torch.sigmoid(self.gate)
        hidden = text_hidden + gate * attn_out
        return hidden


# ─── Rotary embedding helper (extracted from Qwen2RotaryEmbedding) ─────────────

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_single(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors.

    This is the same logic as Qwen2's apply_rotary_pos_emb but for single tensors.
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def compute_qwen_rotary_embeddings(
    hidden_states: torch.Tensor, position_ids: torch.Tensor, rotary_emb_module
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Qwen2 rotary embeddings exactly as Qwen2RotaryEmbedding.forward does.

    Avoids hooking rotary_emb by replicating its logic here.
    This mirrors: Qwen2RotaryEmbedding.forward(x, position_ids) → (cos, sin).
    """
    inv_freq_expanded = (
        rotary_emb_module.inv_freq[None, :, None]
        .float()
        .expand(position_ids.shape[0], -1, 1)
        .to(hidden_states.device)
    )
    position_ids_expanded = position_ids[:, None, :].float()

    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()

    return cos.to(dtype=hidden_states.dtype), sin.to(dtype=hidden_states.dtype)


# ─── QwenLMWrapper ──────────────────────────────────────────────────────────

class QwenLMWrapper(nn.Module):
    """
    Hook-based wrapper for Qwen2.5.

    Uses hooks on decoder layer self-attention outputs to inject visual cross-attention.
    Position embeddings are computed directly (not via rotary_emb hook) to avoid
    decorator conflicts with @use_kernelized_func.

    Adapter injection position: after self-attention output, before residual addition.
    Supports two adapter types:
    - GatedCrossAttentionAdapter (USE_DUAL_ADAPTER=0): full-rank, single gate
    - DualGatedCrossAttentionAdapter (USE_DUAL_ADAPTER=1): low-rank dual-branch, ReLU gating
    """

    def __init__(
        self,
        qwen_for_casual_lm,
        num_layers: int,
        hidden_size: int,
        num_visual_tokens: int = 64,
        use_adapter_layers: Optional[List[int]] = None,
        adapter_num_heads: int = 8,
        use_dual_adapter: bool = False,
        dual_adapter_rank: int = 64,
    ):
        super().__init__()
        self.qwen_lm = qwen_for_casual_lm
        # qwen_for_casual_lm may be:
        #   - Qwen2ForCausalLM (no LoRA): qwen_lm.model = Qwen2Model (has embed_tokens)
        #   - PeftModel wrapping ForCausalLM (LoRA): qwen_lm.model = Qwen2ForCausalLM (no embed_tokens)
        #       → qwen_lm.model.model = Qwen2Model (has embed_tokens)
        # Traverse both cases safely to find Qwen2Model.
        base = self.qwen_lm
        while hasattr(base, 'model') and not hasattr(base, 'embed_tokens'):
            base = base.model
        self.qwen_base = base if hasattr(base, 'embed_tokens') else self.qwen_lm.model
        self._embed_tokens = self.qwen_base.embed_tokens
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_visual_tokens = num_visual_tokens
        self.use_dual_adapter = use_dual_adapter
        self.dual_adapter_rank = dual_adapter_rank

        if use_adapter_layers is None:
            use_adapter_layers = list(range(num_layers - 4, num_layers))
        self.use_adapter_layers = sorted(set(use_adapter_layers))

        # Select adapter class
        if use_dual_adapter:
            from .dual_adapter import DualGatedCrossAttentionAdapter

            adapter_class = DualGatedCrossAttentionAdapter
            print(f"[Dual Adapter] Using DualGatedCrossAttentionAdapter (rank={dual_adapter_rank})")
        else:
            adapter_class = GatedCrossAttentionAdapter
            print(f"[Adapter] Using GatedCrossAttentionAdapter (full-rank)")

        # Build one adapter per selected layer
        self.adapters = nn.ModuleDict()
        for layer_idx in self.use_adapter_layers:
            if use_dual_adapter:
                self.adapters[str(layer_idx)] = adapter_class(
                    hidden_size=hidden_size,
                    num_heads=adapter_num_heads,
                    rank=dual_adapter_rank,
                )
            else:
                self.adapters[str(layer_idx)] = adapter_class(
                    hidden_size=hidden_size,
                    num_heads=adapter_num_heads,
                )

        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._visual_tokens: Optional[torch.Tensor] = None
        self._visual_mask: Optional[torch.Tensor] = None
        self._allow_gradient = False  # Override: allow gradients when LoRA is active (LoRA params need gradients)

    def set_allow_gradient(self, value: bool):
        """Override detach behavior — True = allow gradients to flow (no detach for LoRA)."""
        self._allow_gradient = value

    def _make_attn_hook(self, layer_idx: int, backbone_frozen: bool):
        """Hook on self.self_attn output to inject gated visual cross-attention.

        Fires AFTER self-attention computes its output, BEFORE residual is added.
        Qwen2Attention.forward returns (attn_output, attn_weights), so we wrap the tuple.

        When backbone is frozen: DETACH the Qwen output before injecting adapter.
        This breaks the autograd graph so PyTorch does NOT retain activation memory
        for the frozen Qwen layers. Adapter gradients still flow correctly via the
        added residual (which is a trainable parameter).
        """
        adapter = self.adapters[str(layer_idx)]

        def hook(module, args, output):
            # output = (attn_output, attn_weights) from Qwen2Attention
            if not isinstance(output, tuple):
                attn_output = output
            else:
                attn_output = output[0]

            # CRITICAL: detach() breaks autograd graph for frozen backbone.
            # PyTorch will NOT store activation memory for this layer's computation.
            # Adapter output is added back — its gradient flows via trainable params.
            # NOTE: Never detach when LoRA is active — LoRA A/B params need gradients flowing
            # through the backbone's attention outputs to update correctly.
            if backbone_frozen and not self._allow_gradient:
                attn_output = attn_output.detach()

            if self._visual_tokens is None:
                return output

            # Cast to float32 for GatedCrossAttentionAdapter (float32 weights).
            # Adapter output (float32) gets mixed into bf16 residual — cast back.
            adapted_fp32 = adapter(
                text_hidden=attn_output.to(torch.float32),
                visual_tokens=self._visual_tokens.to(torch.float32),
                visual_mask=self._visual_mask,
            )
            adapted_bf16 = adapted_fp32.to(attn_output.dtype)

            # Return same type as original output:
            # Qwen2: tuple (attn_output, attn_weights) → return (adapted, None)
            # Qwen3: tensor attn_output (no tuple) → return adapted directly
            if isinstance(output, tuple):
                return (adapted_bf16, None)
            else:
                return adapted_bf16

        return hook

    def _register_hooks(self):
        self._unregister_hooks()
        # Capture backbone_frozen once at registration time.
        backbone_frozen = self._is_backbone_frozen()
        for layer_idx in self.use_adapter_layers:
            layer = self.qwen_base.layers[layer_idx]
            h = layer.self_attn.register_forward_hook(
                self._make_attn_hook(layer_idx, backbone_frozen), with_kwargs=False
            )
            self._hooks.append(h)

    def _unregister_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        visual_tokens: Optional[torch.Tensor] = None,
        visual_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Qwen2 forward with visual cross-attention injection.

        Key points:
        - Uses embed_tokens (not tok_embeddings like InternLM)
        - Hook on self.self_attn output for cross-attention injection
        - No per-layer RoPE recompute (RoPE fine in Qwen2 checkpoints)
        - When backbone is frozen (USE_LORA=0): sets qwen_base.eval() to prevent
          PyTorch from storing activation memory during .train() mode, avoiding OOM.
        """
        self._visual_tokens = visual_tokens
        self._visual_mask = visual_mask

        inputs_embeds = self._embed_tokens(text_input_ids)
        # Cast to bf16 to match Qwen model dtype (COMPUTE_DTYPE = bfloat16).
        # GatedCrossAttentionAdapter weights are float32 but autocast handles promotion.
        inputs_embeds = inputs_embeds.to(self.qwen_base.dtype)
        if visual_tokens is not None:
            visual_tokens = visual_tokens.to(self.qwen_base.dtype)

        backbone_frozen = self._is_backbone_frozen()
        # Always register hooks — they inject visual cross-attention via gated residual.
        # When backbone is frozen: detach() inside the hook prevents PyTorch from storing
        # activation memory for all 28 Qwen layers during backpropagation.
        # Adapter (GatedCrossAttentionAdapter) runs in both cases with trainable params.
        self._register_hooks()
        try:
            outputs = self.qwen_base(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                use_cache=False,
            )
            last_hidden = outputs.last_hidden_state
        finally:
            self._unregister_hooks()

        return last_hidden, outputs.hidden_states if output_hidden_states else None

    def _is_backbone_frozen(self) -> bool:
        """Returns True if Qwen base backbone is frozen (all requires_grad=False)."""
        return not any(p.requires_grad for p in self.qwen_base.parameters())


