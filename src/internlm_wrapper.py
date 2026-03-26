from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import COMPUTE_DTYPE


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

        # sigmoid(-10) ~ 0, visual injection starts nearly disabled
        self.gate = nn.Parameter(torch.tensor(-10.0))

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


class InternLMWrapper(nn.Module):
    """Hook-based wrapper for InternLM2.

    Uses PyTorch forward hooks to inject GatedCrossAttention adapters directly
    into specific decoder layers of the frozen InternLM2 model. This avoids
    manually replicating layer internals (QKV split, RoPE, GQA) which can
    cause numerical instability on some GPU configurations.

    Cross-attention adapters are injected BETWEEN self-attention and FFN
    (the correct position per the architecture diagram).
    """

    def __init__(
        self,
        internlm_for_casual_lm,
        num_layers: int,
        hidden_size: int,
        num_visual_tokens: int = 64,
        use_adapter_layers: Optional[List[int]] = None,
        adapter_num_heads: int = 8,
    ):
        super().__init__()
        self.internlm_lm = internlm_for_casual_lm
        self.internlm_base = self.internlm_lm.model
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_visual_tokens = num_visual_tokens

        if use_adapter_layers is None:
            use_adapter_layers = list(range(num_layers - 4, num_layers))
        self.use_adapter_layers = sorted(set(use_adapter_layers))

        self.adapters = nn.ModuleDict()
        for layer_idx in self.use_adapter_layers:
            self.adapters[str(layer_idx)] = GatedCrossAttentionAdapter(
                hidden_size=hidden_size,
                num_heads=adapter_num_heads,
            )

        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._visual_tokens: Optional[torch.Tensor] = None
        self._visual_mask: Optional[torch.Tensor] = None

    def _make_hook(self, layer_idx: int):
        """Hook on decoder layer output to inject GatedCrossAttention."""
        adapter = self.adapters[str(layer_idx)]
        def hook(module, args, output):
            # InternLM2DecoderLayer returns tuple (hidden_states,)
            hidden = output[0] if isinstance(output, tuple) else output
            if self._visual_tokens is None:
                return output
            new_hidden = adapter(
                text_hidden=hidden,
                visual_tokens=self._visual_tokens,
                visual_mask=self._visual_mask,
            )
            return (new_hidden,)
        return hook

    def _register_hooks(self):
        self._unregister_hooks()
        for layer_idx in self.use_adapter_layers:
            layer = self.internlm_base.layers[layer_idx]
            h = layer.register_forward_hook(self._make_hook(layer_idx))
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
        Uses the full InternLM2ForCausalLM model forward path via input_embeds.
        Cross-attention adapters are injected via hooks at the correct position
        (between self-attn and FFN within each decoder layer).

        Supports two input shapes:
        - [B, L]          — standard single-aspect per sample
        - [B, A, L]       — multi-aspect (A aspects per sample); A dimension is
                            flattened into B for the InternLM forward pass and
                            restored on the output so callers see consistent shapes.
        """
        # Detect multi-aspect input [B, A, L] and flatten to [B*A, L]
        orig_shape = text_input_ids.shape   # [B, L] or [B, A, L]
        if text_input_ids.dim() == 3:
            B, A, L = text_input_ids.shape
            text_input_ids = text_input_ids.reshape(B * A, L)    # [B*A, L]
            if attention_mask is not None:
                attention_mask = attention_mask.reshape(B * A, L)
            if visual_tokens is not None:
                # Expand visual tokens: [B, N, D] -> [B*A, N, D]
                visual_tokens = visual_tokens.unsqueeze(1).expand(B, A, *visual_tokens.shape[1:]).reshape(B * A, *visual_tokens.shape[1:])
            if visual_mask is not None:
                visual_mask = visual_mask.unsqueeze(1).expand(B, A, *visual_mask.shape[1:]).reshape(B * A, *visual_mask.shape[1:])
        else:
            A = None  # no multi-aspect dim

        self._visual_tokens = visual_tokens
        self._visual_mask = visual_mask

        inputs_embeds = self.internlm_base.tok_embeddings(text_input_ids)

        # RECOMPUTE rotary_emb.inv_freq for every layer.
        # The inv_freq values stored in the checkpoint are corrupted (values like -1e41),
        # causing the RoPE matmul to overflow and produce NaN. We fix this by
        # recomputing inv_freq from the layer's own config (dim, base=rope_theta).
        target_device = inputs_embeds.device
        for layer in self.internlm_base.layers:
            rotary = layer.attention.rotary_emb
            dim = rotary.dim
            base = rotary.base  # rope_theta
            new_inv = (1.0 / (
                base ** (
                    torch.arange(0, dim, 2, dtype=torch.float32, device=target_device) / dim
                )
            ))
            rotary.inv_freq = nn.Parameter(new_inv, requires_grad=False)

        self._register_hooks()
        try:
            outputs = self.internlm_base(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=False,
                output_hidden_states=output_hidden_states,
            )
            last_hidden = outputs.last_hidden_state
        finally:
            self._unregister_hooks()
            self._visual_tokens = None
            self._visual_mask = None

        # Restore original aspect dimension if multi-aspect input was detected
        if A is not None:
            last_hidden = last_hidden.view(B, A, -1, last_hidden.size(-1))  # [B, A, L, D]

        return last_hidden, outputs.hidden_states if output_hidden_states else None
