from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedCrossAttentionAdapter(nn.Module):
    """Post-LLM visual injection block.

    Text hidden states attend to visual tokens, then receive gated residual injection.

    Input:
        text_hidden:   [B, L_text, H]
        visual_tokens: [B, N_vis, H]
        visual_mask:   [B, N_vis] bool, True = valid, False = padding
    Output:
        hidden:        [B, L_text, H]
    """

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

        self.ln = nn.LayerNorm(hidden_size)

        # sigmoid(-10) ~ 0, so visual injection starts almost disabled
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

        q = self.q_proj(text_hidden)
        k = self.k_proj(visual_tokens)
        v = self.v_proj(visual_tokens)

        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)   # [B, h, L, d]
        k = k.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, N, d]
        v = v.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, N, d]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # [B, h, L, N]

        if visual_mask is not None:
            invalid = ~visual_mask.bool()
            attn_scores = attn_scores.masked_fill(
                invalid[:, None, None, :],
                torch.finfo(attn_scores.dtype).min,
            )

        attn_probs = F.softmax(attn_scores.float(), dim=-1).to(dtype=attn_scores.dtype)
        attn_out = torch.matmul(attn_probs, v)  # [B, h, L, d]

        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, hidden)
        attn_out = self.o_proj(attn_out)

        gate = torch.sigmoid(self.gate)
        hidden = self.ln(text_hidden + gate * attn_out)
        return hidden


class InternLMWrapper(nn.Module):
    """Layer-by-layer wrapper around InternLM2 base model.

    InternLM2DecoderLayer forward flow (pre-norm):
        residual = hidden_states
        hidden_states = attention_norm(hidden_states)
        attn_out = attention(hidden_states)
        hidden_states = residual + attn_out
        hidden_states = ffn_norm(hidden_states)
        hidden_states = residual + feed_forward(hidden_states)

    This wrapper replicates that flow step-by-step and injects a GatedCrossAttentionAdapter
    at the CORRECT position: BETWEEN self-attention output and FFN input.
    This preserves pre-norm, causal mask, padding mask, cache_position, and sdpa path.

    Frozen parts: embed_tokens, all attention/FFN weights in decoder layers.
    Trainable: GatedCrossAttention adapters only.
    """

    def __init__(
        self,
        internlm_base_model,
        num_layers: int,
        hidden_size: int,
        num_visual_tokens: int = 64,
        use_adapter_layers: Optional[List[int]] = None,
        adapter_num_heads: int = 8,
    ):
        super().__init__()
        self.internlm_base = internlm_base_model
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_visual_tokens = num_visual_tokens

        if use_adapter_layers is None:
            use_adapter_layers = list(range(num_layers - 4, num_layers))
        self.use_adapter_layers = sorted(set(use_adapter_layers))

        # We keep one adapter module per requested "layer index" for naming / flexibility,
        # but they are applied as a post-LLM adapter stack.
        self.adapters = nn.ModuleDict()
        for layer_idx in self.use_adapter_layers:
            self.adapters[str(layer_idx)] = GatedCrossAttentionAdapter(
                hidden_size=hidden_size,
                num_heads=adapter_num_heads,
            )

    def _has_nonfinite(self, x: torch.Tensor) -> bool:
        return not torch.isfinite(x).all()

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
        Args:
            text_input_ids:    [B, L_text]
            text_position_ids: [B, L_text]
            attention_mask:    [B, L_text] padding mask
            visual_tokens:     [B, N_vis, H] or None
            visual_mask:       [B, N_vis] bool, True=valid
            output_hidden_states: whether to return intermediate states
            cache_position:    [L] positions for KV cache

        Returns:
            last_hidden: [B, L_text, H]
            all_hidden_states: Optional[List[[B, L_text, H]]]
        """
        # Get token embeddings (frozen) — InternLM2Model uses tok_embeddings
        inputs_embeds = self.internlm_base.tok_embeddings(text_input_ids)

        all_hidden_states: Optional[List[torch.Tensor]] = []
        if output_hidden_states:
            all_hidden_states.append(inputs_embeds)

        hidden_states = inputs_embeds

        for layer_idx, layer in enumerate(self.internlm_base.layers):
            # Step 1: residual = hidden_states
            residual = hidden_states

            # Step 2: pre-norm -> self-attention
            normed_hidden = layer.attention_norm(hidden_states)
            attn_out, self_attn_weights, present_key_value = layer.attention(
                hidden_states=normed_hidden,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=cache_position,
            )
            # Step 3: residual + attn_output
            hidden_states = residual + attn_out

            # Step 4: Inject cross-attention adapter BETWEEN self-attn and FFN
            #         (correct position per Diagram.md)
            if layer_idx in self.use_adapter_layers and visual_tokens is not None:
                # causal_vis_mask=None: no causal restriction on visual tokens
                hidden_states = self.adapters[str(layer_idx)](
                    text_hidden=hidden_states,
                    visual_tokens=visual_tokens,
                    visual_mask=None,
                )

            if self._has_nonfinite(hidden_states):
                if output_hidden_states:
                    return hidden_states, all_hidden_states
                return hidden_states, None

            # Step 5: residual = hidden_states (post-attn)
            residual = hidden_states

            # Step 6: pre-norm -> FFN
            normed_hidden = layer.ffn_norm(hidden_states)
            ffn_out = layer.feed_forward(normed_hidden)

            # Step 7: residual + FFN output -> next layer input
            hidden_states = residual + ffn_out

            if self._has_nonfinite(hidden_states):
                if output_hidden_states:
                    return hidden_states, all_hidden_states
                return hidden_states, None

            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        # Final norm
        hidden_states = self.internlm_base.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return hidden_states, all_hidden_states