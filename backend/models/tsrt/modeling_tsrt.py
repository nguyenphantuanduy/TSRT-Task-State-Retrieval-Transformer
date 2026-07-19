from __future__ import annotations
from collections.abc import Callable
from typing import Unpack
import torch
import torch.nn as nn
from transformers import Cache
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    apply_rotary_pos_emb,
    FlashAttentionKwargs,
    ALL_ATTENTION_FUNCTIONS,
    eager_attention_forward,
    rotate_half,
    repeat_kv,
    Qwen3Attention,
    Qwen3MLP,
)
from transformers.integrations import (
    use_kernel_forward_from_hub,
    use_kernel_func_from_hub,
    use_kernelized_func,
)

from transformers.utils import TransformersKwargs

from .configuration_tsrt import TSRTConfig
from .cache_utils import (
    TSRTDecoderCache,
    TSRTDocumentCache,
    TSRTEmbeddingCache,
    TSRTCache
)

from transformers.modeling_layers import GradientCheckpointingLayer

@use_kernel_func_from_hub("rotary_pos_emb")
def apply_rotary_pos_emb_query(
    q,
    cos,
    sin,
    unsqueeze_dim=1,
):
    """
    Apply RoPE to query states.

    Args:
        q:
            (B, H, Lq, d)

        cos, sin:
            (1, Lq, d)

    Returns:
        q_embed:
            (B, H, Lq, d)
    """

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # cos/sin:
    # (1, 1, Lq, d)
    #
    # Broadcast:
    # q    (B, H, Lq, d)
    # cos  (1, 1, Lq, d)

    q_embed = (
        q * cos
        + rotate_half(q) * sin
    )

    return q_embed

@use_kernel_func_from_hub("rotary_pos_emb")
def apply_rotary_pos_emb_multidoc_key(
    k,
    cos,
    sin,
):
    """
    Apply RoPE to multi-document key states.

    Args:
        k:
            (B, D, H, Lk, d)

        cos, sin:
            (1, Lk, d)

    Returns:
        k_embed:
            (B, D, H, Lk, d)
    """

    B, D, H, Lk, d = k.shape

    # (B, D, H, Lk, d)
    # -> (B * D, H, Lk, d)

    k = k.reshape(B * D, H, Lk, d)

    # (1, Lk, d)
    # -> (1, 1, Lk, d)

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    # Broadcast:
    # k    (B * D, H, Lk, d)
    # cos  (1, 1, Lk, d)

    k_embed = (
        k * cos
        + rotate_half(k) * sin
    )

    # (B * D, H, Lk, d)
    # -> (B, D, H, Lk, d)

    k_embed = k_embed.reshape(B, D, H, Lk, d)

    return k_embed

def tsrt_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    retrieval_memory: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    # ==========================================================
    # GQA
    # ==========================================================

    key_states = repeat_kv(
        key,
        module.num_key_value_groups,
    )

    value_states = repeat_kv(
        value,
        module.num_key_value_groups,
    )

    # ==========================================================
    # Attention logits
    # ==========================================================

    attn_weights = (
        torch.matmul(
            query,
            key_states.transpose(2, 3),
        )
        * scaling
    )

    # ==========================================================
    # Retrieval bias
    # ==========================================================

    B, L, D = retrieval_memory.shape
    _, _, total_kv_len, _ = key.shape

    Lk = total_kv_len // D

    retrieval_bias = (
        1
        + torch.log(
            torch.clamp(
                (retrieval_memory + 1) / 2,
                min=1e-6,
            )
        )
    )

    retrieval_bias = (
        retrieval_bias
        .unsqueeze(-1)                 # (B, L, D, 1)
        .expand(-1, -1, -1, Lk)        # (B, L, D, Lk)
        .reshape(B, L, D * Lk)         # (B, L, D*Lk)
        .unsqueeze(1)                  # (B, 1, L, D*Lk)
    )

    attn_weights = attn_weights + retrieval_bias

    # ==========================================================
    # Attention mask
    # ==========================================================

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # ==========================================================
    # Softmax
    # ==========================================================

    attn_weights = nn.functional.softmax(
        attn_weights,
        dim=-1,
        dtype=torch.float32,
    ).to(query.dtype)

    attn_weights = nn.functional.dropout(
        attn_weights,
        p=dropout,
        training=module.training,
    )

    # ==========================================================
    # Output
    # ==========================================================

    attn_output = torch.matmul(
        attn_weights,
        value_states,
    )

    attn_output = (
        attn_output
        .transpose(1, 2)
        .contiguous()
    )

    return attn_output, attn_weights

@use_kernelized_func(apply_rotary_pos_emb)
class TSRTCrossAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config: TSRTConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None
    
    def forward(
        self,
        decoder_hidden_states: torch.Tensor,                        # (B, L, h)
        encoder_hidden_states: torch.Tensor,                        # (B, D, L', h)
        retrieval_memory: torch.Tensor,                             # (B, L, D)
        decoder_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        encoder_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: TSRTDocumentCache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # ==========================================================
        # Query
        # ==========================================================

        query_input_shape = decoder_hidden_states.shape[:-1]        # (B, L)

        query_hidden_shape = (
            *query_input_shape,
            -1,
            self.head_dim,
        )                                                           # (B, L, H, d)

        query_states = self.q_norm(
            self.q_proj(decoder_hidden_states)
            .view(query_hidden_shape)
        ).transpose(1, 2)                                           # (B, H, L, d)

        decoder_cos, decoder_sin = decoder_position_embeddings

        query_states = apply_rotary_pos_emb_query(
            q=query_states,
            cos=decoder_cos,
            sin=decoder_sin,
        )

        # ==========================================================
        # Key / Value
        # ==========================================================

        has_cached_kv = (
            past_key_values is not None
            and past_key_values.has_kv(self.layer_idx)
        )

        if has_cached_kv:

            key_states, value_states = past_key_values.get_kv(
                self.layer_idx
            )

        else:

            key_value_input_shape = encoder_hidden_states.shape[:-1]    # (B, D, L')

            key_value_hidden_shape = (
                *key_value_input_shape,
                -1,
                self.head_dim,
            )                                                           # (B, D, L', H_kv, d)

            key_states = self.k_norm(
                self.k_proj(encoder_hidden_states)
                .view(key_value_hidden_shape)
            ).transpose(2, 3)                                           # (B, D, H_kv, L', d)

            value_states = (
                self.v_proj(encoder_hidden_states)
                .view(key_value_hidden_shape)
            ).transpose(2, 3)                                           # (B, D, H_kv, L', d)

            encoder_cos, encoder_sin = encoder_position_embeddings

            key_states = apply_rotary_pos_emb_multidoc_key(
                k=key_states,
                cos=encoder_cos,
                sin=encoder_sin,
            )

            if past_key_values is not None:

                key_states, value_states = past_key_values.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                )
            
        B, D, H_kv, Lk, d = key_states.shape

        key_states = (
            key_states
            .transpose(1, 2)
            .reshape(B, H_kv, D * Lk, d)
        )

        value_states = (
            value_states
            .transpose(1, 2)
            .reshape(B, H_kv, D * Lk, d)
        )

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation,
            tsrt_eager_attention_forward,
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            retrieval_memory,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        # (B, L, H, d)
        attn_output = (
            attn_output
            .reshape(*query_input_shape, -1)
            .contiguous()
        )

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights
    

class TSRTRetrievalDecisionHead(nn.Module):
    """
    Predict retrieval probability for each decoder token.

    Input:
        hidden_states:
            (B, L, h)

    Output:
        retrieval_decision:
            (B, L, 1)

            range:
                [0, 1]

            0 -> no retrieval
            1 -> retrieve
    """

    def __init__(self, config: TSRTConfig):
        super().__init__()

        hidden_size = config.hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(
                hidden_size,
                hidden_size,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_size,
                1,
            ),
        )

        self.activation = nn.Sigmoid()

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states:
                (B, L, h)

        Returns:
            retrieval_decision:
                (B, L, 1)
        """

        logits = self.mlp(hidden_states)

        decision = self.activation(logits)

        return decision
    

class TSRTRetrievalProjection(nn.Module):

    def __init__(self, config: TSRTConfig):
        super().__init__()

        self.config = config

        retrieval_dim = getattr(
            config,
            "retrieval_embedding_size",
            config.hidden_size,
        )

        # h -> h'
        self.proj = nn.Sequential(
            nn.Linear(
                config.hidden_size,
                retrieval_dim,
            ),
            nn.GELU(),
        )

        # h' -> 1
        self.score = nn.Sequential(
            nn.Linear(
                retrieval_dim,
                1,
            )
        )

    @staticmethod
    def emb_calc(
        hidden_embs: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args

        hidden_embs:
            (B, L, h')
            or
            (B, D, L, h')

        weights:
            (B, L, 1)
            or
            (B, D, L, 1)

        Returns

        (B, 1, h')
        or
        (B, D, h')
        """

        weights = torch.softmax(
            weights,
            dim=-2,
        )

        weighted_embs = hidden_embs * weights

        return weighted_embs.sum(
            dim=-2,
        )

    def forward(
        self,
        decoder_hidden_state: torch.Tensor,     # (B, L, h)
        encoder_hidden_state: torch.Tensor,     # (B, D, L', h)
        past_embedding: TSRTEmbeddingCache | None,
        attention_mask: torch.Tensor | None = None,
    ):
        # ==================================================
        # Decoder projection
        # ==================================================

        decoder_hidden_embs = self.proj(
            decoder_hidden_state
        )                                       # (B, L, h')

        decoder_weights = self.score(
            decoder_hidden_embs
        )                                       # (B, L, 1)

        # ==================================================
        # Cache update
        # ==================================================

        if past_embedding is not None:

            decoder_weights, decoder_hidden_embs = (
                past_embedding.update(
                    decoder_weights,
                    decoder_hidden_embs,
                )
            )

            decoder_embs = self.emb_calc(decoder_hidden_embs, decoder_weights).unsqueeze(1)
        
        else:

            # ==================================================
            # Prefix embeddings
            # ==================================================

            seq_len = decoder_hidden_embs.shape[1]

            decoder_embs = []

            for idx in range(seq_len):

                emb = self.emb_calc(
                    decoder_hidden_embs[:, : idx + 1],
                    decoder_weights[:, : idx + 1],
                )                                   # (B, h')

                decoder_embs.append(
                    emb.unsqueeze(1)
                )                                   # (B, 1, h')

            decoder_embs = torch.cat(
                decoder_embs,
                dim=1,
            )                                       # (B, L, h')

        # ==================================================
        # Document embeddings
        # ==================================================

        if (
            past_embedding is not None
            and past_embedding.has_doc_embs()
        ):

            doc_embs = past_embedding.get_doc_embs()

        else:

            encoder_hidden_embs = self.proj(
                encoder_hidden_state
            )                                   # (B, D, L', h')

            encoder_weights = self.score(
                encoder_hidden_embs
            )                                   # (B, D, L', 1)

            doc_embs = self.emb_calc(
                encoder_hidden_embs,
                encoder_weights,
            )                                   # (B, D, h')

            if past_embedding is not None:

                doc_embs = past_embedding.update_doc_embs(
                    doc_embs
                )

        return (
            decoder_embs,   # (B, L, h') or (B, 1, h')
            doc_embs,       # (B, D, h')
        )


class TSRTLayer(GradientCheckpointingLayer):

    def __init__(self, config: TSRTConfig, self_attn_layer_idx: int, cross_attn_layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3Attention(config=config, layer_idx=self_attn_layer_idx)
        self.mlp = Qwen3MLP(config)
        self.cross_attn = TSRTCrossAttention(config, cross_attn_layer_idx)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_self_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_cross_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        decoder_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        retrieval_memory: torch.Tensor,
        decoder_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        encoder_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        self_attention_mask: torch.Tensor | None = None,
        cross_attention_mask: torch.Tensor | None = None,
        self_attn_past_key_values: Cache | None = None,
        cross_attn_past_key_values: TSRTDocumentCache | None = None,
        use_cache: bool | None = False,
        position_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        residual = decoder_hidden_states
        decoder_hidden_states = self.input_layernorm(decoder_hidden_states)

        # Self Attention
        decoder_hidden_states, _ = self.self_attn(
            hidden_states=decoder_hidden_states,
            attention_mask=self_attention_mask,
            position_ids=position_ids,
            past_key_values=self_attn_past_key_values,
            use_cache=use_cache,
            position_embeddings=decoder_position_embeddings,
            **kwargs,
        )

        decoder_hidden_states = residual + decoder_hidden_states

        residual = decoder_hidden_states
        decoder_hidden_states = self.post_self_attention_layernorm(decoder_hidden_states)

        # Cross Attention
        decoder_hidden_states, _ = self.cross_attn(
            decoder_hidden_states=decoder_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            retrieval_memory=retrieval_memory,
            decoder_position_embeddings=decoder_position_embeddings,
            encoder_position_embeddings=encoder_position_embeddings,
            attention_mask=cross_attention_mask,
            past_key_values=cross_attn_past_key_values,
            use_cache=use_cache,
            position_ids=position_ids,
            **kwargs,
        )

        decoder_hidden_states = residual + decoder_hidden_states

        # Fully Connected
        residual = decoder_hidden_states
        decoder_hidden_states = self.post_cross_attention_layernorm(decoder_hidden_states)
        decoder_hidden_states = self.mlp(decoder_hidden_states)
        decoder_hidden_states = residual + decoder_hidden_states

        return decoder_hidden_states