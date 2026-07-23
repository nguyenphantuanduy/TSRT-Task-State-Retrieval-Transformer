from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from typing import Unpack
import torch
import torch.nn as nn
from transformers import Cache
from typing import Tuple
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
    Qwen3DecoderLayer, 
    Qwen3RotaryEmbedding,
)
from transformers.integrations import (
    use_kernel_forward_from_hub,
    use_kernel_func_from_hub,
    use_kernelized_func,
)
import torch.nn.functional as F
from transformers.utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
)
from transformers.utils.generic import (
    maybe_autocast,
    merge_with_config_defaults,
)

from transformers.utils.output_capturing import (
    capture_outputs,
)

from .configuration_tsrt import TSRTConfig
from .cache_utils import (
    TSRTDecoderCache,
    TSRTDocumentCache,
    TSRTEmbeddingCache,
    TSRTChosenDocumentCache,
    TSRTCache
)
from .utils import (
    prepare_document_attention_mask, 
    prepare_cross_attention_mask, 
    prepare_projection_mask, 
    compute_retrieval_decision_loss,
    compute_retrieval_ranking_loss,
    compute_retrieval_scoring_loss,
    compute_positive_negative_scores,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_utils import PreTrainedModel

from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)

from .modeling_outputs import TSRTModelOutputWithPast
from transformers.generation import GenerationMixin

RETRIEVAL_DECISION_THRESHOLD = 0.7

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
        attention_mask: torch.Tensor | None = None,
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

        attention_mask:
            (B, L)
            or
            (B, D, L)

        Returns

        (B, h')
        or
        (B, D, h')
        """

        if attention_mask is not None:
            weights = weights + attention_mask.unsqueeze(-1)

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
        document_attention_mask: torch.Tensor | None = None  # (B, D, L')
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
                document_attention_mask
            )                                   # (B, D, h')

            if past_embedding is not None:

                doc_embs = past_embedding.update_doc_embs(
                    doc_embs
                )

        return (
            decoder_embs,   # (B, L, h') or (B, 1, h')
            doc_embs,       # (B, D, h')
        )


@dataclass
class TSRTRetrievalMemory:
    """
    Retrieval memory used by TSRT cross-attention.
    """

    retrieval_memory: torch.Tensor
    encoder_hidden_states: torch.Tensor
    document_padding_mask: torch.Tensor

class TSRTRetrievalMemoryHead(nn.Module):
    """
    Update retrieval memory from usefulness scores and retrieval decisions.
    """

    def forward(
        self,
        usefulness_score: torch.Tensor,      # (B, L, D)
        retrieval_decision: torch.Tensor,    # (B, L, 1)
        encoder_hidden_states: torch.Tensor, # (B, D, L', h)
        document_padding_mask: torch.Tensor, # (B, D, L')
        cache: TSRTChosenDocumentCache,
        retrieve_top_k: int | None = None,
        usefulness_threshold: float | None = None,
    ) -> TSRTRetrievalMemory:
        
        _, _, L_doc, hidden_size = encoder_hidden_states.shape

        if retrieve_top_k is None and usefulness_threshold is None:
            seq_len = usefulness_score.shape[1]

            retrieval_memory = []

            for i in range(seq_len):
                current_usefulness = usefulness_score[:, i]        # (B, D)
                current_decision = retrieval_decision[:, i]        # (B, 1)

                if i == 0:
                    current_memory = (
                        current_usefulness
                        * current_decision
                    )
                else:
                    current_memory = (
                        current_usefulness
                        * current_decision
                        + retrieval_memory[-1]
                        * (1 - current_decision)
                    )

                retrieval_memory.append(current_memory)

            retrieval_memory = torch.stack(
                retrieval_memory,
                dim=1,
            )   # (B, L, D)

            return TSRTRetrievalMemory(
                retrieval_memory=retrieval_memory,
                encoder_hidden_states=encoder_hidden_states,
                document_padding_mask=document_padding_mask,
            )
        
        else:
            retrieval_decision = retrieval_decision[:, -1, :].squeeze(-1)       # (B,)
            usefulness_score = usefulness_score[:, -1, :]                       # (B, D)

            current_choosen_document = []
            current_retrieval_memory = []
            current_document_padding_mask = []

            last_chosen_document, last_document_padding_mask, last_retrieval_memory = cache.get() if cache is not None else (None, None, None)

            for sample_idx in range(retrieval_decision.shape[0]):
                sample_retrieval_decision = retrieval_decision[sample_idx]   # scalar tensor
                sample_usefulness_score = usefulness_score[sample_idx,:]     # (D,)
                sample_encoder_hidden_states = encoder_hidden_states[sample_idx, :, :, :]   # (D, L', h)
                sample_document_padding_mask = document_padding_mask[sample_idx, :, :]      # (D, L')

                if sample_retrieval_decision >= RETRIEVAL_DECISION_THRESHOLD:
                    # (D,)
                    valid_document_mask = sample_document_padding_mask.any(dim=-1)

                    # Chỉ giữ các document không phải padding
                    valid_encoder_hidden_states = sample_encoder_hidden_states[valid_document_mask]   # (D_valid, L', h)
                    valid_usefulness_score = sample_usefulness_score[valid_document_mask]              # (D_valid,)
                    valid_document_padding_mask = sample_document_padding_mask[valid_document_mask]    # (D_valid, L')

                    if usefulness_threshold is not None and valid_usefulness_score.size(0) > 0:
                        usefulness_mask = valid_usefulness_score >= usefulness_threshold

                        valid_encoder_hidden_states = valid_encoder_hidden_states[usefulness_mask]
                        valid_usefulness_score = valid_usefulness_score[usefulness_mask]
                        valid_document_padding_mask = valid_document_padding_mask[usefulness_mask]

                    if (
                        retrieve_top_k is not None
                        and valid_usefulness_score.size(0) > 0
                    ):
                        k = min(retrieve_top_k, valid_usefulness_score.size(0))

                        topk_indices = torch.topk(
                            valid_usefulness_score,
                            k=k,
                            dim=0,
                            largest=True,
                            sorted=True,
                        ).indices

                        valid_encoder_hidden_states = valid_encoder_hidden_states[topk_indices]
                        valid_usefulness_score = valid_usefulness_score[topk_indices]
                        valid_document_padding_mask = valid_document_padding_mask[topk_indices]

                    valid_retrieval_memory = valid_usefulness_score * sample_retrieval_decision

                else:
                    if last_chosen_document is not None:
                        valid_encoder_hidden_states = last_chosen_document[sample_idx]          # (D_cache, L', h)
                        valid_document_padding_mask = last_document_padding_mask[sample_idx]    # (D_cache, L')
                        valid_retrieval_memory = last_retrieval_memory[sample_idx]              # (D_cache,)
                    else:
                        valid_encoder_hidden_states = encoder_hidden_states.new_empty(
                            (0, L_doc, hidden_size)
                        )  # (0, L', h)

                        valid_document_padding_mask = document_padding_mask.new_empty(
                            (0, L_doc)
                        )  # (0, L')

                        valid_retrieval_memory = usefulness_score.new_empty(
                            (0,)
                        )  # (0,)

                current_choosen_document.append(valid_encoder_hidden_states)
                current_retrieval_memory.append(valid_retrieval_memory)
                current_document_padding_mask.append(valid_document_padding_mask)

            D_max = 0
            L_max = 0

            for padding_mask in current_document_padding_mask:
                if padding_mask.numel() == 0:
                    continue

                # (D,)
                valid_doc_mask = padding_mask.any(dim=-1)
                D_real = valid_doc_mask.sum().item()
                D_max = max(D_max, D_real)

                if D_real > 0:
                    # (D_real,)
                    token_lengths = padding_mask[valid_doc_mask].sum(dim=-1)
                    L_real = token_lengths.max().item()
                    L_max = max(L_max, L_real)

            # Không còn document nào trong toàn batch
            if D_max == 0:
                return TSRTRetrievalMemory(
                    retrieval_memory=None,
                    encoder_hidden_states=None,
                    document_padding_mask=None,
                )
            
            # ------------------------------------------------------------
            # Pad & stack
            # ------------------------------------------------------------
            batch_encoder_hidden_states = []
            batch_retrieval_memory = []
            batch_document_padding_mask = []

            for hidden_states, retrieval_memory, padding_mask in zip(
                current_choosen_document,
                current_retrieval_memory,
                current_document_padding_mask,
            ):
                # --------------------------------------------------------
                # Empty sample
                # --------------------------------------------------------
                if hidden_states.size(0) == 0:
                    batch_encoder_hidden_states.append(
                        hidden_states.new_zeros(D_max, L_max, hidden_size)
                    )

                    batch_retrieval_memory.append(
                        retrieval_memory.new_zeros(D_max)
                    )

                    batch_document_padding_mask.append(
                        padding_mask.new_zeros(D_max, L_max)
                    )

                    continue

                # --------------------------------------------------------
                # Remove padded docs
                # --------------------------------------------------------
                valid_doc_mask = padding_mask.any(dim=-1)

                hidden_states = hidden_states[valid_doc_mask]
                retrieval_memory = retrieval_memory[valid_doc_mask]
                padding_mask = padding_mask[valid_doc_mask]

                D_real = hidden_states.size(0)

                # --------------------------------------------------------
                # Truncate sequence length to L_max
                # --------------------------------------------------------
                hidden_states = hidden_states[:, :L_max]
                padding_mask = padding_mask[:, :L_max]

                # --------------------------------------------------------
                # Pad L
                # --------------------------------------------------------
                pad_L = L_max - hidden_states.size(1)

                if pad_L > 0:
                    hidden_states = F.pad(
                        hidden_states,
                        (0, 0, 0, pad_L),
                        value=0,
                    )

                    padding_mask = F.pad(
                        padding_mask,
                        (0, pad_L),
                        value=0,
                    )

                # --------------------------------------------------------
                # Pad D
                # --------------------------------------------------------
                pad_D = D_max - D_real

                if pad_D > 0:
                    hidden_states = torch.cat(
                        [
                            hidden_states,
                            hidden_states.new_zeros(
                                pad_D,
                                L_max,
                                hidden_size,
                            ),
                        ],
                        dim=0,
                    )

                    retrieval_memory = torch.cat(
                        [
                            retrieval_memory,
                            retrieval_memory.new_zeros(
                                pad_D,
                            ),
                        ],
                        dim=0,
                    )

                    padding_mask = torch.cat(
                        [
                            padding_mask,
                            padding_mask.new_zeros(
                                pad_D,
                                L_max,
                            ),
                        ],
                        dim=0,
                    )

                batch_encoder_hidden_states.append(hidden_states)
                batch_retrieval_memory.append(retrieval_memory)
                batch_document_padding_mask.append(padding_mask)

            encoder_hidden_states = torch.stack(batch_encoder_hidden_states, dim=0)
            retrieval_memory = torch.stack(batch_retrieval_memory, dim=0).unsqueeze(1) 
            document_padding_mask = torch.stack(batch_document_padding_mask, dim=0)

            cache.update(
                chosen_document=encoder_hidden_states,
                document_padding_mask=document_padding_mask,
                retrieval_memory=retrieval_memory,
            )

            return TSRTRetrievalMemory(
                retrieval_memory=retrieval_memory,
                encoder_hidden_states=encoder_hidden_states,
                document_padding_mask=document_padding_mask,
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

        if encoder_hidden_states is not None: 
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

            
            residual = decoder_hidden_states
            decoder_hidden_states = self.post_cross_attention_layernorm(decoder_hidden_states)

        # Fully Connected
        decoder_hidden_states = self.mlp(decoder_hidden_states)
        decoder_hidden_states = residual + decoder_hidden_states

        return decoder_hidden_states


class TSRTPreTrainedModel(PreTrainedModel):
    config: TSRTConfig

    config_class = TSRTConfig

    base_model_prefix = "model"

    supports_gradient_checkpointing = True

    _no_split_modules = [
        "Qwen3DecoderLayer",
        "TSRTLayer",
        "TSRTRetrievalProjection",
        "TSRTRetrievalDecisionHead",
    ]

    _skip_keys_device_placement = ["past_key_values"]

    _supports_flash_attn = False
    _supports_sdpa = False
    _supports_flex_attn = False

    _can_compile_fullgraph = False
    _supports_attention_backend = False

    _can_record_outputs = {
        "hidden_states": TSRTLayer,
        "self_attentions": Qwen3Attention,
        "cross_attentions": TSRTCrossAttention,
    }

class TSRTModel(TSRTPreTrainedModel):
    def __init__(self, config: TSRTConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.decoder_layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_decoder_layers)]
        )
        
        self.encoder_layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_encoder_layers)]
        )

        self.tsrt_layers = nn.ModuleList(
            [TSRTLayer(config, layer_idx + config.num_decoder_layers, layer_idx) for layer_idx in range(config.num_tsrt_layers)]
        )
        
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        self.retrieval_decision_head = TSRTRetrievalDecisionHead(config)
        self.retrieval_projection = TSRTRetrievalProjection(config)
        self.retrieval_memory_head = TSRTRetrievalMemoryHead()

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        document_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        document_padding_mask: torch.Tensor | None = None,
        question_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        retrieve_top_k: int | None = None,
        usefulness_threshold: float | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        # ==================================================
        # Prepare cache and mask condition
        # ==================================================


        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        use_question_mask = not use_cache or past_key_values is None

        if (
            use_cache
            and (
                past_key_values is None
                or not isinstance(past_key_values, TSRTCache)
            )
        ):
            past_key_values = TSRTCache(config=self.config)

        # ==================================================
        # Decoder
        # ==================================================
        
        if position_ids is None:
            past_seen_tokens = past_key_values.decoder_cache.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)
        
        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values.decoder_cache if past_key_values is not None else None,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        decoder_hidden_states = inputs_embeds
        decoder_position_embeddings = self.rotary_emb(decoder_hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.decoder_layers[: self.config.num_decoder_layers]):
            decoder_hidden_states = decoder_layer(
                decoder_hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_embeddings=decoder_position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values.decoder_cache if past_key_values is not None else None,
                use_cache=use_cache,
                **kwargs,
            )
        
        # ==================================================
        # Document Encoder
        # ==================================================

        if past_key_values is not None and past_key_values.document_cache.has_encoder_state():
            encoder_hidden_states = past_key_values.document_cache.get_encoder_state()
            encoder_position_ids = torch.arange(encoder_hidden_states.shape[1], device=encoder_hidden_states.device)
            encoder_position_ids = encoder_position_ids.unsqueeze(0)

            encoder_position_embeddings = self.rotary_emb(encoder_hidden_states, encoder_position_ids)
        
        else:
            B, D, L = document_ids.shape
            document_ids = document_ids.reshape(
                -1,
                document_ids.shape[-1],
            )
            encoder_hidden_states = self.embed_tokens(document_ids)
            encoder_position_ids = torch.arange(encoder_hidden_states.shape[1], device=encoder_hidden_states.device)
            encoder_position_ids = encoder_position_ids.unsqueeze(0)

            encoder_position_embeddings = self.rotary_emb(encoder_hidden_states, encoder_position_ids)

            encoder_attention_mask = prepare_document_attention_mask(document_padding_mask)

            for i, encoder in enumerate(self.encoder_layers[: self.config.num_encoder_layers]):
                encoder_hidden_states = encoder(
                    encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    position_embeddings=encoder_position_embeddings,
                    position_ids=encoder_position_ids,
                    use_cache=False,
                    **kwargs,
                )
            
            encoder_hidden_states = encoder_hidden_states.reshape(B, D, L, -1)
            if past_key_values is not None:
                past_key_values.document_cache.update_encoder_state(
                    encoder_hidden_states
                )

        # ==================================================
        # Calculate retrieval decision
        # ==================================================

        retrieval_decision = self.retrieval_decision_head(decoder_hidden_states) 
        if question_mask is not None:
            question_mask = question_mask.unsqueeze(-1)

        if use_question_mask and question_mask is not None:
            retrieval_decision = retrieval_decision * question_mask

        # ==================================================
        # Calculate usefulness score (cosine similarity)
        # ==================================================

        projection_mask = prepare_projection_mask(document_padding_mask)
        past_embedding = (
            past_key_values.embedding_cache
            if past_key_values is not None
            else None
        )
        decoder_embs, doc_embs = self.retrieval_projection(
            decoder_hidden_state=decoder_hidden_states,
            encoder_hidden_state=encoder_hidden_states,
            past_embedding=past_embedding,
            document_attention_mask=projection_mask,
        )

        decoder_embs = F.normalize(
            decoder_embs,
            p=2,
            dim=-1,
        )

        doc_embs = F.normalize(
            doc_embs,
            p=2,
            dim=-1,
        )

        usefulness_score = torch.matmul(decoder_embs, doc_embs.transpose(-1, -2))     # (B, L, D)

        # ==================================================
        # Calculate retrieval memory
        # ==================================================

        retrieval_memory_head_output = self.retrieval_memory_head(
            usefulness_score=usefulness_score,
            retrieval_decision=retrieval_decision,
            encoder_hidden_states=encoder_hidden_states,
            document_padding_mask=document_padding_mask,
            cache=past_key_values.chosen_document_cache if past_key_values is not None else None,
            retrieve_top_k=retrieve_top_k,
            usefulness_threshold=usefulness_threshold,
        )

        retrieval_memory = retrieval_memory_head_output.retrieval_memory
        encoder_hidden_states = retrieval_memory_head_output.encoder_hidden_states
        document_padding_mask_after_retrieval = retrieval_memory_head_output.document_padding_mask

        encoder_position_ids = torch.arange(encoder_hidden_states.shape[2], device=encoder_hidden_states.device)
        encoder_position_ids = encoder_position_ids.unsqueeze(0)
        encoder_position_embeddings = self.rotary_emb(encoder_hidden_states, encoder_position_ids)

        # ==================================================
        # TSRT Decode 
        # ==================================================
        tsrt_hidden_states = decoder_hidden_states
        decoder_offset = self.config.num_decoder_layers
        cross_attention_mask = prepare_cross_attention_mask(document_padding_mask_after_retrieval) if document_padding_mask_after_retrieval is not None else None

        for i, tsrt_layer in enumerate(self.tsrt_layers[: self.config.num_tsrt_layers]):
            tsrt_hidden_states = tsrt_layer(
                decoder_hidden_states=tsrt_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                retrieval_memory=retrieval_memory,
                decoder_position_embeddings=decoder_position_embeddings,
                encoder_position_embeddings=encoder_position_embeddings,
                self_attention_mask=causal_mask_mapping[self.config.layer_types[i + decoder_offset]],
                cross_attention_mask=cross_attention_mask,
                self_attn_past_key_values=past_key_values.decoder_cache if past_key_values is not None else None,
                cross_attn_past_key_values=past_key_values.document_cache if past_key_values is not None else None,
                use_cache=use_cache,
                position_ids=position_ids,
                **kwargs,
            )
        
        tsrt_hidden_states = self.norm(tsrt_hidden_states)

        return TSRTModelOutputWithPast(
            last_hidden_state=tsrt_hidden_states,
            past_key_values=past_key_values,
            retrieval_decision=retrieval_decision.squeeze(-1),
            usefulness_score=usefulness_score,
        )


class TSRTForCausalLM(TSRTPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = TSRTModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.logged_losses = {
            "lm_loss": None,
            "retrieval_decision_loss": None,
            "retrieval_ranking_loss": None,
            "retrieval_scoring_loss": None,
            "positive_score": None,
            "negative_score": None,
            "decision_predict": None,
            "non_decision_predict": None,
        }
        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        document_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        document_padding_mask: torch.Tensor | None = None,
        question_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        labels: torch.LongTensor | None = None,
        retrieval_decision_labels: torch.LongTensor | None = None,
        usefulness_score_matrix: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        retrieve_top_k: int | None = None,
        usefulness_threshold: float | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs: TSRTModelOutputWithPast = self.model(
            input_ids=input_ids,
            document_ids=document_ids,
            attention_mask=attention_mask,
            document_padding_mask=document_padding_mask,
            question_mask=question_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            retrieve_top_k=retrieve_top_k,
            usefulness_threshold=usefulness_threshold,
            **kwargs,
        )

        num_items_in_batch = kwargs.get(
            "num_items_in_batch",
            None,
        )
        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        for key in self.logged_losses:
            self.logged_losses[key] = None

        if labels is not None:
            lm_loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
            loss = lm_loss
            self.logged_losses["lm_loss"] = lm_loss.detach()
        
        if retrieval_decision_labels is not None:
            retrieval_decision_score = outputs.retrieval_decision
            retrieval_decision_loss, decision_predict, non_decision_predict = compute_retrieval_decision_loss(
                retrieval_decision_scores=retrieval_decision_score,
                retrieval_decision_labels=retrieval_decision_labels,
                num_items_in_batch=num_items_in_batch,
            )
            loss = loss + 0.3 * retrieval_decision_loss
            self.logged_losses["retrieval_decision_loss"] = retrieval_decision_loss.detach()
            self.logged_losses["decision_predict"] = decision_predict.detach()
            self.logged_losses["non_decision_predict"] = non_decision_predict.detach()

        if usefulness_score_matrix is not None:
            usefulness_score = outputs.usefulness_score
            retrieval_ranking_loss = compute_retrieval_ranking_loss(
                usefulness_scores=usefulness_score,
                usefulness_score_matrix=usefulness_score_matrix,
            )
            self.logged_losses["retrieval_ranking_loss"] = retrieval_ranking_loss.detach()

            retrieval_scoring_loss = compute_retrieval_scoring_loss(
                usefulness_scores=usefulness_score,
                usefulness_score_matrix=usefulness_score_matrix,
                num_items_in_batch=num_items_in_batch,
            )
            self.logged_losses["retrieval_scoring_loss"] = retrieval_scoring_loss.detach()

            loss = (
                loss
                + 0.3 * retrieval_ranking_loss
                + 0.3 * retrieval_scoring_loss
            )

            positive_score, negative_score = compute_positive_negative_scores(
                usefulness_scores=usefulness_score,
                usefulness_score_matrix=usefulness_score_matrix,
            )

            self.logged_losses["positive_score"] = positive_score.detach()
            self.logged_losses["negative_score"] = negative_score.detach()

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

__all__ = [
    "TSRTModel",
    "TSRTForCausalLM",
    "TSRTPreTrainedModel",
]



