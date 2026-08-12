"""TSRT model configuration"""

from huggingface_hub.dataclasses import strict

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


@strict
class TSRTConfig(Qwen3Config):

    model_type = "tsrt"

    keys_to_ignore_at_inference = ["past_key_values"]

    # ==========================================================
    # Tensor Parallel
    # ==========================================================

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    # ==========================================================
    # Pipeline Parallel
    # ==========================================================

    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    # ==========================================================
    # TSRT
    # ==========================================================

    num_decoder_layers: int = 14
    num_encoder_layers: int = 14
    num_tsrt_layers: int = 14
    retrieval_bias_gamma: float = 4.0
    retrieval_embedding_size: int = 1024


__all__ = ["TSRTConfig"]