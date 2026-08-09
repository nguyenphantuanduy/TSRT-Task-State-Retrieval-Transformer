import os

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModel,
)

from models.tsrt.modeling_tsrt import (
    TSRTForCausalLM,
    TSRTRetriever,
)

from models.tsrt.configuration_tsrt import (
    TSRTConfig,
)


QWEN_MODEL = "Qwen/Qwen3-1.7B"

NUM_DECODER = 7
NUM_ENCODER = 14
NUM_TSRT = 21


# ==========================================================
# Helper
# ==========================================================

def copy_module_state(
    src,
    dst,
    prefix_src="",
    prefix_dst="",
):
    """
    Copy parameters using state_dict key mapping.
    """

    src_state = src.state_dict()
    dst_state = dst.state_dict()

    new_state = {}

    for k in dst_state.keys():

        src_key = k

        if prefix_dst and k.startswith(prefix_dst):
            src_key = prefix_src + k[len(prefix_dst):]

        if src_key not in src_state:
            raise KeyError(
                f"Missing key\n"
                f"dst: {k}\n"
                f"src: {src_key}"
            )

        new_state[k] = src_state[src_key]

    dst.load_state_dict(
        new_state,
        strict=True,
    )


def copy_linear_attention(
    qwen_attn,
    tsrt_attn,
):
    """
    Copy Qwen3Attention
    -> TSRTCrossAttention

    Copies:

        q_proj
        k_proj
        v_proj
        o_proj
        q_norm
        k_norm
    """

    mapping = {
        "q_proj.weight": "q_proj.weight",
        "k_proj.weight": "k_proj.weight",
        "v_proj.weight": "v_proj.weight",
        "o_proj.weight": "o_proj.weight",

        "q_norm.weight": "q_norm.weight",
        "k_norm.weight": "k_norm.weight",
    }

    src = qwen_attn.state_dict()
    dst = tsrt_attn.state_dict()

    for dst_key, src_key in mapping.items():

        if dst_key not in dst:
            raise KeyError(dst_key)

        dst[dst_key].copy_(
            src[src_key]
        )

    tsrt_attn.load_state_dict(
        dst,
        strict=False,
    )


# ==========================================================
# TSRT Causal LM
# ==========================================================

def build_tsrt_from_qwen():

    print("Loading Qwen...")

    qwen = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    qwen.eval()

    print("Building TSRT config...")

    config = AutoConfig.from_pretrained(
        "models/tsrt",
        trust_remote_code=True,
    )

    print("Initializing TSRT...")

    tsrt = TSRTForCausalLM(
        config
    )

    # ==========================================
    # Embedding
    # ==========================================

    print("Copy embedding")

    tsrt.model.embed_tokens.load_state_dict(
        qwen.model.embed_tokens.state_dict()
    )

    # ==========================================
    # Norm + RoPE
    # ==========================================

    print("Copy norm")

    tsrt.model.norm.load_state_dict(
        qwen.model.norm.state_dict()
    )

    print("Copy rotary")

    tsrt.model.rotary_emb.load_state_dict(
        qwen.model.rotary_emb.state_dict()
    )

    # ==========================================
    # LM head
    # ==========================================

    print("Copy lm_head")

    tsrt.lm_head.load_state_dict(
        qwen.lm_head.state_dict()
    )

    # ==========================================
    # Decoder 0-6
    # ==========================================

    print("Copy decoder layers")

    for i in range(NUM_DECODER):

        print(
            f"decoder layer {i}"
        )

        tsrt.model.decoder_layers[i].load_state_dict(
            qwen.model.layers[i].state_dict()
        )

    # ==========================================
    # Encoder 0-13
    #
    # Same as decoder first 14
    # ==========================================

    print("Copy encoder layers")

    for i in range(NUM_ENCODER):

        print(
            f"encoder layer {i}"
        )

        tsrt.model.encoder_layers[i].load_state_dict(
            qwen.model.layers[i].state_dict()
        )

    # ==========================================
    # TSRT layers
    #
    # Qwen layer 7-27
    # ==========================================

    print("Copy TSRT layers")

    for i in range(NUM_TSRT):

        print(
            f"tsrt layer {i}"
        )

        qwen_layer = qwen.model.layers[
            i + NUM_DECODER
        ]

        tsrt_layer = tsrt.model.tsrt_layers[
            i
        ]

        # ----------------------------
        # Norm
        # ----------------------------

        tsrt_layer.input_layernorm.load_state_dict(
            qwen_layer.input_layernorm.state_dict()
        )

        tsrt_layer.post_self_attention_layernorm.load_state_dict(
            qwen_layer.post_attention_layernorm.state_dict()
        )

        tsrt_layer.post_cross_attention_layernorm.load_state_dict(
            qwen_layer.post_attention_layernorm.state_dict()
        )

        # ----------------------------
        # Self attention
        # ----------------------------

        tsrt_layer.self_attn.load_state_dict(
            qwen_layer.self_attn.state_dict()
        )

        # ----------------------------
        # Cross attention
        # ----------------------------

        copy_linear_attention(
            qwen_layer.self_attn,
            tsrt_layer.cross_attn,
        )

        # ----------------------------
        # MLP
        # ----------------------------

        tsrt_layer.mlp.load_state_dict(
            qwen_layer.mlp.state_dict()
        )

    print("TSRT Causal LM transfer done.")

    return tsrt


# ==========================================================
# TSRT Retriever
# ==========================================================

def build_tsrt_retriever_from_qwen():

    print("Loading Qwen...")

    qwen = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    qwen.eval()

    # ==========================================
    # Build Retriever config
    # ==========================================

    print("Building TSRT Retriever config...")

    config = AutoConfig.from_pretrained(
        "models/tsrt/retriever",
        trust_remote_code=True,
    )

    # ==========================================
    # Initialize Retriever
    # ==========================================

    print("Initializing TSRT Retriever...")

    retriever = TSRTRetriever(
        config
    )

    # ==========================================
    # Embedding
    # ==========================================

    print("Copy retriever embedding...")

    retriever.embed_tokens.load_state_dict(
        qwen.model.embed_tokens.state_dict()
    )

    # ==========================================
    # Rotary embedding
    # ==========================================

    print("Copy retriever rotary...")

    retriever.rotary_emb.load_state_dict(
        qwen.model.rotary_emb.state_dict()
    )

    # ==========================================
    # Decoder 0-6
    # ==========================================

    print("Copy retriever decoder layers...")

    for i in range(NUM_DECODER):

        print(
            f"retriever decoder layer {i}"
        )

        retriever.decoder_layers[i].load_state_dict(
            qwen.model.layers[i].state_dict()
        )

    # ==========================================
    # Encoder 0-13
    #
    # Same as decoder first 14
    # ==========================================

    print("Copy retriever encoder layers...")

    for i in range(NUM_ENCODER):

        print(
            f"retriever encoder layer {i}"
        )

        retriever.encoder_layers[i].load_state_dict(
            qwen.model.layers[i].state_dict()
        )

    # ==========================================
    # Retrieval projection
    #
    # No direct Qwen counterpart.
    #
    # Keep the initialized weights from
    # TSRTRetrieverRetrievalProjection.
    # ==========================================

    print(
        "Keep retriever retrieval_projection "
        "with initialized weights."
    )

    print("TSRT Retriever transfer done.")

    return retriever


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    # ======================================================
    # Build TSRT Causal LM
    # ======================================================

    model = build_tsrt_from_qwen()

    save_path = "./models/tsrt"

    model.save_pretrained(
        save_path,
        safe_serialization=True,
    )

    print(
        "Saved TSRT Causal LM:",
        save_path,
    )

    # ======================================================
    # Build TSRT Retriever
    # ======================================================

    retriever = build_tsrt_retriever_from_qwen()

    retriever_save_path = os.path.join(
        save_path,
        "retriever",
    )

    os.makedirs(
        retriever_save_path,
        exist_ok=True,
    )

    retriever.save_pretrained(
        retriever_save_path,
        safe_serialization=True,
    )

    print(
        "Saved TSRT Retriever:",
        retriever_save_path,
    )
