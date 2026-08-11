import torch
from transformers import AutoConfig, AutoModelForCausalLM

from models.tsrt.modeling_tsrt import TSRTForCausalLM
from models.tsrt.configuration_tsrt import TSRTConfig


QWEN_MODEL = "Qwen/Qwen3-1.7B"

NUM_DECODER = 7
NUM_ENCODER = 7
NUM_TSRT = 14


# ==========================================================
# Copy Linear Attention
# ==========================================================

def copy_linear_attention(
    qwen_attn,
    tsrt_attn,
):
    """
    Copy Qwen3Attention
    -> TSRTCrossAttention

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

        if src_key not in src:
            raise KeyError(src_key)

        dst[dst_key].copy_(
            src[src_key]
        )

    tsrt_attn.load_state_dict(
        dst,
        strict=False,
    )


# ==========================================================
# Build Mini TSRT from Qwen
# ==========================================================

def build_mini_tsrt_from_qwen():

    # ==========================================
    # Load Qwen
    # ==========================================

    print("Loading Qwen...")

    qwen = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    qwen.eval()

    # ==========================================
    # Build Mini TSRT config
    # ==========================================

    print("Building Mini TSRT config...")

    config = AutoConfig.from_pretrained(
        "models/tsrt",
        trust_remote_code=True,
    )

    # ==========================================
    # Override Mini architecture
    # ==========================================

    config.num_decoder_layers = NUM_DECODER
    config.num_encoder_layers = NUM_ENCODER
    config.num_tsrt_layers = NUM_TSRT

    config.num_hidden_layers = (
        NUM_DECODER
        + NUM_ENCODER
        + NUM_TSRT
    )

    print(
        f"Mini TSRT architecture: "
        f"{NUM_DECODER} decoder + "
        f"{NUM_ENCODER} encoder + "
        f"{NUM_TSRT} TSRT = "
        f"{config.num_hidden_layers} layers"
    )

    # ==========================================
    # Initialize Mini TSRT
    # ==========================================

    print("Initializing Mini TSRT...")

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
    #
    # Qwen layer 0-6
    # ==========================================

    print("Copy Mini decoder layers")

    for i in range(NUM_DECODER):

        print(
            f"decoder layer {i} "
            f"<- Qwen layer {i}"
        )

        tsrt.model.decoder_layers[i].load_state_dict(
            qwen.model.layers[i].state_dict()
        )

    # ==========================================
    # Encoder 0-6
    #
    # Qwen layer 0-6
    # ==========================================

    print("Copy Mini encoder layers")

    for i in range(NUM_ENCODER):

        print(
            f"encoder layer {i} "
            f"<- Qwen layer {i}"
        )

        tsrt.model.encoder_layers[i].load_state_dict(
            qwen.model.layers[i].state_dict()
        )

    # ==========================================
    # TSRT layers
    #
    # Qwen layer 14-27
    #
    # 14 LAST QWEN LAYERS
    #
    # TSRT:
    #
    # input_layernorm
    #      <- qwen.input_layernorm
    #
    # post_self_attention_layernorm
    # post_cross_attention_layernorm
    #      <- qwen.post_attention_layernorm
    #
    # self_attn
    # cross_attn
    #      <- qwen.self_attn
    #
    # mlp
    #      <- qwen.mlp
    #
    # ==========================================

    print("Copy Mini TSRT layers")

    QWEN_TSRT_START = 14

    for i in range(NUM_TSRT):

        qwen_index = (
            QWEN_TSRT_START + i
        )

        print(
            f"TSRT layer {i} "
            f"<- Qwen layer {qwen_index}"
        )

        qwen_layer = qwen.model.layers[
            qwen_index
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

    # ==========================================
    # Done
    # ==========================================

    print(
        "Mini TSRT transfer done."
    )

    return tsrt


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    model = build_mini_tsrt_from_qwen()

    # ==============================================
    # Save Mini TSRT
    # ==============================================

    save_path = "./models/tsrt/mini"

    model.save_pretrained(
        save_path,
        safe_serialization=True,
    )

    print(
        "Saved Mini TSRT:",
        save_path,
    )
