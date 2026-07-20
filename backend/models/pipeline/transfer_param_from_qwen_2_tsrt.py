import torch
from transformers import AutoConfig, AutoModelForCausalLM

from models.tsrt.modeling_tsrt import TSRTForCausalLM
from models.tsrt.configuration_tsrt import TSRTConfig


QWEN_MODEL = "Qwen/Qwen3-1.7B"


def copy_module_state(
    src,
    dst,
    prefix_src="",
    prefix_dst="",
):
    """
    Copy parameter using state_dict key mapping.
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
        trust_remote_code=True
    )


    print("Initializing TSRT")

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
    # Decoder 0-13
    # ==========================================

    print("Copy decoder layers")

    for i in range(14):

        print(
            f"decoder layer {i}"
        )

        tsrt.model.decoder_layers[i].load_state_dict(
            qwen.model.layers[i].state_dict()
        )


    # ==========================================
    # Encoder 0-13
    # Same as decoder first 14
    # ==========================================

    print("Copy encoder layers")

    for i in range(14):

        print(
            f"encoder layer {i}"
        )

        tsrt.model.encoder_layers[i].load_state_dict(
            qwen.model.layers[i].state_dict()
        )


    # ==========================================
    # TSRT layers
    #
    # Qwen layer 14-27
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


    print("Copy TSRT layers")


    for i in range(14):

        print(
            f"tsrt layer {i}"
        )

        qwen_layer = qwen.model.layers[
            i + 14
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


    print("Done")


    return tsrt



if __name__ == "__main__":

    model = build_tsrt_from_qwen()


    save_path = "./models/tsrt"


    model.save_pretrained(
        save_path,
        safe_serialization=True,
    )


    print(
        "Saved:",
        save_path
    )