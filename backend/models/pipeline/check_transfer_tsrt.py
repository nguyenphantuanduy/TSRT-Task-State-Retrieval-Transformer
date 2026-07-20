import torch

from transformers import AutoModelForCausalLM, AutoConfig

from models.tsrt.modeling_tsrt import TSRTForCausalLM


QWEN_MODEL = "Qwen/Qwen3-1.7B"
TSRT_PATH = "./models/tsrt"


def compare_state_dict(
    name,
    src,
    dst,
    atol=1e-6,
):
    """
    Compare two modules.
    """

    src_state = src.state_dict()
    dst_state = dst.state_dict()

    if src_state.keys() != dst_state.keys():
        print(f"[FAIL] {name}: key mismatch")
        print("src only:", src_state.keys() - dst_state.keys())
        print("dst only:", dst_state.keys() - src_state.keys())
        return False


    passed = True

    for k in src_state.keys():

        a = src_state[k].float()
        b = dst_state[k].float()

        max_err = (a - b).abs().max().item()
        mean_err = (a - b).abs().mean().item()

        ok = torch.allclose(
            a,
            b,
            atol=atol,
            rtol=0,
        )

        if not ok:
            passed = False
            print(
                f"[FAIL] {name}.{k}"
                f"\n  max_err={max_err}"
                f"\n  mean_err={mean_err}"
            )

    if passed:
        print(
            f"[OK] {name}"
        )

    return passed



def compare_linear_attention(
    name,
    qwen_attn,
    tsrt_attn,
):

    mapping = {

        "q_proj.weight":
            "q_proj.weight",

        "k_proj.weight":
            "k_proj.weight",

        "v_proj.weight":
            "v_proj.weight",

        "o_proj.weight":
            "o_proj.weight",

        "q_norm.weight":
            "q_norm.weight",

        "k_norm.weight":
            "k_norm.weight",
    }


    passed = True

    src = qwen_attn.state_dict()
    dst = tsrt_attn.state_dict()


    for dst_key, src_key in mapping.items():

        a = src[src_key].float()
        b = dst[dst_key].float()


        max_err = (
            a - b
        ).abs().max().item()


        ok = torch.allclose(
            a,
            b,
            atol=1e-6,
            rtol=0,
        )


        if not ok:

            passed = False

            print(
                f"[FAIL] {name}.{dst_key}"
                f"\n max_err={max_err}"
            )


    if passed:
        print(
            f"[OK] {name}"
        )


    return passed



def check_transfer():


    print("=" * 60)
    print("Loading Qwen")
    print("=" * 60)


    qwen = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    qwen.eval()



    print("=" * 60)
    print("Loading TSRT")
    print("=" * 60)


    tsrt = TSRTForCausalLM.from_pretrained(
        TSRT_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    tsrt.eval()



    results = []



    # ======================================
    # Embedding
    # ======================================

    results.append(
        compare_state_dict(
            "embedding",
            qwen.model.embed_tokens,
            tsrt.model.embed_tokens,
        )
    )


    # ======================================
    # Norm
    # ======================================

    results.append(
        compare_state_dict(
            "norm",
            qwen.model.norm,
            tsrt.model.norm,
        )
    )


    # ======================================
    # RoPE
    # ======================================

    results.append(
        compare_state_dict(
            "rotary_emb",
            qwen.model.rotary_emb,
            tsrt.model.rotary_emb,
        )
    )


    # ======================================
    # LM Head
    # ======================================

    results.append(
        compare_state_dict(
            "lm_head",
            qwen.lm_head,
            tsrt.lm_head,
        )
    )



    # ======================================
    # Decoder
    # ======================================

    print("\nDecoder")

    for i in range(14):

        results.append(
            compare_state_dict(
                f"decoder_layer_{i}",
                qwen.model.layers[i],
                tsrt.model.decoder_layers[i],
            )
        )



    # ======================================
    # Encoder
    # ======================================

    print("\nEncoder")

    for i in range(14):

        results.append(
            compare_state_dict(
                f"encoder_layer_{i}",
                qwen.model.layers[i],
                tsrt.model.encoder_layers[i],
            )
        )



    # ======================================
    # TSRT layers
    # ======================================

    print("\nTSRT Layers")


    for i in range(14):

        print(
            f"\nTSRT layer {i}"
        )


        qwen_layer = qwen.model.layers[i+14]

        tsrt_layer = tsrt.model.tsrt_layers[i]


        results.append(
            compare_state_dict(
                f"tsrt_{i}_input_norm",
                qwen_layer.input_layernorm,
                tsrt_layer.input_layernorm,
            )
        )


        results.append(
            compare_state_dict(
                f"tsrt_{i}_post_self_norm",
                qwen_layer.post_attention_layernorm,
                tsrt_layer.post_self_attention_layernorm,
            )
        )


        results.append(
            compare_state_dict(
                f"tsrt_{i}_post_cross_norm",
                qwen_layer.post_attention_layernorm,
                tsrt_layer.post_cross_attention_layernorm,
            )
        )


        results.append(
            compare_state_dict(
                f"tsrt_{i}_self_attn",
                qwen_layer.self_attn,
                tsrt_layer.self_attn,
            )
        )


        results.append(
            compare_linear_attention(
                f"tsrt_{i}_cross_attn",
                qwen_layer.self_attn,
                tsrt_layer.cross_attn,
            )
        )


        results.append(
            compare_state_dict(
                f"tsrt_{i}_mlp",
                qwen_layer.mlp,
                tsrt_layer.mlp,
            )
        )



    print("\n" + "=" * 60)

    if all(results):

        print(
            "ALL WEIGHTS TRANSFERRED CORRECTLY"
        )

    else:

        print(
            "TRANSFER ERROR DETECTED"
        )

    print("=" * 60)



if __name__ == "__main__":

    check_transfer()