from __future__ import annotations

import torch
from transformers import AutoModel, AutoModelForCausalLM


# ==========================================================
# CONFIGURATION
# ==========================================================

MODEL_NAME = "tsrt-lab/TSRT-Qwen3-1.7B"

OUTPUT_DIR = "./mini_best_model"


# ==========================================================
# TRANSFER MODULE
# ==========================================================

def transfer_module(
    source_module: torch.nn.Module,
    target_module: torch.nn.Module,
    module_name: str,
) -> None:
    """
    Transfer all parameters/buffers from source_module
    to target_module with strict state-dict validation.
    """

    source_state = source_module.state_dict()
    target_state = target_module.state_dict()

    source_keys = set(source_state.keys())
    target_keys = set(target_state.keys())

    missing_in_source = target_keys - source_keys
    extra_in_source = source_keys - target_keys

    if missing_in_source or extra_in_source:
        raise RuntimeError(
            f"State-dict mismatch for {module_name}.\n"
            f"Missing in source: "
            f"{sorted(missing_in_source)}\n"
            f"Extra in source: "
            f"{sorted(extra_in_source)}"
        )

    for key in source_state:

        if source_state[key].shape != target_state[key].shape:
            raise RuntimeError(
                f"Shape mismatch for "
                f"{module_name}.{key}: "
                f"source={tuple(source_state[key].shape)}, "
                f"target={tuple(target_state[key].shape)}"
            )

    target_module.load_state_dict(
        source_state,
        strict=True,
    )

    print(
        f"[OK] Transferred: {module_name}"
    )


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # =====================================================
    # LOAD MINI TSRTForCausalLM
    # =====================================================

    print(
        "Loading Mini TSRTForCausalLM..."
    )

    causal_lm = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        subfolder="mini",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=device,
    )

    # =====================================================
    # LOAD MINI TSRTRetriever
    # =====================================================

    print(
        "Loading Mini TSRTRetriever..."
    )

    retriever = AutoModel.from_pretrained(
        MODEL_NAME,
        subfolder="mini/retriever",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=device,
    )

    # =====================================================
    # MODULE MAPPING
    #
    # Mini TSRTRetriever:
    #
    #   decoder_layers  -> 7 layers
    #   encoder_layers  -> 7 layers
    #   rotary_emb
    #   retrieval_projection
    #
    # Mini TSRTForCausalLM:
    #
    #   model.decoder_layers
    #   model.encoder_layers
    #   model.rotary_emb
    #   model.retrieval_projection
    #
    # =====================================================

    print(
        "\nTransferring Mini Retriever weights..."
    )

    # =====================================================
    # Decoder
    # =====================================================

    transfer_module(
        source_module=retriever.decoder_layers,
        target_module=causal_lm.model.decoder_layers,
        module_name="mini.decoder_layers",
    )

    # =====================================================
    # Encoder
    # =====================================================

    transfer_module(
        source_module=retriever.encoder_layers,
        target_module=causal_lm.model.encoder_layers,
        module_name="mini.encoder_layers",
    )

    # =====================================================
    # Rotary Embedding
    # =====================================================

    transfer_module(
        source_module=retriever.rotary_emb,
        target_module=causal_lm.model.rotary_emb,
        module_name="mini.rotary_emb",
    )

    # =====================================================
    # Retrieval Projection
    # =====================================================

    transfer_module(
        source_module=retriever.retrieval_projection,
        target_module=causal_lm.model.retrieval_projection,
        module_name="mini.retrieval_projection",
    )

    # =====================================================
    # SAVE
    # =====================================================

    print(
        f"\nSaving transferred Mini model to: "
        f"{OUTPUT_DIR}"
    )

    causal_lm.save_pretrained(
        OUTPUT_DIR,
        safe_serialization=True,
    )

    print(
        f"Done. Mini model saved to: "
        f"{OUTPUT_DIR}"
    )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()
