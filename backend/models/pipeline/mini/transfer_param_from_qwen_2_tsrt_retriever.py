import os

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
)

from models.tsrt.modeling_tsrt import TSRTRetriever


QWEN_MODEL = "Qwen/Qwen3-1.7B"

# ==========================================================
# Mini TSRT Retriever
# ==========================================================

NUM_DECODER = 7
NUM_ENCODER = 7


def build_mini_tsrt_retriever_from_qwen():

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
    # Build Mini Retriever config
    # ==========================================

    print("Building Mini TSRT Retriever config...")

    config = AutoConfig.from_pretrained(
        "models/tsrt",
        subfolder="mini/retriever",
        trust_remote_code=True,
    )

    # ==========================================
    # Override Mini architecture
    # ==========================================

    config.num_decoder_layers = NUM_DECODER
    config.num_encoder_layers = NUM_ENCODER
    config.num_tsrt_layers = 14
    config.num_hidden_layers = (
        NUM_DECODER
        + NUM_ENCODER
        + config.num_tsrt_layers
    )

    print(
        f"Mini Retriever architecture: "
        f"{NUM_DECODER} decoder + "
        f"{NUM_ENCODER} encoder + "
        f"{config.num_tsrt_layers} TSRT = "
        f"{config.num_hidden_layers} layers"
    )

    # ==========================================
    # Initialize Retriever
    # ==========================================

    print("Initializing Mini TSRT Retriever...")

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
    #
    # Copy Qwen decoder layers 0-6
    # ==========================================

    print("Copy Mini retriever decoder layers...")

    for i in range(NUM_DECODER):

        print(
            f"retriever decoder layer {i}"
        )

        retriever.decoder_layers[i].load_state_dict(
            qwen.model.layers[i].state_dict()
        )

    # ==========================================
    # Encoder 0-6
    #
    # Copy Qwen decoder layers 0-6
    # ==========================================

    print("Copy Mini retriever encoder layers...")

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
    # Keep initialized weights from
    # TSRTRetriever.
    # ==========================================

    print(
        "Keep Mini retriever retrieval_projection "
        "with initialized weights."
    )

    print("Mini TSRT Retriever transfer done.")

    return retriever


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    retriever = build_mini_tsrt_retriever_from_qwen()

    # ==============================================
    # Save Mini Retriever
    # ==============================================

    save_path = "./models/tsrt/mini_retriever"

    os.makedirs(
        save_path,
        exist_ok=True,
    )

    retriever.save_pretrained(
        save_path,
        safe_serialization=True,
    )

    print(
        "Saved Mini TSRT Retriever:",
        save_path,
    )
