import os

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
)

from models.tsrt.modeling_tsrt import TSRTRetriever


QWEN_MODEL = "Qwen/Qwen3-1.7B"

NUM_DECODER = 7
NUM_ENCODER = 14


# ==========================================================
# TSRT Retriever
# ==========================================================

def build_tsrt_retriever_from_qwen():

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
    # Build Retriever config
    # ==========================================

    print("Building TSRT Retriever config...")

    config = AutoConfig.from_pretrained(
        "models/tsrt",
        subfolder="retriever",
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
    # Same as Qwen decoder layers 0-13
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

    retriever = build_tsrt_retriever_from_qwen()

    # ==============================================
    # Save Retriever
    # ==============================================

    save_path = "./models/tsrt/retriever"

    os.makedirs(
        save_path,
        exist_ok=True,
    )

    retriever.save_pretrained(
        save_path,
        safe_serialization=True,
    )

    print(
        "Saved TSRT Retriever:",
        save_path,
    )