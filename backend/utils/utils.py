import torch
from models.tsrt.modeling_tsrt import TSRTForCausalLM


def batch_tokenize_documents(
    samples: list[list[str]],
    tokenizer,
    max_length: int,
):
    """
    Args:
        samples:
            [
                [doc1, doc2],
                [doc3],
                [doc4, doc5, doc6]
            ]

    Returns:
        {
            "input_ids": (B, D, L),
            "attention_mask": (B, D, L),
        }
    """

    batch_size = len(samples)

    # =====================
    # PAD NUM DOCS
    # =====================

    max_docs = max(len(sample) for sample in samples)

    padded_docs = []

    for sample in samples:
        padded_docs.append(
            sample + [""] * (max_docs - len(sample))
        )

    # =====================
    # FLATTEN (B,D)->(BD)
    # =====================

    flat_docs = [
        doc
        for sample in padded_docs
        for doc in sample
    ]

    # =====================
    # TOKENIZE
    # =====================

    encoded = tokenizer(
        flat_docs,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    seq_len = input_ids.shape[-1]

    # =====================
    # RESTORE (BD,L)->(B,D,L)
    # =====================

    input_ids = input_ids.view(
        batch_size,
        max_docs,
        seq_len,
    )

    attention_mask = attention_mask.view(
        batch_size,
        max_docs,
        seq_len,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

def freeze_for_tsrt_training(model: TSRTForCausalLM):
    """
    Training strategy:

    1. Unfreeze everything.
    2. Freeze:
        - Token embedding
        - LM head
        - All decoder layers
        - FFN (MLP) of encoder layers
    3. Unfreeze last 2 FFN of encoder and decoder.

    Remaining trainable:
        - Encoder self-attention
        - Last 2 encoder FFN
        - Last 2 decoder FFN
        - Entire TSRT layers
        - Retrieval heads
        - Final norm
    """

    # ==========================================================
    # Unfreeze everything
    # ==========================================================

    for param in model.parameters():
        param.requires_grad = True

    # ==========================================================
    # Freeze embedding
    # ==========================================================

    for param in model.model.embed_tokens.parameters():
        param.requires_grad = False

    # ==========================================================
    # Freeze LM head
    # ==========================================================

    for param in model.lm_head.parameters():
        param.requires_grad = False

    # ==========================================================
    # Freeze decoder
    # ==========================================================

    for layer in model.model.decoder_layers:
        for param in layer.parameters():
            param.requires_grad = False

    # ==========================================================
    # Freeze encoder FFN only
    # ==========================================================

    for layer in model.model.encoder_layers:
        for param in layer.mlp.parameters():
            param.requires_grad = False

    # ==========================================================
    # Unfreeze last 4 decoder FFN
    # ==========================================================

    for layer in model.model.decoder_layers[-4:]:
        for param in layer.mlp.parameters():
            param.requires_grad = True

    # ==========================================================
    # Unfreeze last 4 encoder FFN
    # ==========================================================

    for layer in model.model.encoder_layers[-4:]:
        for param in layer.mlp.parameters():
            param.requires_grad = True

    return model
