import torch
from models.tsrt.modeling_tsrt import TSRTForCausalLM, TSRTRetriever


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
    """ Training strategy: 
    1. Unfreeze everything. 
    2. Freeze: 
        - Token embedding 
        - LM head 
        - All encoder layers 
        - All decoder layers
    3. Unfreeze: 
        - Last 2 decoder layers 
        - Last 4 encoder layers 
    Remaining trainable: 
        - Last 2 decoder layers 
        - Last 4 encoder layers 
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
    # Freeze entire decoder
    # ==========================================================

    for layer in model.model.decoder_layers:
        for param in layer.parameters():
            param.requires_grad = False

    # ==========================================================
    # Freeze entire encoder
    # ==========================================================

    for layer in model.model.encoder_layers:
        for param in layer.parameters():
            param.requires_grad = False

    # ==========================================================
    # Unfreeze last 2 decoder layers
    # ==========================================================

    for layer in model.model.decoder_layers[-2:]:
        for param in layer.parameters():
            param.requires_grad = True

    # ==========================================================
    # Unfreeze last 4 encoder layers
    # ==========================================================

    for layer in model.model.encoder_layers[-4:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Print statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Trainable params: {trainable_params:,}")
    print(f"Total params:     {total_params:,}")
    print(f"Trainable ratio:  {100 * trainable_params / total_params:.2f}%")

    return model

def freeze_for_tsrt_retriever_training(
    model: TSRTRetriever,
):
    """
    Training strategy for TSRT retriever:

    1. Unfreeze everything.
    2. Freeze:
        - Token embedding
        - First 3 decoder layers
        - First 7 encoder layers

    Remaining trainable:
        - Last 4 decoder layers
        - Last 7 encoder layers
        - Retrieval projection
        - Other retriever-specific parameters
    """

    # ==========================================================
    # Unfreeze everything
    # ==========================================================

    for param in model.parameters():
        param.requires_grad = True

    # ==========================================================
    # Freeze embedding
    # ==========================================================

    for param in model.embed_tokens.parameters():
        param.requires_grad = False

    # ==========================================================
    # Freeze first 3 decoder layers
    # ==========================================================

    for layer in model.decoder_layers[:3]:
        for param in layer.parameters():
            param.requires_grad = False

    # ==========================================================
    # Freeze first 8 encoder layers
    # ==========================================================

    for layer in model.encoder_layers[:7]:
        for param in layer.parameters():
            param.requires_grad = False

    # ==========================================================
    # Print statistics
    # ==========================================================

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable params: {trainable_params:,}"
    )

    print(
        f"Total params:     {total_params:,}"
    )

    print(
        f"Trainable ratio:  "
        f"{100 * trainable_params / total_params:.2f}%"
    )

    return model
