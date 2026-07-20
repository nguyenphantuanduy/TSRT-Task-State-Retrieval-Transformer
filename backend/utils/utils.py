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



if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-1.7B",
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    samples = [
        [
            "Paris is the capital of France.",
            "France is located in Europe.",
        ],
        [
            "The Moon is Earth's only natural satellite.",
        ],
        [
            "PyTorch is a deep learning framework.",
            "Transformers are neural network architectures.",
            "HotpotQA is a multi-hop question answering dataset.",
        ],
    ]

    outputs = batch_tokenize_documents(
        samples=samples,
        tokenizer=tokenizer,
        max_length=32,
    )

    input_ids = outputs["input_ids"]
    attention_mask = outputs["attention_mask"]

    print("=== Shapes ===")
    print("input_ids     :", input_ids.shape)
    print("attention_mask:", attention_mask.shape)

    B, D, L = input_ids.shape

    print("\n=== Batch Info ===")
    print(f"B = {B}")
    print(f"D = {D}")
    print(f"L = {L}")

    print("\n=== Attention Mask ===")
    print(attention_mask)

    print("\n=== Padded Document Example ===")

    # sample thứ 2 chỉ có 1 doc nên doc thứ 2 và 3 là padding doc
    print(
        attention_mask[1]
    )

    print("\n=== Decode Example ===")

    for d in range(D):
        text = tokenizer.decode(
            input_ids[0, d],
            skip_special_tokens=True,
        )

        print(f"\nDoc {d}")
        print(text)

    print("\n=== Doc Mask Derived From Attention Mask ===")

    doc_mask = attention_mask.any(dim=-1)

    print(doc_mask)
    print("shape:", doc_mask.shape)

def freeze_for_tsrt_training(model: TSRTForCausalLM):
    """
    Training strategy:

    1. Unfreeze everything.
    2. Freeze:
        - Token embedding
        - LM head
        - All decoder layers
        - FFN (MLP) of encoder layers

    Remaining trainable:
        - Encoder self-attention
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

    return model