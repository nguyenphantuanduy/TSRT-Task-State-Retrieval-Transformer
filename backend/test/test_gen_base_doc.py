from __future__ import annotations

import torch
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
)

from HotpotQA_Distractor.data.load_data import (
    load_tsrt_hotpotqa_teacher,
)

# ==========================================================
# CONFIG
# ==========================================================

QWEN_MODEL = "Qwen/Qwen3-1.7B"

QUESTION_ENCODER = (
    "facebook/dpr-question_encoder-single-nq-base"
)

CONTEXT_ENCODER = (
    "facebook/dpr-ctx_encoder-single-nq-base"
)

NUM_SAMPLES = 5

TOP_K = 5

MAX_NEW_TOKENS = 256

# ==========================================================
# MAIN
# ==========================================================


def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ======================================================
    # DATASET
    # ======================================================

    dataset = load_tsrt_hotpotqa_teacher()

    validation_dataset = dataset["validation"]

    samples = [
        validation_dataset[i]
        for i in range(NUM_SAMPLES)
    ]

    # ======================================================
    # LOAD QWEN
    # ======================================================

    print("Loading Qwen...")

    tokenizer = AutoTokenizer.from_pretrained(
        QWEN_MODEL,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    model.eval()

    # ======================================================
    # LOAD DPR
    # ======================================================

    print("Loading DPR...")

    question_tokenizer = AutoTokenizer.from_pretrained(
        QUESTION_ENCODER
    )

    context_tokenizer = AutoTokenizer.from_pretrained(
        CONTEXT_ENCODER
    )

    question_encoder = AutoModel.from_pretrained(
        QUESTION_ENCODER
    ).to(device)

    context_encoder = AutoModel.from_pretrained(
        CONTEXT_ENCODER
    ).to(device)

    question_encoder.eval()
    context_encoder.eval()

    # ======================================================
    # TEST
    # ======================================================

    for sample_id, sample in enumerate(samples):

        print("=" * 100)
        print(f"Sample {sample_id}")
        print()

        question = sample["question"]

        # --------------------------------------------------
        # Build documents
        # --------------------------------------------------

        titles = []
        documents = []

        for title, sentences in zip(
            sample["context"]["title"],
            sample["context"]["sentences"],
        ):

            titles.append(title)

            documents.append(
                title
                + "\n"
                + " ".join(sentences)
            )

        # --------------------------------------------------
        # Encode question
        # --------------------------------------------------

        q_inputs = question_tokenizer(
            question,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(device)

        with torch.no_grad():

            q_embedding = (
                question_encoder(**q_inputs)
                .pooler_output
            )

        q_embedding = F.normalize(
            q_embedding,
            dim=-1,
        )

        # --------------------------------------------------
        # Encode documents
        # --------------------------------------------------

        ctx_inputs = context_tokenizer(
            documents,
            padding=True,
            truncation=True,
            max_length=384,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():

            doc_embeddings = (
                context_encoder(**ctx_inputs)
                .pooler_output
            )

        doc_embeddings = F.normalize(
            doc_embeddings,
            dim=-1,
        )

        # --------------------------------------------------
        # Retrieval
        # --------------------------------------------------

        scores = torch.matmul(
            doc_embeddings,
            q_embedding.T,
        ).squeeze(-1)

        top_scores, top_indices = torch.topk(
            scores,
            k=min(TOP_K, len(documents)),
        )

        retrieved_documents = [
            documents[idx]
            for idx in top_indices.tolist()
        ]

        # --------------------------------------------------
        # Prompt
        # --------------------------------------------------

        context = "\n\n".join(
            retrieved_documents
        )

        prompt = f"""You are given several retrieved documents.

Answer the question ONLY using the provided documents.

Documents:
{context}

Question:
{question}

Answer:
"""

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        ).to(device)

        # --------------------------------------------------
        # Generate
        # --------------------------------------------------

        with torch.no_grad():

            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = tokenizer.decode(
            outputs[0][
                inputs.input_ids.shape[1]:
            ],
            skip_special_tokens=True,
        )

        # ==================================================
        # PRINT
        # ==================================================

        print("QUESTION")
        print(question)

        print()
        print("GROUND TRUTH")
        print(sample["answer"])

        print()
        print("GENERATED")
        print(generated)

        print()
        print("SUPPORTING FACTS")
        print(
            sample["supporting_facts"]["title"]
        )

        print()
        print("DPR RETRIEVED DOCUMENTS")

        for rank, (idx, score) in enumerate(
            zip(
                top_indices.tolist(),
                top_scores.tolist(),
            ),
            start=1,
        ):

            positive = (
                titles[idx]
                in sample["supporting_facts"]["title"]
            )

            tag = (
                "[POS]"
                if positive
                else "[NEG]"
            )

            print("-" * 80)
            print(
                f"Top {rank} | Score={score:.4f}"
            )
            print(tag, titles[idx])
            print(documents[idx])

        print()
        print("ALL DOCUMENTS")

        for title, sentences in zip(
            sample["context"]["title"],
            sample["context"]["sentences"],
        ):

            positive = (
                title
                in sample["supporting_facts"]["title"]
            )

            tag = (
                "[POS]"
                if positive
                else "[NEG]"
            )

            print("-" * 80)
            print(tag, title)
            print(" ".join(sentences))

        print()


if __name__ == "__main__":
    main()