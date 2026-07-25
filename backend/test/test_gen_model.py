from __future__ import annotations

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from HotpotQA_Distractor.collator import TSRTDataCollator
from HotpotQA_Distractor.data.load_data import load_tsrt_hotpotqa_teacher


# ==========================================================
# MODEL
# ==========================================================

MODEL_PATH = "./best_model"


# ==========================================================
# MAIN
# ==========================================================

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ======================================================
    # DATASET
    # ======================================================

    dataset = load_tsrt_hotpotqa_teacher()

    validation_dataset = dataset["validation"]

    # ======================================================
    # TOKENIZER
    # ======================================================

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    # ======================================================
    # MODEL
    # ======================================================

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=device,
    )

    model.eval()

    # ======================================================
    # COLLATOR
    # ======================================================

    collator = TSRTDataCollator(
        tokenizer=tokenizer,
        document_max_length=384,
    )

    # ======================================================
    # TEST FIRST N SAMPLES
    # ======================================================

    NUM_SAMPLES = 5

    samples = [
        validation_dataset[i]
        for i in range(NUM_SAMPLES)
    ]

    batch = collator(samples)

    # ======================================================
    # BUILD QUESTION INPUT ONLY
    # ======================================================

    questions = [
        sample["question"] + "\n"
        for sample in samples
    ]

    question_inputs = tokenizer(
        questions,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = question_inputs.input_ids.to(device)
    attention_mask = question_inputs.attention_mask.to(device)

    document_ids = batch["document_ids"].to(device)

    document_padding_mask = (
        batch["document_padding_mask"]
        .to(device)
    )

    # ======================================================
    # GENERATE
    # ======================================================

    with torch.no_grad():

        outputs = model.generate(

            input_ids=input_ids,

            attention_mask=attention_mask,

            document_ids=document_ids,

            document_padding_mask=document_padding_mask,

            max_new_tokens=1024,

            do_sample=False,

            eos_token_id=tokenizer.eos_token_id,

            pad_token_id=tokenizer.pad_token_id,

            retrieve_top_k=5,

            usefulness_threshold=0.4,

            use_cache=True,
        )

    # ======================================================
    # PRINT
    # ======================================================

    for i in range(NUM_SAMPLES):

        print("=" * 100)

        print(f"Sample {i}")

        print()

        print("QUESTION")
        print(samples[i]["question"])

        print()

        print("GROUND TRUTH")
        print(samples[i]["answer"])

        print()

        print("GENERATED")

        generated = tokenizer.decode(
            outputs[i][input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        print(generated)

        print()

        print("SUPPORTING FACTS")

        print(samples[i]["supporting_facts"]["title"])

        print()

        print("DOCUMENTS")

        for title, sentences in zip(
            samples[i]["context"]["title"],
            samples[i]["context"]["sentences"],
        ):

            positive = (
                title
                in samples[i]["supporting_facts"]["title"]
            )

            tag = "[POS]" if positive else "[NEG]"

            print("-" * 80)
            print(tag, title)
            print(" ".join(sentences))

        print()


if __name__ == "__main__":
    main()