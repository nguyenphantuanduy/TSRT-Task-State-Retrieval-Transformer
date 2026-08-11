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
        document_max_length=512,
    )

    # ======================================================
    # TEST SAMPLES
    # ======================================================

    NUM_SAMPLES = 20

    # ======================================================
    # GENERATE ONE SAMPLE AT A TIME
    # ======================================================

    for i in range(NUM_SAMPLES):

        print("=" * 100)
        print(f"Sample {i + 1}/{NUM_SAMPLES}")
        print("=" * 100)

        sample = validation_dataset[i]

        # ==================================================
        # COLLATE SINGLE SAMPLE
        # ==================================================

        batch = collator([sample])

        # ==================================================
        # BUILD QUESTION INPUT ONLY
        # ==================================================

        question = sample["question"] + "\n"

        question_inputs = tokenizer(
            [question],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = question_inputs.input_ids.to(device)
        attention_mask = question_inputs.attention_mask.to(device)

        document_ids = batch["document_ids"].to(device)

        document_padding_mask = (
            batch["document_padding_mask"].to(device)
        )

        # ==================================================
        # GENERATE
        # ==================================================

        print("Generating...")

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
                use_cache=True,
            )

        # ==================================================
        # DECODE
        # ==================================================

        generated = tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        # ==================================================
        # PRINT RESULTS
        # ==================================================

        print()
        print("QUESTION")
        print(sample["question"])

        print()
        print("GROUND TRUTH")
        print(sample["answer"])

        print()
        print("GENERATED")
        print(generated)

        print()

        # ==================================================
        # CLEANUP GPU MEMORY
        # ==================================================

        del (
            batch,
            question_inputs,
            input_ids,
            attention_mask,
            document_ids,
            document_padding_mask,
            outputs,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()
