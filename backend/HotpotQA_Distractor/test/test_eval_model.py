from __future__ import annotations

import torch

from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
)

from models.tsrt.trainer import TSRTTrainer
from ..retriever_collator import TSRTRetrieverCollator
from ..data.load_data import load_tsrt_hotpotqa_teacher
from utils.utils import freeze_for_tsrt_retriever_training


MODEL_NAME = "tsrt-lab/TSRT-Qwen3-1.7B"


def test_eval():

    # =====================================================
    # DEVICE
    # =====================================================

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    # =====================================================
    # DATASET
    # =====================================================

    print("Loading dataset...")

    dataset = load_tsrt_hotpotqa_teacher()

    validation_dataset = (
        dataset["validation"]
        .select(range(2000))
    )

    print(f"Validation samples: {len(validation_dataset)}")

    # =====================================================
    # TOKENIZER
    # =====================================================

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # =====================================================
    # MODEL
    # =====================================================

    print("Loading retriever...")

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        subfolder="retriever",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=device,
    )

    # =====================================================
    # FREEZE
    # =====================================================

    print("Freezing retriever...")

    model = freeze_for_tsrt_retriever_training(model)

    # =====================================================
    # COLLATOR
    # =====================================================

    collator = TSRTRetrieverCollator(
        tokenizer=tokenizer,
        document_max_length=512,
    )

    # =====================================================
    # TRAINING ARGS
    # =====================================================

    training_args = TrainingArguments(
        output_dir="./eval_test",

        per_device_eval_batch_size=1,

        bf16=True,

        report_to="none",

        dataloader_num_workers=4,

        remove_unused_columns=False,

        # Important:
        # simulate the same metric configuration
        # used during training.
        metric_for_best_model="loss",
        greater_is_better=False,
    )

    # =====================================================
    # TRAINER
    # =====================================================

    trainer = TSRTTrainer(
        model=model,
        args=training_args,

        eval_dataset=validation_dataset,

        processing_class=tokenizer,

        data_collator=collator,
    )

    # =====================================================
    # EVALUATE
    # =====================================================

    print()
    print("=" * 60)
    print("RUNNING EVALUATION")
    print("=" * 60)

    metrics = trainer.evaluate()

    # =====================================================
    # PRINT METRICS
    # =====================================================

    print()
    print("=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)

    for key, value in metrics.items():
        print(f"{key}: {value}")

    # =====================================================
    # CHECK METRIC NAMES
    # =====================================================

    print()
    print("=" * 60)
    print("METRIC CHECK")
    print("=" * 60)

    print("Available keys:")

    for key in metrics.keys():
        print(f"  - {key}")

    print()

    if "eval_retrieval_ranking_loss" in metrics:
        print(
            "SUCCESS: eval_retrieval_ranking_loss exists."
        )
    else:
        print(
            "WARNING: eval_retrieval_ranking_loss NOT found."
        )

    if "retrieval_ranking_loss" in metrics:
        print(
            "WARNING: retrieval_ranking_loss exists "
            "without eval_ prefix."
        )

    # =====================================================
    # CHECK LOSS
    # =====================================================

    print()
    print("=" * 60)
    print("LOSS CHECK")
    print("=" * 60)

    if "eval_loss" in metrics:
        print(f"eval_loss: {metrics['eval_loss']}")
    else:
        print("eval_loss: NOT FOUND")

    if "eval_retrieval_ranking_loss" in metrics:
        print(
            f"eval_retrieval_ranking_loss: "
            f"{metrics['eval_retrieval_ranking_loss']}"
        )

    # =====================================================
    # RESULT
    # =====================================================

    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)

    if "eval_retrieval_ranking_loss" in metrics:
        print(
            "Evaluation metric naming looks correct."
        )
        print(
            "The Trainer should be able to use "
            "'retrieval_ranking_loss' as metric_for_best_model."
        )
    else:
        print(
            "Evaluation metric naming is still incorrect."
        )
        print(
            "TSRTTrainer.evaluate() is returning the "
            "ranking loss without the eval_ prefix."
        )


if __name__ == "__main__":
    test_eval()