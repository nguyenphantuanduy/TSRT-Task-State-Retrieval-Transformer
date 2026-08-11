from __future__ import annotations

import json

import torch

from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    EarlyStoppingCallback,
)

from models.tsrt.trainer import TSRTTrainer
from ...retriever_collator import TSRTRetrieverCollator
from ...data.load_data import load_tsrt_hotpotqa_teacher
from utils.utils import (
    freeze_for_mini_tsrt_retriever_training,
)


# ==========================================================
# MODEL
# ==========================================================

MODEL_NAME = "tsrt-lab/TSRT-Qwen3-1.7B"

MODEL_SUBFOLDER = "mini/retriever"

CHECKPOINT_DIR = "./mini_retriever_checkpoint"

BEST_MODEL_DIR = "./mini_best_retriever"


# ==========================================================
# TRAIN
# ==========================================================

def train():

    # =====================================================
    # DEVICE
    # =====================================================

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # =====================================================
    # DATASET
    # =====================================================

    print("Loading dataset...")

    dataset = load_tsrt_hotpotqa_teacher()

    train_dataset = (
        dataset["train"]
        .shuffle(seed=42)
    )

    validation_dataset = (
        dataset["validation"]
        .select(range(2000))
    )

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

    print("Loading Mini TSRT Retriever...")

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        subfolder=MODEL_SUBFOLDER,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=device,
    )

    # =====================================================
    # FREEZE
    # =====================================================

    print(
        "Applying Mini Retriever "
        "training strategy..."
    )

    model = (
        freeze_for_mini_tsrt_retriever_training(
            model
        )
    )

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
        output_dir=CHECKPOINT_DIR,

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,

        gradient_accumulation_steps=32,

        save_total_limit=2,

        learning_rate=2e-5,
        weight_decay=0.01,

        num_train_epochs=1,

        bf16=True,

        # -------------------------------------------------
        # LOGGING
        # -------------------------------------------------

        logging_strategy="steps",
        logging_steps=10,

        # -------------------------------------------------
        # EVALUATION
        # -------------------------------------------------

        eval_strategy="steps",
        eval_steps=250,

        # -------------------------------------------------
        # CHECKPOINT
        # -------------------------------------------------

        save_strategy="steps",
        save_steps=250,

        load_best_model_at_end=True,

        metric_for_best_model="loss",
        greater_is_better=False,

        # -------------------------------------------------
        # OTHER
        # -------------------------------------------------

        report_to="none",

        dataloader_num_workers=4,

        remove_unused_columns=False,
    )

    # =====================================================
    # TRAINER
    # =====================================================

    trainer = TSRTTrainer(
        model=model,
        args=training_args,

        train_dataset=train_dataset,
        eval_dataset=validation_dataset,

        processing_class=tokenizer,

        data_collator=collator,

        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.0,
            ),
        ],
    )

    # =====================================================
    # TRAIN
    # =====================================================

    trainer.train()

    # =====================================================
    # SAVE LOG HISTORY
    # =====================================================

    print(
        "Saving Mini Retriever "
        "training logs..."
    )

    log_path = (
        f"{CHECKPOINT_DIR}/training_log.txt"
    )

    with open(
        log_path,
        "w",
        encoding="utf-8",
    ) as f:

        for log in trainer.state.log_history:

            f.write(
                json.dumps(
                    log,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )

    print(
        f"Training logs saved to: "
        f"{log_path}"
    )

    # =====================================================
    # SAVE BEST MODEL
    # =====================================================

    print(
        "Saving best Mini Retriever..."
    )

    trainer.save_model(
        BEST_MODEL_DIR
    )

    tokenizer.save_pretrained(
        BEST_MODEL_DIR
    )

    print(
        f"Mini Retriever saved to: "
        f"{BEST_MODEL_DIR}"
    )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    train()
