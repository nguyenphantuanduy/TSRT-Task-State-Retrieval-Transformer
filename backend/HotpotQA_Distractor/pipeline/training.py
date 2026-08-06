from __future__ import annotations

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    EarlyStoppingCallback,
)

from ..collator import TSRTDataCollator
from ..data.load_data import load_tsrt_hotpotqa_teacher
from models.tsrt.trainer import TSRTTrainer
from utils.utils import freeze_for_tsrt_training


MODEL_NAME = "nguyenphantuanduy/TSRT-Qwen3-1.7B"


def train():

    # =====================================================
    # DEVICE
    # =====================================================

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =====================================================
    # DATASET
    # =====================================================

    print("Loading dataset...")

    dataset = load_tsrt_hotpotqa_teacher()

    train_dataset = dataset["train"].shuffle(seed=42)

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

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=device,
    )

    # =====================================================
    # FREEZE
    # =====================================================

    print("Freezing model...")

    model = freeze_for_tsrt_training(model)

    # =====================================================
    # COLLATOR
    # =====================================================

    collator = TSRTDataCollator(
        tokenizer=tokenizer,
        document_max_length=384,
    )

    # =====================================================
    # TRAINING ARGS
    # =====================================================

    training_args = TrainingArguments(
        output_dir="./checkpoint",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,

        gradient_accumulation_steps=32,
        save_total_limit=2,
        learning_rate=2e-5,
        weight_decay=0.01,

        num_train_epochs=0.1,

        bf16=True,

        logging_strategy="steps",
        logging_steps=10,

        eval_strategy="steps",
        eval_steps=500,

        save_strategy="steps",
        save_steps=500,

        load_best_model_at_end=True,

        metric_for_best_model="eval_loss",
        greater_is_better=False,

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
            )
        ],
    )

    # =====================================================
    # TRAIN
    # =====================================================

    trainer.train()

    # =====================================================
    # SAVE
    # =====================================================

    trainer.save_model("./best_model")
    tokenizer.save_pretrained("./best_model")


if __name__ == "__main__":
    train()