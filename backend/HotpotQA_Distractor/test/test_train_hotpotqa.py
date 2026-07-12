import torch

from collections import defaultdict
from torch.utils.data import IterableDataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

from HotpotQA_Distractor.data.load_data import load_hotpotqa


MODEL_NAME = "Qwen/Qwen3-1.7B"

MAX_LENGTH = 2048

EASY_SAMPLES = 4300
MEDIUM_SAMPLES = 13500
HARD_SAMPLES = 4500

TOTAL_SAMPLES = (
    EASY_SAMPLES
    + MEDIUM_SAMPLES
    + HARD_SAMPLES
)

GRAD_ACCUM_STEPS = 64

MAX_STEPS = (
    TOTAL_SAMPLES
    // GRAD_ACCUM_STEPS
)

OUTPUT_DIR = "./checkpoint_hotpotqa"


def freeze_all(model):

    for p in model.parameters():
        p.requires_grad = False


def unfreeze_attention_last_12_ffn_and_norm(model):

    for name, param in model.named_parameters():

        lower_name = name.lower()

        # ==========================
        # Attention (all layers)
        # ==========================
        if any(
            k in lower_name
            for k in [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ]
        ):
            param.requires_grad = True
            continue

        # ==========================
        # RMSNorm (all layers)
        # input_layernorm
        # post_attention_layernorm
        # final model.norm
        # ==========================
        if any(
            k in lower_name
            for k in [
                "input_layernorm",
                "post_attention_layernorm",
                ".norm",
            ]
        ):
            param.requires_grad = True
            continue

        # ==========================
        # FFN 12 layer cuối
        # gate_proj
        # up_proj
        # down_proj
        # ==========================
        for layer_idx in range(12, 24):

            layer_prefix = f"layers.{layer_idx}."

            if layer_prefix not in lower_name:
                continue

            if any(
                k in lower_name
                for k in [
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]
            ):
                param.requires_grad = True
                break


def count_parameters(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


def build_prompt(sample):

    docs = []

    for title, sentences in zip(
        sample["context"]["title"],
        sample["context"]["sentences"]
    ):
        docs.append(
            f"{title}\n{' '.join(sentences)}"
        )

    context = "\n\n".join(docs)

    prompt = (
        f"Question: {sample['question']}\n\n"
        f"Context:\n{context}\n\n"
        f"Answer:"
    )

    return prompt


class HotpotBalancedDataset(IterableDataset):

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        easy_samples,
        medium_samples,
        hard_samples,
    ):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer

        self.targets = {
            "easy": easy_samples,
            "medium": medium_samples,
            "hard": hard_samples,
        }

    def __iter__(self):

        counts = defaultdict(int)

        for sample in self.dataset:

            level = sample["level"].lower()

            if level not in self.targets:
                continue

            if counts[level] >= self.targets[level]:
                continue

            prompt = build_prompt(sample)

            answer = str(sample["answer"]).strip()

            full_text = (
                prompt
                + " "
                + answer
            )

            full_encoded = self.tokenizer(
                full_text,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )

            prompt_encoded = self.tokenizer(
                prompt,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )

            input_ids = (
                full_encoded["input_ids"]
                .squeeze(0)
            )

            attention_mask = (
                full_encoded["attention_mask"]
                .squeeze(0)
            )

            labels = input_ids.clone()

            prompt_len = (
                prompt_encoded["input_ids"]
                .shape[1]
            )

            labels[:prompt_len] = -100

            yield {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

            counts[level] += 1

            if counts[level] % 500 == 0:

                print(
                    f"{level:<6}: "
                    f"{counts[level]}/"
                    f"{self.targets[level]}"
                )

            done = all(
                counts[k] >= self.targets[k]
                for k in self.targets
            )

            if done:

                print("\nCollected samples:")
                print(
                    f"easy   : {counts['easy']}"
                )
                print(
                    f"medium : {counts['medium']}"
                )
                print(
                    f"hard   : {counts['hard']}"
                )

                break


class DataCollator:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):

        input_ids = torch.nn.utils.rnn.pad_sequence(
            [x["input_ids"] for x in batch],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )

        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [x["attention_mask"] for x in batch],
            batch_first=True,
            padding_value=0
        )

        labels = torch.nn.utils.rnn.pad_sequence(
            [x["labels"] for x in batch],
            batch_first=True,
            padding_value=-100
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def main():

    dataset = load_hotpotqa("distractor")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    freeze_all(model)

    unfreeze_attention_last_12_ffn_and_norm(model)

    total_params, trainable_params = (
        count_parameters(model)
    )

    print()
    print("=" * 80)
    print(f"Total params     : {total_params:,}")
    print(f"Trainable params : {trainable_params:,}")
    print(f"Total samples    : {TOTAL_SAMPLES}")
    print(f"Max steps        : {MAX_STEPS}")
    print("=" * 80)

    train_dataset = HotpotBalancedDataset(
        dataset["train"],
        tokenizer,
        easy_samples=EASY_SAMPLES,
        medium_samples=MEDIUM_SAMPLES,
        hard_samples=HARD_SAMPLES,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        per_device_train_batch_size=1,

        gradient_accumulation_steps=GRAD_ACCUM_STEPS,

        max_steps=MAX_STEPS,

        learning_rate=1e-5,

        bf16=True,

        logging_steps=10,

        save_strategy="steps",

        save_steps=100,

        save_total_limit=2,

        eval_strategy="no",

        report_to="none",

        remove_unused_columns=False,

        dataloader_num_workers=0,

        hub_model_id="nguyenphantuanduy/temp-models",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollator(tokenizer),
    )

    trainer.train()

    trainer.save_model(
        OUTPUT_DIR
    )

    tokenizer.save_pretrained(
        OUTPUT_DIR
    )

    trainer.push_to_hub()

    tokenizer.push_to_hub(
        "nguyenphantuanduy/temp-models"
    )

    print()
    print("Training finished.")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()