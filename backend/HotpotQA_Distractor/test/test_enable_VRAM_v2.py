import copy
import torch
from collections import defaultdict
from torch.utils.data import IterableDataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

from HotpotQA_Distractor.data.load_data import load_hotpotqa

MODEL_NAME = "Qwen/Qwen3-1.7B"
MAX_LENGTH = 2048

EASY_SAMPLES = 640
MEDIUM_SAMPLES = 640
HARD_SAMPLES = 640

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

DEVICE = "cuda"

def freeze_all(model):

    for p in model.parameters():
        p.requires_grad = False


def unfreeze_attention_ffn_42layer(model):

    for name, param in model.named_parameters():

        lower_name = name.lower()

        # ==================================================
        # Attention layers 0 -> 27
        # ==================================================
        for layer_idx in range(42):

            prefix = f"model.layers.{layer_idx}."

            if prefix not in lower_name:
                continue

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

            break

        # ==================================================
        # FFN layers 28 -> 41
        # ==================================================
        for layer_idx in range(28, 42):

            prefix = f"model.layers.{layer_idx}."

            if prefix not in lower_name:
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


def print_breakdown(model):

    attn = 0
    ffn = 0

    for name, p in model.named_parameters():

        if not p.requires_grad:
            continue

        if any(
            k in name
            for k in [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ]
        ):
            attn += p.numel()

        elif any(
            k in name
            for k in [
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        ):
            ffn += p.numel()

    print()
    print("Breakdown")
    print("-" * 80)
    print(f"Attention params : {attn:,}")
    print(f"FFN params       : {ffn:,}")
    print("-" * 80)


def expand_to_42_layers(model):

    layers = model.model.layers

    original_num_layers = len(layers)

    print(f"Original layers : {original_num_layers}")

    while len(layers) < 56:

        src_idx = len(layers) % original_num_layers

        layers.append(
            copy.deepcopy(
                layers[src_idx]
            )
        )

    print(f"Expanded layers : {len(layers)}")

    return model

def build_input(sample):

    docs = []

    for title, sentences in zip(
        sample["context"]["title"],
        sample["context"]["sentences"]
    ):
        docs.append(
            f"{title}\n{' '.join(sentences)}"
        )

    context = "\n\n".join(docs)

    text = (
        f"Question: {sample['question']}\n\n"
        f"Context:\n{context}\n\n"
        f"Answer: {sample['answer']}"
    )

    return text


class HotpotBalancedDataset(IterableDataset):

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        easy_samples=640,
        medium_samples=640,
        hard_samples=640
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

            text = build_input(sample)

            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )

            yield {
                "input_ids":
                    encoded["input_ids"].squeeze(0),

                "attention_mask":
                    encoded["attention_mask"].squeeze(0),

                "labels":
                    encoded["input_ids"].squeeze(0),
            }

            counts[level] += 1

            if counts[level] % 100 == 0:
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


class VRAMCallback(TrainerCallback):

    def on_step_end(
        self,
        args,
        state,
        control,
        **kwargs
    ):

        if not torch.cuda.is_available():
            return

        allocated = (
            torch.cuda.memory_allocated()
            / 1024**3
        )

        reserved = (
            torch.cuda.memory_reserved()
            / 1024**3
        )

        peak = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

        print(
            f"\n[STEP {state.global_step}] "
            f"allocated={allocated:.2f}GB "
            f"reserved={reserved:.2f}GB "
            f"peak={peak:.2f}GB")


# def main():

#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_NAME,
#         torch_dtype=torch.bfloat16,
#         trust_remote_code=True,
#     )

#     print(
#         f"Before expand: "
#         f"{len(model.model.layers)}"
#     )

#     model = expand_to_42_layers(model)

#     print(
#         f"After expand : "
#         f"{len(model.model.layers)}"
#     )

#     freeze_all(model)

#     unfreeze_attention_ffn_42layer(model)

#     total, trainable = count_parameters(model)

#     print()
#     print("=" * 80)
#     print(
#         f"Total params     : "
#         f"{total:,}"
#     )
#     print(
#         f"Trainable params : "
#         f"{trainable:,}"
#     )
#     print("=" * 80)

#     print_breakdown(model)

#     print()
#     print("First trainable tensors:")
#     print("-" * 80)

#     shown = 0

#     for name, p in model.named_parameters():

#         if not p.requires_grad:
#             continue

#         print(name)

#         shown += 1

#         if shown >= 30:
#             break


if __name__ == "__main__":

    dataset = load_hotpotqa("distractor")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print(
        f"Before expand : "
        f"{len(model.model.layers)}"
    )

    model = expand_to_42_layers(model)

    print(
        f"After expand  : "
        f"{len(model.model.layers)}"
    )

    freeze_all(model)

    unfreeze_attention_ffn_42layer(model)

    total, trainable = count_parameters(model)

    print()
    print("=" * 80)
    print(f"Total params     : {total:,}")
    print(f"Trainable params : {trainable:,}")
    print("=" * 80)

    print_breakdown(model)

    train_dataset = HotpotBalancedDataset(
        dataset["train"],
        tokenizer,
        easy_samples=32,
        medium_samples=32,
        hard_samples=32,
    )

    training_args = TrainingArguments(
        output_dir="./tmp_vram_test",

        per_device_train_batch_size=1,

        gradient_accumulation_steps=GRAD_ACCUM_STEPS,

        max_steps=MAX_STEPS,

        learning_rate=1e-5,

        bf16=True,

        logging_steps=1,

        save_strategy="no",

        eval_strategy="no",

        report_to="none",

        dataloader_num_workers=0,

        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollator(tokenizer),
        callbacks=[VRAMCallback()],
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    trainer.train()

    print()
    print("===== FINAL VRAM =====")

    allocated = (
        torch.cuda.memory_allocated()
        / 1024**3
    )

    reserved = (
        torch.cuda.memory_reserved()
        / 1024**3
    )

    peak = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    print(
        f"Allocated : {allocated:.2f} GB"
    )

    print(
        f"Reserved  : {reserved:.2f} GB"
    )

    print(
        f"Peak      : {peak:.2f} GB"
    )