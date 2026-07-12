import torch
from torch.utils.data import IterableDataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

from data.load_data import load_hotpotqa


MODEL_NAME = "Qwen/Qwen3-1.7B"

MAX_LENGTH = 4096
MAX_SAMPLES = 640   # 10 optimizer step nếu grad_acc=64
DEVICE = "cuda"


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_attention_only(model):

    for name, param in model.named_parameters():

        name = name.lower()

        if any(
            k in name
            for k in [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ]
        ):
            param.requires_grad = True


def count_parameters(model):

    total = sum(p.numel() for p in model.parameters())

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


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


class HotpotIterableDataset(IterableDataset):

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        max_samples=None
    ):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_samples = max_samples

    def __iter__(self):

        count = 0

        for sample in self.dataset:

            if (
                self.max_samples is not None
                and count >= self.max_samples
            ):
                break

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

            count += 1


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
            "labels": labels
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
            f"peak={peak:.2f}GB"
        )


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
    unfreeze_attention_only(model)

    total_params, trainable_params = count_parameters(model)

    print()
    print("=" * 80)
    print(f"Total params     : {total_params:,}")
    print(f"Trainable params : {trainable_params:,}")
    print("=" * 80)

    train_dataset = HotpotIterableDataset(
        dataset["train"],
        tokenizer,
        max_samples=MAX_SAMPLES
    )

    training_args = TrainingArguments(
        output_dir="./tmp_vram_test",

        per_device_train_batch_size=1,

        gradient_accumulation_steps=64,

        max_steps=10,

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

    torch.cuda.reset_peak_memory_stats()

    trainer.train()

    print("\n===== FINAL VRAM =====")

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

    print(f"Allocated : {allocated:.2f} GB")
    print(f"Reserved  : {reserved:.2f} GB")
    print(f"Peak      : {peak:.2f} GB")


if __name__ == "__main__":
    main()