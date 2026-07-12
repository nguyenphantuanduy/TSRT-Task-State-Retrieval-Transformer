import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from HotpotQA_Distractor.data.load_data import load_hotpotqa


BASE_MODEL = "Qwen/Qwen3-1.7B"

FINETUNED_MODEL = (
    "nguyenphantuanduy/temp-models"
)

MAX_NEW_TOKENS = 64
MAX_LENGTH = 2048


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

    return (
        f"Question: {sample['question']}\n\n"
        f"Context:\n{context}\n\n"
        f"Answer:"
    )


def collect_samples(dataset):

    selected = {}

    for sample in dataset:

        level = sample["level"].lower()

        if level not in [
            "easy",
            "medium",
            "hard",
        ]:
            continue

        if level not in selected:
            selected[level] = sample

        if len(selected) == 3:
            break

    return selected


def generate_answer(
    model,
    tokenizer,
    sample,
):

    prompt = build_prompt(sample)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    inputs = {
        k: v.to(model.device)
        for k, v in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    prediction = (
        generated[len(prompt):]
        .strip()
    )

    return prediction


def load_model(model_name):

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    model.eval()

    return tokenizer, model


def print_result(
    level,
    sample,
    base_answer,
    finetuned_answer,
):

    print()
    print("=" * 120)
    print(f"LEVEL : {level.upper()}")
    print("=" * 120)

    print()
    print("QUESTION")
    print("-" * 120)
    print(sample["question"])

    print()
    print("GROUND TRUTH")
    print("-" * 120)
    print(sample["answer"])

    print()
    print("QWEN3-1.7B")
    print("-" * 120)
    print(base_answer)

    print()
    print("TSRT FINETUNED")
    print("-" * 120)
    print(finetuned_answer)


def main():

    print()
    print("=" * 120)
    print("LOADING BASE MODEL")
    print("=" * 120)

    base_tokenizer, base_model = (
        load_model(BASE_MODEL)
    )

    print()
    print("=" * 120)
    print("LOADING FINETUNED MODEL")
    print("=" * 120)

    ft_tokenizer, ft_model = (
        load_model(FINETUNED_MODEL)
    )

    print()
    print("=" * 120)
    print("LOADING HOTPOTQA")
    print("=" * 120)

    dataset = load_hotpotqa(
        "distractor"
    )

    samples = collect_samples(
        dataset["validation"]
    )

    for level in [
        "easy",
        "medium",
        "hard",
    ]:

        sample = samples[level]

        print()
        print(
            f"Generating {level}..."
        )

        base_answer = generate_answer(
            base_model,
            base_tokenizer,
            sample,
        )

        finetuned_answer = generate_answer(
            ft_model,
            ft_tokenizer,
            sample,
        )

        print_result(
            level,
            sample,
            base_answer,
            finetuned_answer,
        )


if __name__ == "__main__":
    main()