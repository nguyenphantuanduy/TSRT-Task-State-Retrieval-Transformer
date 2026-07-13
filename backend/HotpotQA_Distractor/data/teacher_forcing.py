from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from HotpotQA_Distractor.data.load_data import (
    load_hotpotqa,
)

import torch


MODEL_NAME = "Qwen/Qwen3-8B"

MAX_INPUT_LENGTH = 4096
MAX_NEW_TOKENS = 512


def build_teacher_prompt(sample):

    support_titles = set(
        sample["supporting_facts"]["title"]
    )

    positive_docs = []

    for title, sentences in zip(
        sample["context"]["title"],
        sample["context"]["sentences"],
    ):

        if title not in support_titles:
            continue

        positive_docs.append(
            f"[DOCUMENT]\n"
            f"Title: {title}\n"
            f"Content:\n"
            f"{' '.join(sentences)}"
        )

    supporting_facts = []

    for title, sent_id in zip(
        sample["supporting_facts"]["title"],
        sample["supporting_facts"]["sent_id"],
    ):
        supporting_facts.append(
            f"- {title} | sentence {sent_id}"
        )

    docs_text = "\n\n".join(
        positive_docs
    )

    facts_text = "\n".join(
        supporting_facts
    )

    prompt = f"""
You are creating supervision data for a retrieval-augmented language model.

Your job:

1. Answer the question using the provided evidence.
2. Write a short reasoning chain.
3. Whenever the reasoning requires looking at external evidence,
   insert the token:

<RETRIEVAL>

Rules:

- Use <RETRIEVAL> naturally before evidence-based reasoning.
- Insert <RETRIEVAL> only when a human would need to consult evidence.
- Avoid placing <RETRIEVAL> before every sentence.
- Use as few retrieval markers as possible.
- Only place <RETRIEVAL> when the next reasoning step depends on information from the provided documents.
- Use the evidence documents as the source of reasoning.
- Keep the reasoning concise.
- Do NOT output JSON.
- Do NOT explain the labeling process.
- End with exactly:

Final Answer: ...

Question:
{sample["question"]}

Ground Truth Answer:
{sample["answer"]}

Supporting Facts:
{facts_text}

Evidence Documents:
{docs_text}

Example:

Question:
Which university did John Smith attend?

Output:

<RETRIEVAL>
The document about John Smith states that he attended Harvard University.

Therefore John Smith attended Harvard University.

Final Answer: Harvard University
""".strip()

    return prompt


def extract_teacher_text(
    generated_text,
    prompt,
):
    return generated_text[
        len(prompt):
    ].strip()


def teacher_labeling():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()

    dataset = load_hotpotqa(
        "distractor"
    )

    for sample in dataset["train"]:

        prompt = build_teacher_prompt(
            sample
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
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

        generated_text = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        teacher_text = extract_teacher_text(
            generated_text,
            prompt,
        )

        yield {
            "id": sample["id"],
            "question": sample["question"],
            "answer": sample["answer"],
            "prompt": prompt,
            "teacher_text": teacher_text,
            "raw_sample": sample,
        }


def print_hotpot_sample(sample):

    support_titles = set(
        sample["supporting_facts"]["title"]
    )

    print()
    print("=" * 120)
    print("QUESTION")
    print("=" * 120)
    print(sample["question"])

    print()
    print("=" * 120)
    print("GROUND TRUTH ANSWER")
    print("=" * 120)
    print(sample["answer"])

    print()
    print("=" * 120)
    print("SUPPORTING FACTS")
    print("=" * 120)

    for title, sent_id in zip(
        sample["supporting_facts"]["title"],
        sample["supporting_facts"]["sent_id"],
    ):
        print(
            f"{title} | sentence {sent_id}"
        )

    print()
    print("=" * 120)
    print("POSITIVE DOCUMENTS")
    print("=" * 120)

    for title, sentences in zip(
        sample["context"]["title"],
        sample["context"]["sentences"],
    ):

        if title not in support_titles:
            continue

        print()
        print(f"[DOC] {title}")
        print("-" * 80)

        print(
            " ".join(sentences)
        )


if __name__ == "__main__":

    stream = teacher_labeling()

    for idx, item in enumerate(stream):

        print()
        print("#" * 120)
        print(
            f"TEACHER SAMPLE {idx}"
        )
        print("#" * 120)

        sample = item["raw_sample"]

        print_hotpot_sample(
            sample
        )

        print()
        print("=" * 120)
        print("PROMPT")
        print("=" * 120)

        print(
            item["prompt"]
        )

        print()
        print("=" * 120)
        print("TEACHER OUTPUT")
        print("=" * 120)

        print(
            item["teacher_text"]
        )

        if idx >= 2:
            break
