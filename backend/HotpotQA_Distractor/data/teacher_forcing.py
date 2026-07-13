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

Task:
Answer the question using the Evidence Documents.

IMPORTANT RULES:

- Evidence Documents are retrieved documents.
- Whenever a reasoning step uses information from any document,
  place <RETRIEVAL> immediately before that step.
- Every document-derived fact must be preceded by <RETRIEVAL>.
- Do not discuss retrieval.
- Do not explain your process.
- Do not write meta-reasoning.
- Keep reasoning short.
- Do not output JSON.
- End with exactly:

Final Answer: ...

Output format:

<RETRIEVAL>
fact from document

<RETRIEVAL>
fact from document

reasoning

Final Answer: ...

Question:
{sample["question"]}

Supporting Facts:
{facts_text}

Evidence Documents:
{docs_text}

Example:

Question:
Which magazine was started first Arthur's Magazine or First for Women?

Output:

<RETRIEVAL>
Arthur's Magazine was published from 1844 to 1846.

<RETRIEVAL>
First for Women was started in 1989.

1844 is earlier than 1989.

Final Answer: Arthur's Magazine
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

        device = next(model.parameters()).device

        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
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
