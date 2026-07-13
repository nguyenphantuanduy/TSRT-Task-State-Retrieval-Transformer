from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

from HotpotQA_Distractor.data.load_data import (
    load_hotpotqa,
)

import torch
import time

MODEL_NAME = "Qwen/Qwen3-8B"

MAX_INPUT_LENGTH = 2048
MAX_NEW_TOKENS = 256
BATCH_SIZE = 64


class StopOnTokens(StoppingCriteria):

    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(
        self,
        input_ids,
        scores,
        **kwargs,
    ):

        batch_finished = []

        for row in input_ids:

            sample_finished = False

            for stop_ids in self.stop_token_ids:

                if row.shape[0] < len(stop_ids):
                    continue

                if torch.equal(
                    row[-len(stop_ids):],
                    stop_ids,
                ):
                    sample_finished = True
                    break

            batch_finished.append(
                sample_finished
            )

        return all(batch_finished)


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
Question:
{sample["question"]}

Ground Truth Answer:
{sample["answer"]}

Supporting Facts:
{facts_text}

Evidence Documents:
{docs_text}

Task:
Generate a concise evidence-based reasoning.

Requirements:
- When referring to information from a document, mention the document title instead of its order.

Reasoning:
""".strip()
    
    return prompt



def extract_teacher_text(
    output_ids,
    input_length,
    tokenizer,
):

    generated_ids = output_ids[input_length:]

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

import re

def postprocess_teacher_text(
    teacher_text: str,
    answer: str,
):
    patterns = [
        r"(^|\n)\s*Final\s+Answer\s*:",
        r"(^|\n)\s*Answer\s*:",
    ]

    cut_pos = None

    for pattern in patterns:

        match = re.search(
            pattern,
            teacher_text,
            flags=re.IGNORECASE,
        )

        if match:

            if cut_pos is None:
                cut_pos = match.start()
            else:
                cut_pos = min(
                    cut_pos,
                    match.start()
                )

    if cut_pos is not None:
        teacher_text = teacher_text[:cut_pos]

    teacher_text = teacher_text.strip()

    return (
        f"{teacher_text}\n\n"
        f"Answer:\n{answer}"
    )


def process_batch(
    batch,
    tokenizer,
    model,
    stopping_criteria,
):

    prompts = [
        build_teacher_prompt(x)
        for x in batch
    ]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        padding=True,
    )

    device = next(model.parameters()).device

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    prompt_length = inputs[
        "input_ids"
    ].shape[1]

    torch.cuda.synchronize()

    start_time = time.time()

    with torch.no_grad():

        outputs = model.generate(
            **inputs,

            max_new_tokens=MAX_NEW_TOKENS,

            do_sample=False,

            stopping_criteria=stopping_criteria,

            pad_token_id=tokenizer.eos_token_id,
        )

    torch.cuda.synchronize()

    generate_time = (
        time.time() - start_time
    )

    print(
        f"Batch={len(batch)} | "
        f"Total={generate_time:.3f}s | "
        f"Per sample={generate_time/len(batch):.3f}s"
    )

    generated_part = outputs[
        :,
        prompt_length:
    ]

    for i, sample in enumerate(batch):

        teacher_text = tokenizer.decode(
            generated_part[i],
            skip_special_tokens=True,
        ).strip()

        teacher_text = postprocess_teacher_text(
            teacher_text=teacher_text,
            answer=sample["answer"],
        )

        yield {
            "id": sample["id"],
            "question": sample["question"],
            "answer": sample["answer"],
            "prompt": prompts[i],
            "teacher_text": teacher_text,
            "raw_sample": sample,
        }

def teacher_labeling():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    stop_words = ["Answer"]

    stop_token_ids = []

    for word in stop_words:

        ids = tokenizer(
            word,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]

        stop_token_ids.append(
            ids.to("cuda")
        )

    stopping_criteria = StoppingCriteriaList(
        [
            StopOnTokens(
                stop_token_ids
            )
        ]
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

    batch = []

    for sample in dataset["train"]:

        batch.append(sample)

        if len(batch) < BATCH_SIZE:
            continue

        yield from process_batch(
            batch=batch,
            tokenizer=tokenizer,
            model=model,
            stopping_criteria=stopping_criteria,
        )

        batch = []

    # xử lý batch cuối còn dư
    if len(batch) > 0:

        print(
            f"Processing final batch: {len(batch)}"
        )

        yield from process_batch(
            batch=batch,
            tokenizer=tokenizer,
            model=model,
            stopping_criteria=stopping_criteria,
        )


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

        if idx >= 63:
            break
