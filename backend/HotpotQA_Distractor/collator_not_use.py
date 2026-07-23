from __future__ import annotations

import torch
import spacy

from spacy.cli import download

from utils.utils import batch_tokenize_documents


def load_spacy_model():

    model_name = "en_core_web_sm"

    try:
        return spacy.load(model_name)

    except OSError:

        print(f"{model_name} not found. Downloading...")

        download(model_name)

        return spacy.load(model_name)


nlp = load_spacy_model()


# ============================================================
# DOCUMENT
# ============================================================

def build_tsrt_document_batch(
    sample: dict,
    tokenizer,
    batch: dict,
    max_length: int,
):

    positive_titles = set(
        sample["supporting_facts"]["title"]
    )

    documents = []
    labels = []

    for title, sentences in zip(
        sample["context"]["title"],
        sample["context"]["sentences"],
    ):

        document = title + "\n" + " ".join(sentences)

        documents.append(document)

        labels.append(
            1 if title in positive_titles else 0
        )

    tokenized = batch_tokenize_documents(
        samples=[documents],
        tokenizer=tokenizer,
        max_length=max_length,
    )

    document_input_ids = tokenized["input_ids"]          # (1,D,L)
    document_attention_mask = tokenized["attention_mask"]

    L = batch["input_ids"].shape[1]
    D = len(labels)

    usefulness_score_matrix = torch.full(
        (1, L, D),
        -1,
        dtype=torch.long,
    )

    usefulness_score_matrix[0] = (
        torch.tensor(labels)
        .unsqueeze(0)
        .expand(L, D)
    )

    batch.update(
        {
            "document_ids": document_input_ids,
            "document_padding_mask": document_attention_mask,
            "usefulness_score_matrix": usefulness_score_matrix,
        }
    )

    return batch


# ============================================================
# QUESTION + ANSWER
# ============================================================

def build_tsrt_question_answer_batch(
    sample: dict,
    tokenizer,
    batch: dict,
):

    question_text = sample["question"] + "\n"

    question_ids = tokenizer(
        question_text,
        add_special_tokens=False,
    )["input_ids"]

    answer_text = sample["teacher_answer"]

    answer_ids = tokenizer(
        answer_text,
        add_special_tokens=False,
    )["input_ids"]

    input_ids = question_ids + answer_ids

    attention_mask = [1] * len(input_ids)

    question_mask = [1] * len(input_ids)

    if len(question_ids) > 1:
        question_mask[: len(question_ids) - 1] = [0] * (
            len(question_ids) - 1
        )

    labels = (
        [-100] * len(question_ids)
        + answer_ids
    )

    retrieval_labels = [0] * len(input_ids)

    if len(question_ids) > 0:
        retrieval_labels[len(question_ids) - 1] = 1

    answer_offset = len(question_ids)

    for _, _, end_char in split_sentences_with_offsets(
        answer_text
    ):

        prefix_ids = tokenizer(
            answer_text[:end_char],
            add_special_tokens=False,
        )["input_ids"]

        token_end = (
            answer_offset
            + len(prefix_ids)
            - 1
        )

        if (
            answer_offset
            <= token_end
            < len(retrieval_labels)
        ):
            retrieval_labels[token_end] = 1

    batch.update(
        {
            "input_ids": torch.tensor(
                [input_ids],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [attention_mask],
                dtype=torch.long,
            ),
            "question_mask": torch.tensor(
                [question_mask],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                [labels],
                dtype=torch.long,
            ),
            "retrieval_decision_labels": torch.tensor(
                [retrieval_labels],
                dtype=torch.long,
            ),
        }
    )

    return batch


# ============================================================
# SENTENCE SPLIT
# ============================================================

def split_sentences_with_offsets(text: str):

    doc = nlp(text)

    return [
        (
            sent.text,
            sent.start_char,
            sent.end_char,
        )
        for sent in doc.sents
    ]


# ============================================================
# MASK USEFULNESS
# ============================================================

def fix_usefulness_score_matrix(
    batch: dict,
):

    usefulness_score_matrix = batch[
        "usefulness_score_matrix"
    ]

    retrieval_decision_labels = batch[
        "retrieval_decision_labels"
    ]

    retrieval_mask = (
        retrieval_decision_labels == 1
    ).unsqueeze(-1)

    usefulness_score_matrix = torch.where(
        retrieval_mask,
        usefulness_score_matrix,
        torch.full_like(
            usefulness_score_matrix,
            -1,
        ),
    )

    batch[
        "usefulness_score_matrix"
    ] = usefulness_score_matrix

    return batch


from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase

@dataclass
class TSRTDataCollator:
    """
    Data collator for TSRT training (batch size = 1 only).
    """

    tokenizer: PreTrainedTokenizerBase
    document_max_length: int = 1024

    def __call__(
        self,
        samples: list[dict],
    ) -> dict:
        """
        Build a TSRT training batch.

        Note:
            This collator assumes per_device_train_batch_size = 1.
        """

        assert (
            len(samples) == 1
        ), "TSRTDataCollator only supports batch_size=1."

        sample = samples[0]

        batch = {}

        # =====================================================
        # Question + Teacher Answer
        # =====================================================

        batch = build_tsrt_question_answer_batch(
            sample=sample,
            tokenizer=self.tokenizer,
            batch=batch,
        )

        # =====================================================
        # Retrieval Documents
        # =====================================================

        batch = build_tsrt_document_batch(
            sample=sample,
            tokenizer=self.tokenizer,
            batch=batch,
            max_length=self.document_max_length,
        )

        # =====================================================
        # Keep usefulness labels only on retrieval tokens
        # =====================================================

        batch = fix_usefulness_score_matrix(
            batch=batch,
        )

        return batch