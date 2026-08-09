import torch
import random

from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase

from utils.utils import batch_tokenize_documents


def build_tsrt_retriever_question_batch(
    samples: list[dict],
    tokenizer: PreTrainedTokenizerBase,
    batch: dict,
):
    """
    Build question batch for TSRT retriever.

    Returns:
        input_ids:      (B, L)
        attention_mask: (B, L)
    """

    questions = [
        sample["question"]
        for sample in samples
    ]

    tokenized = tokenizer(
        questions,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )

    batch.update(
        {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }
    )

    return batch

def build_tsrt_retriever_document_batch(
    samples: list[dict],
    tokenizer: PreTrainedTokenizerBase,
    batch: dict,
    max_length: int,
):
    """
    Build retrieval documents for TSRT retriever.

    Returns:
        document_ids:
            (B, D, L_doc)

        document_padding_mask:
            (B, D, L_doc)

        usefulness_score_matrix:
            (B, D)
    """

    B = len(samples)

    # =====================================================
    # BUILD DOCUMENTS
    # =====================================================

    all_documents = []
    all_labels = []

    # Cache:
    # (title, document_text)
    document_infos = []

    for sample in samples:

        positive_titles = set(
            sample["supporting_facts"]["title"]
        )

        docs = []
        labels = []
        infos = []

        for title, sentences in zip(
            sample["context"]["title"],
            sample["context"]["sentences"],
        ):

            document = (
                title
                + "\n"
                + " ".join(sentences)
            )

            infos.append(
                (title, document)
            )

            docs.append(document)

            labels.append(
                1 if title in positive_titles else 0
            )

        document_infos.append(infos)
        all_documents.append(docs)
        all_labels.append(labels)

    # =====================================================
    # EXTRA NEGATIVE DOCUMENT
    #
    # Each sample contributes one negative
    # document to every other sample.
    # =====================================================

    for i, sample in enumerate(samples):

        positive_titles = set(
            sample["supporting_facts"]["title"]
        )

        for j in range(B):

            if i == j:
                continue

            candidates = []

            for title, document in document_infos[j]:

                if title not in positive_titles:
                    candidates.append(document)

            if candidates:

                all_documents[i].append(
                    random.choice(candidates)
                )

                all_labels[i].append(0)

    # =====================================================
    # TOKENIZE DOCUMENTS
    # =====================================================

    tokenized = batch_tokenize_documents(
        samples=all_documents,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    document_input_ids = tokenized["input_ids"]
    document_attention_mask = tokenized["attention_mask"]

    _, D, _ = document_input_ids.shape

    # =====================================================
    # USEFULNESS SCORE MATRIX
    # =====================================================

    usefulness_score_matrix = torch.full(
        (B, D),
        -1,
        dtype=torch.long,
    )

    for b, labels in enumerate(all_labels):

        usefulness_score_matrix[
            b,
            :len(labels),
        ] = torch.tensor(
            labels,
            dtype=torch.long,
        )

    # =====================================================
    # UPDATE BATCH
    # =====================================================

    batch.update(
        {
            "document_ids": document_input_ids,
            "document_padding_mask": document_attention_mask,
            "usefulness_score_matrix": usefulness_score_matrix,
        }
    )

    return batch


@dataclass
class TSRTRetrieverCollator:
    """
    Data collator for TSRT retriever training.
    """

    tokenizer: PreTrainedTokenizerBase
    document_max_length: int = 1024

    def __call__(
        self,
        samples: list[dict],
    ) -> dict:
        """
        Build a TSRT retriever training batch.

        Returns:
            input_ids:
                (B, L)

            attention_mask:
                (B, L)

            document_ids:
                (B, D, L_doc)

            document_padding_mask:
                (B, D, L_doc)

            usefulness_score_matrix:
                (B, D)
        """

        batch = {}

        # =====================================================
        # Question
        # =====================================================

        batch = build_tsrt_retriever_question_batch(
            samples=samples,
            tokenizer=self.tokenizer,
            batch=batch,
        )

        # =====================================================
        # Retrieval Documents
        # =====================================================

        batch = build_tsrt_retriever_document_batch(
            samples=samples,
            tokenizer=self.tokenizer,
            batch=batch,
            max_length=self.document_max_length,
        )

        return batch
