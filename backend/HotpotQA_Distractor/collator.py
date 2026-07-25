import torch
import random
from utils.utils import batch_tokenize_documents
from transformers import PreTrainedTokenizerBase

from HotpotQA_Distractor.sentence_splitter import split_sentences_with_offsets


def build_tsrt_document_batch(
    samples: list[dict],
    tokenizer,
    batch: dict,
    max_length: int,
):
    """
    Build retrieval documents for a TSRT batch.

    Returns (update batch):
        document_input_ids:        (B, D, L_doc)
        document_attention_mask:   (B, D, L_doc)
        document_padding_mask:     (B, D)
        usefulness_score_matrix:   (B, L, D)
    """

    B = len(samples)

    # =====================================================
    # BUILD DOCUMENTS (ONLY ONCE)
    # =====================================================

    all_documents = []
    all_labels = []

    # cache: (title, document_text)
    document_infos = []

    for sample in samples:

        positive_titles = set(sample["supporting_facts"]["title"])

        docs = []
        labels = []
        infos = []

        for title, sentences in zip(
            sample["context"]["title"],
            sample["context"]["sentences"],
        ):

            document = title + "\n" + " ".join(sentences)

            infos.append((title, document))

            docs.append(document)
            labels.append(
                1 if title in positive_titles else 0
            )

        document_infos.append(infos)
        all_documents.append(docs)
        all_labels.append(labels)

    # =====================================================
    # EXTRA NEGATIVE DOCUMENT
    # Mỗi sample khác đóng góp 1 negative document
    # =====================================================

    for i, sample in enumerate(samples):

        positive_titles = set(sample["supporting_facts"]["title"])

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
    # TOKENIZE
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

    L = batch["input_ids"].shape[1]

    usefulness_score_matrix = torch.full(
        (B, L, D),
        -1,
        dtype=torch.long,
    )

    for b, labels in enumerate(all_labels):

        usefulness_score_matrix[
            b,
            :,
            : len(labels),
        ] = torch.tensor(
            labels,
            dtype=torch.long,
        ).unsqueeze(0)

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

def build_tsrt_question_answer_batch(
    samples: list[dict],
    tokenizer,
    batch: dict,
):
    """
    Build question + teacher answer batch.

    Train format (same as inference):

        tokenize(question + "\\n") + tokenize(answer)

    Returns (update batch):

        input_ids:                  (B, L)
        attention_mask:             (B, L)
        question_mask:              (B, L)
        labels:                     (B, L)
        retrieval_decision_labels:  (B, L)
        answer_position             (B,)

    question_mask:
        0 -> question tokens (except last question token)
        1 -> answer tokens

    labels:
        -100 -> ignore (question + padding)
        token id -> answer tokens

    retrieval_decision_labels:
        -1 -> padding
         0 -> normal token
         1 -> last token of question or last token of each answer sentence
    """

    # =====================================================
    # BUILD
    # =====================================================

    batch_input_ids = []
    batch_attention_mask = []
    batch_question_mask = []
    batch_labels = []
    batch_retrieval_labels = []
    batch_answer_position = []
    max_length = 0

    for sample in samples:

        # -------------------------------------------------
        # Question
        # -------------------------------------------------

        question_text = sample["question"] + "\n"

        question_ids = tokenizer(
            question_text,
            add_special_tokens=False,
        )["input_ids"]

        # -------------------------------------------------
        # Answer
        # -------------------------------------------------

        answer_text = sample["teacher_answer"]

        answer_ids = tokenizer(
            answer_text,
            add_special_tokens=False,
        )["input_ids"]

        answer_ids.append(tokenizer.eos_token_id)

        marker = "Answer:\n"
        answer_start = answer_text.rfind(marker)

        answer_position = -1
        if answer_start != -1:
            answer_start += len(marker)
            
            before_answer_ids = tokenizer(
                answer_text[:answer_start],
                add_special_tokens=False,
            )["input_ids"]

            answer_position = (
                len(question_ids)
                + len(before_answer_ids)
            )

        batch_answer_position.append(answer_position)

        # -------------------------------------------------
        # Full sequence
        # -------------------------------------------------

        input_ids = question_ids + answer_ids
        attention_mask = [1] * len(input_ids)

        # -------------------------------------------------
        # Question mask
        # -------------------------------------------------

        question_mask = [1] * len(input_ids)

        if len(question_ids) > 1:
            question_mask[: len(question_ids) - 1] = [0] * (
                len(question_ids) - 1
            )

        # -------------------------------------------------
        # Labels
        # -------------------------------------------------

        labels = (
            [-100] * len(question_ids)
            + answer_ids
        )

        # -------------------------------------------------
        # Retrieval decision labels
        # -------------------------------------------------

        retrieval_labels = [0] * len(input_ids)

        # last token of question
        if len(question_ids) > 0:
            retrieval_labels[len(question_ids) - 1] = 1

        # last token of every sentence in answer
        answer_offset = len(question_ids)

        sentences = split_sentences_with_offsets(answer_text)

        for sentence, start_char, end_char in sentences[:-1]:
            # tokenize prefix up to sentence end
            prefix_ids = tokenizer(
                answer_text[:end_char],
                add_special_tokens=False,
            )["input_ids"]

            token_end = answer_offset + len(prefix_ids) - 1

            if (
                answer_offset
                <= token_end
                < len(retrieval_labels)
            ):
                retrieval_labels[token_end] = 1

        # -------------------------------------------------

        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)
        batch_question_mask.append(question_mask)
        batch_labels.append(labels)
        batch_retrieval_labels.append(retrieval_labels)

        max_length = max(
            max_length,
            len(input_ids),
        )

    # =====================================================
    # PAD
    # =====================================================

    pad_id = tokenizer.pad_token_id

    for i in range(len(samples)):

        pad_len = max_length - len(batch_input_ids[i])

        batch_input_ids[i].extend(
            [pad_id] * pad_len
        )

        batch_attention_mask[i].extend(
            [0] * pad_len
        )

        batch_question_mask[i].extend(
            [1] * pad_len
        )

        batch_labels[i].extend(
            [-100] * pad_len
        )

        batch_retrieval_labels[i].extend(
            [-1] * pad_len
        )

    # =====================================================
    # UPDATE BATCH
    # =====================================================

    batch.update(
        {
            "input_ids": torch.tensor(
                batch_input_ids,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                batch_attention_mask,
                dtype=torch.long,
            ),
            "question_mask": torch.tensor(
                batch_question_mask,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                batch_labels,
                dtype=torch.long,
            ),
            "retrieval_decision_labels": torch.tensor(
                batch_retrieval_labels,
                dtype=torch.long,
            ),
            "answer_position": torch.tensor(
                batch_answer_position,
                dtype=torch.long,
            ),
        }
    )

    return batch

def fix_usefulness_score_matrix(
    batch: dict,
):
    """
    Mask usefulness score matrix using retrieval decision labels.

    Inputs:
        batch:
            usefulness_score_matrix:
                (B, L, D)

            retrieval_decision_labels:
                (B, L)

                -1 -> padding
                 0 -> normal token
                 1 -> retrieval token

    Behavior:

        For tokens where:
            retrieval_decision_labels != 1

        set all document usefulness labels to:
            -1

    Returns:
        updated batch
    """

    # =====================================================
    # GET TENSORS
    # =====================================================

    usefulness_score_matrix = batch[
        "usefulness_score_matrix"
    ]

    retrieval_decision_labels = batch[
        "retrieval_decision_labels"
    ]


    # =====================================================
    # BUILD MASK
    # =====================================================

    # (B, L)
    retrieval_mask = (
        retrieval_decision_labels == 1
    )


    # (B, L, 1)
    retrieval_mask = retrieval_mask.unsqueeze(-1)


    # =====================================================
    # MASK USEFULNESS MATRIX
    # =====================================================

    usefulness_score_matrix = torch.where(
        retrieval_mask,
        usefulness_score_matrix,
        torch.full_like(
            usefulness_score_matrix,
            -1,
        ),
    )


    # =====================================================
    # UPDATE
    # =====================================================

    batch[
        "usefulness_score_matrix"
    ] = usefulness_score_matrix


    return batch

from dataclasses import dataclass


@dataclass
class TSRTDataCollator:
    """
    Data collator for TSRT training.
    """

    tokenizer: PreTrainedTokenizerBase
    document_max_length: int = 1024

    def __call__(
        self,
        samples: list[dict],
    ) -> dict:
        """
        Build a TSRT training batch.

        Args:
            samples:
                List of dataset samples.

        Returns:
            Batch dictionary ready for TSRT forward().
        """

        batch = {}

        # =====================================================
        # Question + Teacher Answer
        # =====================================================

        batch = build_tsrt_question_answer_batch(
            samples=samples,
            tokenizer=self.tokenizer,
            batch=batch,
        )

        # =====================================================
        # Retrieval Documents
        # =====================================================

        batch = build_tsrt_document_batch(
            samples=samples,
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








if __name__ == "__main__":

    from datasets import load_dataset
    from transformers import AutoTokenizer

    # =====================================================
    # TOKENIZER
    # =====================================================

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-1.7B",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # =====================================================
    # DATASET (STREAMING)
    # =====================================================

    dataset = load_dataset(
        "nguyenphantuanduy/TSRT-HotpotQA-Teacher",
        split="train",
        streaming=True,
    )

    samples = []

    for sample in dataset:
        samples.append(sample)

        if len(samples) == 2:
            break

    # =====================================================
    # COLLATOR
    # =====================================================

    collator = TSRTDataCollator(
        tokenizer=tokenizer,
        document_max_length=1024,
    )

    batch = collator(samples)

    # =====================================================
    # PRINT SHAPES
    # =====================================================

    print("=" * 80)

    for key, value in batch.items():

        print(f"{key}")

        if isinstance(value, torch.Tensor):
            print(f"shape : {tuple(value.shape)}")
            print(f"dtype : {value.dtype}")

            if value.numel() <= 30:
                print(value)

        else:
            print(type(value))

        print("-" * 80)

    # =====================================================
    # CHECK SAMPLE 0
    # =====================================================

    print("\n" + "=" * 80)
    print("QUESTION")
    print(samples[0]["question"])

    print("\n" + "=" * 80)
    print("TEACHER ANSWER")
    print(samples[0]["teacher_answer"])

    print("\n" + "=" * 80)
    print("DECODED INPUT")
    print(
        tokenizer.decode(
            batch["input_ids"][0],
            skip_special_tokens=False,
        )
    )

    print("\n" + "=" * 80)
    print("QUESTION MASK")
    print(batch["question_mask"][0])

    print("\n" + "=" * 80)
    print("RETRIEVAL DECISION LABELS")
    print(batch["retrieval_decision_labels"][0])

    print("\n" + "=" * 80)
    print("USEFULNESS SCORE MATRIX SHAPE")
    print(batch["usefulness_score_matrix"].shape)

    print("\n" + "=" * 80)
    print("DOCUMENT IDS SHAPE")
    print(batch["document_ids"].shape)

    print("\n" + "=" * 80)
    print("FIRST DOCUMENT")

    print(
        tokenizer.decode(
            batch["document_ids"][0, 0],
            skip_special_tokens=False,
        )
    )

    print("\n" + "=" * 80)
    print("DOCUMENT LABELS AT FIRST RETRIEVAL TOKEN")

    retrieval_labels = batch["retrieval_decision_labels"][0]
    first_retrieval = (retrieval_labels == 1).nonzero(as_tuple=True)[0][0]

    print(
        batch["usefulness_score_matrix"][
            0,
            first_retrieval,
        ]
    )