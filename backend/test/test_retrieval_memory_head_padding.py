import torch

from models.tsrt.cache_utils import TSRTChosenDocumentCache
from models.tsrt.modeling_tsrt import TSRTRetrievalMemoryHead


class DummyDocumentCache:

    def reset_all(self):
        pass


def test_threshold_filter():
    """
    usefulness_threshold should remove
    low-score documents.
    """

    head = TSRTRetrievalMemoryHead()

    B = 2
    D = 5
    L_doc = 6
    H = 8

    usefulness_score = torch.tensor(
        [
            [[0.9, 0.8, 0.3, 0.1, 0.95]],
            [[0.4, 0.85, 0.2, 0.91, 0.15]],
        ]
    )

    retrieval_decision = torch.ones(
        B,
        1,
        1,
    )

    encoder_hidden_states = torch.randn(
        B,
        D,
        L_doc,
        H,
    )

    document_padding_mask = torch.ones(
        B,
        D,
        L_doc,
        dtype=torch.bool,
    )

    output = head(
        usefulness_score,
        retrieval_decision,
        encoder_hidden_states,
        document_padding_mask,
        TSRTChosenDocumentCache(),
        DummyDocumentCache(),
        usefulness_threshold=0.8,
    )

    print("\n=== Threshold ===")
    print(output.retrieval_memory.shape)
    print(output.encoder_hidden_states.shape)
    print(output.document_padding_mask.shape)

    assert output.retrieval_memory.shape[1] == 1


def test_topk_threshold():

    head = TSRTRetrievalMemoryHead()

    B = 2
    D = 8
    L_doc = 5
    H = 8

    usefulness_score = torch.rand(
        B,
        1,
        D,
    )

    retrieval_decision = torch.ones(
        B,
        1,
        1,
    )

    encoder_hidden_states = torch.randn(
        B,
        D,
        L_doc,
        H,
    )

    document_padding_mask = torch.ones(
        B,
        D,
        L_doc,
        dtype=torch.bool,
    )

    output = head(
        usefulness_score,
        retrieval_decision,
        encoder_hidden_states,
        document_padding_mask,
        TSRTChosenDocumentCache(),
        DummyDocumentCache(),
        retrieve_top_k=3,
        usefulness_threshold=0.2,
    )

    print("\n=== TopK + Threshold ===")
    print(output.retrieval_memory.shape)
    print(output.encoder_hidden_states.shape)

    assert output.retrieval_memory.shape[2] <= 3


def test_document_padding():

    head = TSRTRetrievalMemoryHead()

    B = 2
    D = 5
    L_doc = 8
    H = 16

    usefulness_score = torch.rand(
        B,
        1,
        D,
    )

    retrieval_decision = torch.ones(
        B,
        1,
        1,
    )

    encoder_hidden_states = torch.randn(
        B,
        D,
        L_doc,
        H,
    )

    document_padding_mask = torch.tensor(
        [
            [
                [1,1,1,1,0,0,0,0],
                [1,1,1,1,1,0,0,0],
                [0,0,0,0,0,0,0,0],
                [1,1,1,0,0,0,0,0],
                [1,1,1,1,1,1,1,1],
            ],
            [
                [1,1,1,1,1,1,0,0],
                [0,0,0,0,0,0,0,0],
                [1,1,1,1,0,0,0,0],
                [1,1,0,0,0,0,0,0],
                [1,1,1,1,1,1,1,0],
            ],
        ],
        dtype=torch.bool,
    )

    output = head(
        usefulness_score,
        retrieval_decision,
        encoder_hidden_states,
        document_padding_mask,
        TSRTChosenDocumentCache(),
        DummyDocumentCache(),
        retrieve_top_k=5,
    )

    print("\n=== Padding ===")
    print(output.retrieval_memory.shape)
    print(output.encoder_hidden_states.shape)
    print(output.document_padding_mask.shape)

    assert output.encoder_hidden_states.shape[2] <= L_doc


def test_all_documents_are_padding():
    """
    Every document is padding.

    Should return None.
    """

    head = TSRTRetrievalMemoryHead()

    B = 2
    D = 5
    L_doc = 8
    H = 16

    usefulness_score = torch.rand(
        B,
        1,
        D,
    )

    retrieval_decision = torch.ones(
        B,
        1,
        1,
    )

    encoder_hidden_states = torch.randn(
        B,
        D,
        L_doc,
        H,
    )

    document_padding_mask = torch.zeros(
        B,
        D,
        L_doc,
        dtype=torch.bool,
    )

    output = head(
        usefulness_score,
        retrieval_decision,
        encoder_hidden_states,
        document_padding_mask,
        TSRTChosenDocumentCache(),
        DummyDocumentCache(),
        retrieve_top_k=3,
    )

    print("\n=== Empty Documents ===")

    print(output.retrieval_memory)
    print(output.encoder_hidden_states)
    print(output.document_padding_mask)

    assert output.retrieval_memory is None
    assert output.encoder_hidden_states is None
    assert output.document_padding_mask is None


def test_mixed_batch():

    head = TSRTRetrievalMemoryHead()

    B = 2
    D = 5
    L_doc = 8
    H = 16

    usefulness_score = torch.rand(
        B,
        1,
        D,
    )

    retrieval_decision = torch.ones(
        B,
        1,
        1,
    )

    encoder_hidden_states = torch.randn(
        B,
        D,
        L_doc,
        H,
    )

    document_padding_mask = torch.ones(
        B,
        D,
        L_doc,
        dtype=torch.bool,
    )

    document_padding_mask[0] = 0

    output = head(
        usefulness_score,
        retrieval_decision,
        encoder_hidden_states,
        document_padding_mask,
        TSRTChosenDocumentCache(),
        DummyDocumentCache(),
        retrieve_top_k=2,
    )

    print("\n=== Mixed Batch ===")
    print(output.retrieval_memory.shape)
    print(output.encoder_hidden_states.shape)
    print(output.document_padding_mask.shape)

    assert output.retrieval_memory.shape[0] == B


if __name__ == "__main__":

    print("=" * 80)
    print("Running Padding / Threshold Tests")
    print("=" * 80)

    test_threshold_filter()
    test_topk_threshold()
    test_document_padding()
    test_all_documents_are_padding()
    test_mixed_batch()

    print("\n" + "=" * 80)
    print("✅ All Padding Tests Passed!")
    print("=" * 80)