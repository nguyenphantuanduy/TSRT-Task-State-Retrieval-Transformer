import torch

from models.tsrt.cache_utils import TSRTChosenDocumentCache
from models.tsrt.modeling_tsrt import TSRTRetrievalMemoryHead


def test_full_sequence_output_shape():
    """
    Full sequence mode.

    Branch:
        retrieve_top_k=None
        usefulness_threshold=None
    """

    B = 2
    L = 5
    D = 4
    L_doc = 8
    H = 16

    head = TSRTRetrievalMemoryHead()

    usefulness_score = torch.rand(B, L, D)

    retrieval_decision = torch.randint(
        0,
        2,
        (B, L, 1),
    ).float()

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
        usefulness_score=usefulness_score,
        retrieval_decision=retrieval_decision,
        encoder_hidden_states=encoder_hidden_states,
        document_padding_mask=document_padding_mask,
        cache=TSRTChosenDocumentCache(),
        document_cache=None,
    )

    assert output.retrieval_memory.shape == (B, L, D)
    assert output.encoder_hidden_states.shape == (B, D, L_doc, H)
    assert output.document_padding_mask.shape == (B, D, L_doc)

    print("\n=== Full Sequence ===")
    print("retrieval_memory      :", output.retrieval_memory.shape)
    print("encoder_hidden_states :", output.encoder_hidden_states.shape)
    print("document_padding_mask :", output.document_padding_mask.shape)


def test_full_sequence_all_zero_decision():
    """
    retrieval_decision = 0 toàn bộ.
    """

    B = 2
    L = 6
    D = 3
    L_doc = 4
    H = 8

    head = TSRTRetrievalMemoryHead()

    usefulness_score = torch.rand(B, L, D)

    retrieval_decision = torch.zeros(
        B,
        L,
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
        usefulness_score=usefulness_score,
        retrieval_decision=retrieval_decision,
        encoder_hidden_states=encoder_hidden_states,
        document_padding_mask=document_padding_mask,
        cache=TSRTChosenDocumentCache(),
        document_cache=None,
    )

    expected = torch.zeros_like(output.retrieval_memory)

    assert output.retrieval_memory.shape == (B, L, D)
    assert torch.allclose(output.retrieval_memory, expected)

    print("\n=== All Zero Decision ===")
    print("retrieval_memory      :", output.retrieval_memory.shape)
    print("encoder_hidden_states :", output.encoder_hidden_states.shape)
    print("document_padding_mask :", output.document_padding_mask.shape)


def test_full_sequence_all_one_decision():
    """
    retrieval_decision = 1 toàn bộ.
    """

    B = 2
    L = 5
    D = 4
    L_doc = 6
    H = 12

    head = TSRTRetrievalMemoryHead()

    usefulness_score = torch.rand(B, L, D)

    retrieval_decision = torch.ones(
        B,
        L,
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
        usefulness_score=usefulness_score,
        retrieval_decision=retrieval_decision,
        encoder_hidden_states=encoder_hidden_states,
        document_padding_mask=document_padding_mask,
        cache=TSRTChosenDocumentCache(),
        document_cache=None,
    )

    assert output.retrieval_memory.shape == (B, L, D)
    assert torch.allclose(
        output.retrieval_memory,
        usefulness_score,
    )

    print("\n=== All One Decision ===")
    print("retrieval_memory      :", output.retrieval_memory.shape)
    print("encoder_hidden_states :", output.encoder_hidden_states.shape)
    print("document_padding_mask :", output.document_padding_mask.shape)


def test_full_sequence_memory_propagation():
    """
    decision:
        1 0 1 0

    expected memory:
        U0
        U0
        U2
        U2
    """

    B = 1
    L = 4
    D = 2
    L_doc = 3
    H = 4

    head = TSRTRetrievalMemoryHead()

    usefulness_score = torch.tensor(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ]
        ]
    )

    retrieval_decision = torch.tensor(
        [
            [
                [1.0],
                [0.0],
                [1.0],
                [0.0],
            ]
        ]
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
        usefulness_score=usefulness_score,
        retrieval_decision=retrieval_decision,
        encoder_hidden_states=encoder_hidden_states,
        document_padding_mask=document_padding_mask,
        cache=TSRTChosenDocumentCache(),
        document_cache=None,
    )

    expected = torch.tensor(
        [
            [
                [1.0, 2.0],
                [1.0, 2.0],
                [5.0, 6.0],
                [5.0, 6.0],
            ]
        ]
    )

    assert torch.allclose(
        output.retrieval_memory,
        expected,
    )

    print("\n=== Memory Propagation ===")
    print("retrieval_memory      :", output.retrieval_memory.shape)
    print("encoder_hidden_states :", output.encoder_hidden_states.shape)
    print("document_padding_mask :", output.document_padding_mask.shape)
    print(output.retrieval_memory)


def test_full_sequence_large_batch():
    """
    Stress test.
    """

    B = 8
    L = 128
    D = 16
    L_doc = 64
    H = 32

    head = TSRTRetrievalMemoryHead()

    usefulness_score = torch.rand(
        B,
        L,
        D,
    )

    retrieval_decision = torch.randint(
        0,
        2,
        (B, L, 1),
    ).float()

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
        usefulness_score=usefulness_score,
        retrieval_decision=retrieval_decision,
        encoder_hidden_states=encoder_hidden_states,
        document_padding_mask=document_padding_mask,
        cache=TSRTChosenDocumentCache(),
        document_cache=None,
    )

    assert output.retrieval_memory.shape == (B, L, D)
    assert output.encoder_hidden_states.shape == (B, D, L_doc, H)
    assert output.document_padding_mask.shape == (B, D, L_doc)

    print("\n=== Large Batch ===")
    print("retrieval_memory      :", output.retrieval_memory.shape)
    print("encoder_hidden_states :", output.encoder_hidden_states.shape)
    print("document_padding_mask :", output.document_padding_mask.shape)


if __name__ == "__main__":
    print("=" * 80)
    print("Running TSRTRetrievalMemoryHead Full Sequence Tests")
    print("=" * 80)

    test_full_sequence_output_shape()
    test_full_sequence_all_zero_decision()
    test_full_sequence_all_one_decision()
    test_full_sequence_memory_propagation()
    test_full_sequence_large_batch()

    print("\n" + "=" * 80)
    print("✅ All Full Sequence Tests Passed!")
    print("=" * 80)