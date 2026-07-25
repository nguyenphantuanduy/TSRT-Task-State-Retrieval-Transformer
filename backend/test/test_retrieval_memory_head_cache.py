import torch

from models.tsrt.cache_utils import TSRTChosenDocumentCache
from models.tsrt.modeling_tsrt import TSRTRetrievalMemoryHead


class DummyDocumentCache:
    """
    Dummy document cache.
    Chỉ dùng để kiểm tra reset_all() có được gọi hay không.
    """

    def __init__(self):
        self.reset_called = False

    def reset_all(self):
        self.reset_called = True


def build_inputs(
    B=2,
    D=5,
    L_doc=8,
    H=16,
):
    usefulness_score = torch.rand(B, 1, D)

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

    return (
        usefulness_score,
        encoder_hidden_states,
        document_padding_mask,
    )


def test_cache_update():
    """
    decision=1

    cache phải được update.
    """

    B = 2
    D = 5

    cache = TSRTChosenDocumentCache()
    document_cache = DummyDocumentCache()

    head = TSRTRetrievalMemoryHead()

    usefulness_score, encoder_hidden_states, padding_mask = build_inputs(
        B=B,
        D=D,
    )

    retrieval_decision = torch.ones(
        B,
        1,
        1,
    )

    output = head(
        usefulness_score,
        retrieval_decision,
        encoder_hidden_states,
        padding_mask,
        cache,
        document_cache,
        retrieve_top_k=3,
    )

    assert cache.has_cache()

    chosen_document, cached_padding, cached_memory = cache.get()

    assert chosen_document is not None
    assert cached_padding is not None
    assert cached_memory is not None

    assert output.retrieval_memory.shape[0] == B
    assert output.retrieval_memory.shape[1] == 1

    print("\n=== Cache Update ===")
    print("retrieval_memory :", output.retrieval_memory.shape)
    print("chosen_document  :", chosen_document.shape)
    print("padding_mask     :", cached_padding.shape)

    assert document_cache.reset_called


def test_cache_reuse():
    """
    decision=1
        ↓
    decision=0

    cache phải được reuse.
    """

    cache = TSRTChosenDocumentCache()

    head = TSRTRetrievalMemoryHead()

    usefulness_score, encoder_hidden_states, padding_mask = build_inputs()

    retrieval = torch.ones(
        2,
        1,
        1,
    )

    head(
        usefulness_score,
        retrieval,
        encoder_hidden_states,
        padding_mask,
        cache,
        DummyDocumentCache(),
        retrieve_top_k=2,
    )

    cached_doc, _, _ = cache.get()

    retrieval = torch.zeros(
        2,
        1,
        1,
    )

    output = head(
        usefulness_score,
        retrieval,
        encoder_hidden_states,
        padding_mask,
        cache,
        DummyDocumentCache(),
        retrieve_top_k=2,
    )

    assert torch.equal(
        output.encoder_hidden_states,
        cached_doc,
    )

    print("\n=== Cache Reuse ===")
    print(output.encoder_hidden_states.shape)


def test_cache_overwrite():
    """
    decision=1

    gọi thêm lần nữa

    cache phải bị overwrite.
    """

    cache = TSRTChosenDocumentCache()

    head = TSRTRetrievalMemoryHead()

    usefulness_score, encoder_hidden_states, padding_mask = build_inputs()

    retrieval = torch.ones(
        2,
        1,
        1,
    )

    head(
        usefulness_score,
        retrieval,
        encoder_hidden_states,
        padding_mask,
        cache,
        DummyDocumentCache(),
        retrieve_top_k=1,
    )

    first_doc, _, _ = cache.get()

    usefulness_score2, encoder_hidden_states2, padding_mask2 = build_inputs()

    head(
        usefulness_score2,
        retrieval,
        encoder_hidden_states2,
        padding_mask2,
        cache,
        DummyDocumentCache(),
        retrieve_top_k=1,
    )

    second_doc, _, _ = cache.get()

    assert not torch.equal(
        first_doc,
        second_doc,
    )

    print("\n=== Cache Overwrite ===")
    print(second_doc.shape)


def test_multiple_forward_calls():
    """
    Mô phỏng autoregressive decoding.

    1
    ↓
    0
    ↓
    0
    ↓
    1
    ↓
    0
    """

    cache = TSRTChosenDocumentCache()

    head = TSRTRetrievalMemoryHead()

    document_cache = DummyDocumentCache()

    decisions = [
        1,
        0,
        0,
        1,
        0,
    ]

    for step, decision in enumerate(decisions):

        usefulness_score, encoder_hidden_states, padding_mask = build_inputs()

        retrieval = torch.full(
            (2, 1, 1),
            float(decision),
        )

        output = head(
            usefulness_score,
            retrieval,
            encoder_hidden_states,
            padding_mask,
            cache,
            document_cache,
            retrieve_top_k=2,
        )

        print(
            f"\nStep {step+1}"
        )

        if output.retrieval_memory is not None:
            print(
                "retrieval_memory :",
                output.retrieval_memory.shape,
            )

            print(
                "encoder_hidden_states :",
                output.encoder_hidden_states.shape,
            )

            print(
                "padding_mask :",
                output.document_padding_mask.shape,
            )

            assert output.retrieval_memory.shape[1] == 1

        assert cache.has_cache()


if __name__ == "__main__":

    print("=" * 80)
    print("Running Incremental Retrieval Tests")
    print("=" * 80)

    test_cache_update()
    test_cache_reuse()
    test_cache_overwrite()
    test_multiple_forward_calls()

    print("\n" + "=" * 80)
    print("✅ All Incremental Retrieval Tests Passed!")
    print("=" * 80)