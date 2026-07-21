import torch

from models.tsrt.modeling_tsrt import TSRTRetrievalMemoryHead
from models.tsrt.cache_utils import TSRTChosenDocumentCache

torch.manual_seed(42)


def title(name):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)


def build_encoder(B=1, D=4, L_doc=5, H=4):
    """
    Mỗi document có giá trị khác nhau để dễ kiểm tra.
    Doc i sẽ toàn giá trị i.
    """
    encoder = torch.zeros(B, D, L_doc, H)

    for b in range(B):
        for d in range(D):
            encoder[b, d].fill_(float(d))

    mask = torch.ones(
        B,
        D,
        L_doc,
        dtype=torch.bool,
    )

    return encoder, mask


################################################################################
# Top-k
################################################################################


def test_top_k():

    title("TOP K")

    head = TSRTRetrievalMemoryHead()
    cache = TSRTChosenDocumentCache()

    usefulness = torch.tensor(
        [[[0.2, 0.9, 0.4, 0.8]]],
        dtype=torch.float,
    )

    decision = torch.ones(1, 1, 1)

    encoder, padding = build_encoder()

    out = head(
        usefulness,
        decision,
        encoder,
        padding,
        cache,
        top_k=2,
    )

    print("retrieval memory")

    print(out.retrieval_memory)

    print()

    print("chosen docs")

    print(out.encoder_hidden_states[:, :, 0, 0])

    expected = torch.tensor([[0.9, 0.8]])

    assert torch.allclose(
        out.retrieval_memory,
        expected,
    )

    expected_doc = torch.tensor([[1.0, 3.0]])

    assert torch.allclose(
        out.encoder_hidden_states[:, :, 0, 0],
        expected_doc,
    )

    print("PASS")


################################################################################
# Threshold
################################################################################


def test_threshold():

    title("THRESHOLD")

    head = TSRTRetrievalMemoryHead()
    cache = TSRTChosenDocumentCache()

    usefulness = torch.tensor(
        [[[0.2, 0.9, 0.4, 0.8]]],
        dtype=torch.float,
    )

    decision = torch.ones(1, 1, 1)

    encoder, padding = build_encoder()

    out = head(
        usefulness,
        decision,
        encoder,
        padding,
        cache,
        usefulness_threshold=0.5,
    )

    print(out.retrieval_memory)

    print(out.encoder_hidden_states[:, :, 0, 0])

    expected = torch.tensor([[0.9, 0.8]])

    assert torch.allclose(
        out.retrieval_memory,
        expected,
    )

    expected_doc = torch.tensor([[1.0, 3.0]])

    assert torch.allclose(
        out.encoder_hidden_states[:, :, 0, 0],
        expected_doc,
    )

    print("PASS")


################################################################################
# Top-k + Threshold
################################################################################


def test_top_k_threshold():

    title("TOP K + THRESHOLD")

    head = TSRTRetrievalMemoryHead()
    cache = TSRTChosenDocumentCache()

    usefulness = torch.tensor(
        [[[0.2, 0.9, 0.85, 0.8]]],
        dtype=torch.float,
    )

    decision = torch.ones(1, 1, 1)

    encoder, padding = build_encoder()

    out = head(
        usefulness,
        decision,
        encoder,
        padding,
        cache,
        top_k=2,
        usefulness_threshold=0.82,
    )

    print(out.retrieval_memory)

    print(out.encoder_hidden_states[:, :, 0, 0])

    expected = torch.tensor([[0.9, 0.85]])

    assert torch.allclose(
        out.retrieval_memory,
        expected,
    )

    expected_doc = torch.tensor([[1.0, 2.0]])

    assert torch.allclose(
        out.encoder_hidden_states[:, :, 0, 0],
        expected_doc,
    )

    print("PASS")

################################################################################
# Batch: Mixed Retrieval
################################################################################


def test_batch_mixed_retrieval():

    title("BATCH MIXED RETRIEVAL")

    head = TSRTRetrievalMemoryHead()
    cache = TSRTChosenDocumentCache()

    B = 3
    D = 4

    usefulness = torch.tensor(
        [
            [[0.1, 0.9, 0.3, 0.8]],   # sample0 -> retrieve
            [[0.4, 0.5, 0.6, 0.7]],   # sample1 -> no retrieve
            [[0.8, 0.1, 0.2, 0.9]],   # sample2 -> retrieve
        ],
        dtype=torch.float,
    )

    decision = torch.tensor(
        [
            [[1.]],
            [[0.]],
            [[1.]],
        ]
    )

    encoder, padding = build_encoder(
        B=B,
        D=D,
    )

    out = head(
        usefulness,
        decision,
        encoder,
        padding,
        cache,
        top_k=2,
    )

    print(out.retrieval_memory)

    # sample1 không retrieve và cache rỗng
    assert out.retrieval_memory[1].sum() == 0

    # sample0
    assert torch.allclose(
        out.retrieval_memory[0],
        torch.tensor([0.9, 0.8]),
    )

    # sample2
    assert torch.allclose(
        out.retrieval_memory[2],
        torch.tensor([0.9, 0.8]),
    )

    print("PASS")


################################################################################
# Batch Cache Reuse
################################################################################


def test_batch_cache_reuse():

    title("BATCH CACHE REUSE")

    head = TSRTRetrievalMemoryHead()
    cache = TSRTChosenDocumentCache()

    B = 2
    D = 4

    usefulness = torch.tensor(
        [
            [[0.1, 0.9, 0.3, 0.8]],
            [[0.8, 0.7, 0.2, 0.1]],
        ],
        dtype=torch.float,
    )

    decision = torch.ones(
        B,
        1,
        1,
    )

    encoder, padding = build_encoder(
        B=B,
        D=D,
    )

    first = head(
        usefulness,
        decision,
        encoder,
        padding,
        cache,
        top_k=2,
    )

    ########################################################

    decision.zero_()

    second = head(
        usefulness,
        decision,
        encoder,
        padding,
        cache,
        top_k=2,
    )

    assert torch.equal(
        first.encoder_hidden_states,
        second.encoder_hidden_states,
    )

    assert torch.equal(
        first.retrieval_memory,
        second.retrieval_memory,
    )

    assert torch.equal(
        first.document_padding_mask,
        second.document_padding_mask,
    )

    print("PASS")


################################################################################
# Batch Cache Overwrite
################################################################################


def test_batch_cache_overwrite():

    title("BATCH CACHE OVERWRITE")

    head = TSRTRetrievalMemoryHead()
    cache = TSRTChosenDocumentCache()

    encoder, padding = build_encoder(
        B=2,
        D=4,
    )

    usefulness1 = torch.tensor(
        [
            [[0.1, 0.9, 0.3, 0.8]],
            [[0.8, 0.7, 0.2, 0.1]],
        ]
    )

    usefulness2 = torch.tensor(
        [
            [[0.9, 0.2, 0.8, 0.1]],
            [[0.3, 0.4, 0.95, 0.96]],
        ]
    )

    decision = torch.ones(
        2,
        1,
        1,
    )

    first = head(
        usefulness1,
        decision,
        encoder,
        padding,
        cache,
        top_k=2,
    )

    second = head(
        usefulness2,
        decision,
        encoder,
        padding,
        cache,
        top_k=2,
    )

    assert not torch.equal(
        first.retrieval_memory,
        second.retrieval_memory,
    )

    print(second.retrieval_memory)

    print("PASS")


################################################################################
# Batch + TopK + Threshold
################################################################################


def test_batch_topk_threshold():

    title("BATCH TOPK + THRESHOLD")

    head = TSRTRetrievalMemoryHead()
    cache = TSRTChosenDocumentCache()

    usefulness = torch.tensor(
        [
            [[0.2, 0.91, 0.87, 0.5]],
            [[0.95, 0.2, 0.1, 0.96]],
        ]
    )

    decision = torch.ones(
        2,
        1,
        1,
    )

    encoder, padding = build_encoder(
        B=2,
        D=4,
    )

    out = head(
        usefulness,
        decision,
        encoder,
        padding,
        cache,
        top_k=2,
        usefulness_threshold=0.85,
    )

    print(out.retrieval_memory)

    assert torch.allclose(
        out.retrieval_memory[0],
        torch.tensor([0.91, 0.87]),
    )

    assert torch.allclose(
        out.retrieval_memory[1],
        torch.tensor([0.96, 0.95]),
    )

    print("PASS")


################################################################################


def main():

    test_top_k()

    test_threshold()

    test_top_k_threshold()

    test_batch_mixed_retrieval()

    test_batch_cache_reuse()

    test_batch_cache_overwrite()

    test_batch_topk_threshold()

    print("\n")
    print("=" * 80)
    print("ALL HARD TESTS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()