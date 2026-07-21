import torch

from models.tsrt.modeling_tsrt import TSRTRetrievalMemoryHead
from models.tsrt.cache_utils import TSRTChosenDocumentCache

torch.manual_seed(42)


def title(name):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)


################################################################################
# Helper
################################################################################

def build_encoder_with_padding():

    """
    B=2

    sample0

        doc0 : len=5
        doc1 : len=3
        doc2 : padding doc
        doc3 : padding doc

    sample1

        doc0 : len=4
        doc1 : len=2
        doc2 : len=1
        doc3 : padding doc
    """

    B = 2
    D = 4
    L = 5
    H = 4

    encoder = torch.zeros(
        B,
        D,
        L,
        H,
    )

    for b in range(B):
        for d in range(D):
            encoder[b, d].fill_(d + 1)

    mask = torch.zeros(
        B,
        D,
        L,
        dtype=torch.bool,
    )

    ####################################################
    # sample0
    ####################################################

    mask[0,0,:5] = True
    mask[0,1,:3] = True

    ####################################################
    # sample1
    ####################################################

    mask[1,0,:4] = True
    mask[1,1,:2] = True
    mask[1,2,:1] = True

    return encoder, mask


################################################################################
# Document Padding
################################################################################

def test_document_padding():

    title("DOCUMENT PADDING")

    head = TSRTRetrievalMemoryHead()

    cache = TSRTChosenDocumentCache()

    usefulness = torch.tensor(
        [
            [[0.9,0.8,0.7,0.6]],
            [[0.5,0.4,0.3,0.2]],
        ]
    )

    decision = torch.ones(
        2,
        1,
        1,
    )

    encoder, padding = build_encoder_with_padding()

    out = head(
        usefulness,
        decision,
        encoder,
        padding,
        cache,
        top_k=4,
    )

    print("encoder")

    print(out.encoder_hidden_states.shape)

    print()

    print("padding")

    print(out.document_padding_mask)

    print()

    print("retrieval")

    print(out.retrieval_memory)

    #
    # sample0 chỉ còn 2 document
    #

    assert torch.all(
        out.retrieval_memory[0,2:] == 0
    )

    #
    # sample1 còn 3 document
    #

    assert torch.all(
        out.retrieval_memory[1,3:] == 0
    )

    print("PASS")


################################################################################
# Token Padding
################################################################################

def test_token_padding():

    title("TOKEN PADDING")

    head = TSRTRetrievalMemoryHead()

    cache = TSRTChosenDocumentCache()

    usefulness = torch.tensor(
        [
            [[0.9,0.8,0.7,0.6]],
            [[0.5,0.4,0.3,0.2]],
        ]
    )

    decision = torch.ones(
        2,
        1,
        1,
    )

    encoder, padding = build_encoder_with_padding()

    out = head(
        usefulness,
        decision,
        encoder,
        padding,
        cache,
        top_k=4,
    )

    print(out.document_padding_mask)

    #
    # sample0
    #

    assert out.document_padding_mask[
        0,
        1,
        3:
    ].sum() == 0

    #
    # sample1
    #

    assert out.document_padding_mask[
        1,
        2,
        1:
    ].sum() == 0

    print("PASS")


################################################################################
# Empty Documents
################################################################################

def test_empty_document():

    title("EMPTY DOCUMENT")

    head = TSRTRetrievalMemoryHead()

    cache = TSRTChosenDocumentCache()

    usefulness = torch.tensor(
        [
            [[0.9,0.8,0.7,0.6]]
        ]
    )

    decision = torch.ones(
        1,
        1,
        1,
    )

    encoder = torch.randn(
        1,
        4,
        5,
        8,
    )

    padding = torch.zeros(
        1,
        4,
        5,
        dtype=torch.bool,
    )

    out = head(
        usefulness,
        decision,
        encoder,
        padding,
        cache,
        top_k=4,
    )

    print(out.encoder_hidden_states)
    print(out.document_padding_mask)
    print(out.retrieval_memory)

    assert out.encoder_hidden_states is None
    assert out.document_padding_mask is None
    assert out.retrieval_memory is None

    print("PASS")

################################################################################
# Batch: Empty sample + Normal sample
################################################################################

def test_batch_empty_sample():

    title("BATCH EMPTY SAMPLE")

    head = TSRTRetrievalMemoryHead()
    cache = TSRTChosenDocumentCache()

    encoder, padding = build_encoder_with_padding()

    #
    # sample0:
    #   doc0 valid
    #   doc1 valid
    #
    # sample1:
    #   toàn bộ document padding
    #
    padding[1] = False

    usefulness = torch.tensor(
        [
            [[0.9,0.8,0.7,0.6]],
            [[0.5,0.4,0.3,0.2]],
        ]
    )

    decision = torch.ones(
        2,
        1,
        1,
    )

    out = head(
        usefulness,
        decision,
        encoder,
        padding,
        cache,
        top_k=4,
    )

    print(out.retrieval_memory)

    #
    # sample0 còn document
    #

    assert out.retrieval_memory[0].sum() > 0

    #
    # sample1 được pad thành toàn zero
    #

    assert out.retrieval_memory[1].sum() == 0

    assert out.document_padding_mask[1].sum() == 0

    print("PASS")


################################################################################
# Pad D
################################################################################

def test_pad_document_dimension():

    title("PAD DOCUMENT DIMENSION")

    head = TSRTRetrievalMemoryHead()

    cache = TSRTChosenDocumentCache()

    encoder, padding = build_encoder_with_padding()

    usefulness = torch.tensor(
        [
            [[0.9,0.8,0.7,0.6]],
            [[0.95,0.94,0.2,0.1]],
        ]
    )

    decision = torch.ones(
        2,
        1,
        1,
    )

    out = head(
        usefulness,
        decision,
        encoder,
        padding,
        cache,
        top_k=4,
    )

    #
    # sample0 có 2 doc
    # sample1 có 3 doc
    #
    # ==> D_max = 3
    #

    print(out.encoder_hidden_states.shape)

    assert out.encoder_hidden_states.shape[1] == 3

    #
    # doc cuối sample0 là padding
    #

    assert torch.all(
        out.encoder_hidden_states[
            0,
            2,
        ] == 0
    )

    print("PASS")


################################################################################
# Cache After Padding
################################################################################

def test_cache_after_padding():

    title("CACHE AFTER PADDING")

    head = TSRTRetrievalMemoryHead()

    cache = TSRTChosenDocumentCache()

    encoder, padding = build_encoder_with_padding()

    usefulness = torch.tensor(
        [
            [[0.9,0.8,0.7,0.6]],
            [[0.95,0.94,0.2,0.1]],
        ]
    )

    decision = torch.ones(
        2,
        1,
        1,
    )

    first = head(
        usefulness,
        decision,
        encoder,
        padding,
        cache,
        top_k=4,
    )

    decision.zero_()

    second = head(
        usefulness,
        decision,
        encoder,
        padding,
        cache,
        top_k=4,
    )

    assert torch.equal(
        first.encoder_hidden_states,
        second.encoder_hidden_states,
    )

    assert torch.equal(
        first.document_padding_mask,
        second.document_padding_mask,
    )

    assert torch.equal(
        first.retrieval_memory,
        second.retrieval_memory,
    )

    print("PASS")


################################################################################
# All Empty Batch
################################################################################

def test_all_empty_batch():

    title("ALL EMPTY BATCH")

    head = TSRTRetrievalMemoryHead()

    cache = TSRTChosenDocumentCache()

    encoder = torch.randn(
        2,
        4,
        5,
        8,
    )

    padding = torch.zeros(
        2,
        4,
        5,
        dtype=torch.bool,
    )

    usefulness = torch.rand(
        2,
        1,
        4,
    )

    decision = torch.ones(
        2,
        1,
        1,
    )

    out = head(
        usefulness,
        decision,
        encoder,
        padding,
        cache,
        top_k=4,
    )

    assert out.encoder_hidden_states is None
    assert out.document_padding_mask is None
    assert out.retrieval_memory is None

    print("PASS")

################################################################################
# Threshold removes all docs for one sample
################################################################################

def test_threshold_empty_one_sample():

    title("THRESHOLD EMPTY ONE SAMPLE")

    head = TSRTRetrievalMemoryHead()
    cache = TSRTChosenDocumentCache()

    encoder, padding = build_encoder_with_padding()

    usefulness = torch.tensor(
        [
            # sample0 -> tất cả đều bị threshold loại
            [[0.10, 0.20, 0.30, 0.40]],

            # sample1 -> vẫn còn 2 document
            [[0.95, 0.96, 0.20, 0.10]],
        ],
        dtype=torch.float,
    )

    decision = torch.ones(
        2,
        1,
        1,
    )

    out = head(
        usefulness,
        decision,
        encoder,
        padding,
        cache,
        usefulness_threshold=0.9,
    )

    print("retrieval memory")

    print(out.retrieval_memory)

    print()

    print("padding mask")

    print(out.document_padding_mask)

    ########################################################
    # sample0
    ########################################################

    assert out.retrieval_memory[0].sum() == 0

    assert out.document_padding_mask[0].sum() == 0

    assert torch.all(
        out.encoder_hidden_states[0] == 0
    )

    ########################################################
    # sample1
    ########################################################

    assert torch.allclose(
        out.retrieval_memory[1],
        torch.tensor([0.95, 0.96]),
    )

    assert out.document_padding_mask[1].sum() > 0

    print("PASS")

################################################################################
# Main
################################################################################

def main():

    test_document_padding()

    test_token_padding()

    test_empty_document()

    test_batch_empty_sample()

    test_pad_document_dimension()

    test_cache_after_padding()

    test_all_empty_batch()

    test_threshold_empty_one_sample()

    print()
    print("=" * 80)
    print("ALL PADDING TESTS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()