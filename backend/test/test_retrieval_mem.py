import torch

from models.tsrt.modeling_tsrt import TSRTRetrievalMemoryHead
from models.tsrt.cache_utils import TSRTChosenDocumentCache, TSRTDocumentCache

torch.manual_seed(42)


def title(name):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)


def build_inputs(
    B=2,
    D=4,
    L_doc=5,
    H=8,
):
    usefulness_score = torch.rand(B, 1, D)

    retrieval_decision = torch.ones(B, 1, 1)

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
        retrieval_decision,
        encoder_hidden_states,
        document_padding_mask,
    )


################################################################################
# Case 1
################################################################################

def test_first_retrieval():

    title("FIRST RETRIEVAL")

    (
        usefulness_score,
        retrieval_decision,
        encoder_hidden_states,
        document_padding_mask,
    ) = build_inputs()

    cache = TSRTChosenDocumentCache()

    head = TSRTRetrievalMemoryHead()

    output = head(
        usefulness_score,
        retrieval_decision,
        encoder_hidden_states,
        document_padding_mask,
        cache=cache,
        document_cache=TSRTDocumentCache(),
        retrieve_top_k=2,
    )

    print(output.encoder_hidden_states.shape)
    print(output.document_padding_mask.shape)
    print(output.retrieval_memory.shape)

    print()

    print("Cache:", cache.has_cache())


################################################################################
# Case 2
################################################################################

def test_no_retrieval_without_cache():

    title("NO RETRIEVAL + EMPTY CACHE")

    (
        usefulness_score,
        retrieval_decision,
        encoder_hidden_states,
        document_padding_mask,
    ) = build_inputs()

    retrieval_decision.zero_()

    cache = TSRTChosenDocumentCache()

    head = TSRTRetrievalMemoryHead()

    output = head(
        usefulness_score,
        retrieval_decision,
        encoder_hidden_states,
        document_padding_mask,
        cache=cache,
        document_cache=TSRTDocumentCache(),
        retrieve_top_k=2,
    )

    print(output.encoder_hidden_states)
    print(output.retrieval_memory)
    print(output.document_padding_mask)


################################################################################
# Case 3
################################################################################

def test_reuse_cache():

    title("REUSE CACHE")

    (
        usefulness_score,
        retrieval_decision,
        encoder_hidden_states,
        document_padding_mask,
    ) = build_inputs()

    cache = TSRTChosenDocumentCache()

    head = TSRTRetrievalMemoryHead()

    output1 = head(
        usefulness_score,
        retrieval_decision,
        encoder_hidden_states,
        document_padding_mask,
        cache=cache,
        document_cache=TSRTDocumentCache(),
        retrieve_top_k=2,
    )

    retrieval_decision.zero_()

    output2 = head(
        usefulness_score,
        retrieval_decision,
        encoder_hidden_states,
        document_padding_mask,
        cache=cache,
        document_cache=TSRTDocumentCache(),
        retrieve_top_k=2,
    )

    print(torch.equal(
        output1.encoder_hidden_states,
        output2.encoder_hidden_states,
    ))

    print(torch.equal(
        output1.retrieval_memory,
        output2.retrieval_memory,
    ))

    print(torch.equal(
        output1.document_padding_mask,
        output2.document_padding_mask,
    ))


################################################################################
# Case 4
################################################################################

def test_overwrite_cache():

    title("OVERWRITE CACHE")

    (
        usefulness_score,
        retrieval_decision,
        encoder_hidden_states,
        document_padding_mask,
    ) = build_inputs()

    cache = TSRTChosenDocumentCache()

    head = TSRTRetrievalMemoryHead()

    output1 = head(
        usefulness_score,
        retrieval_decision,
        encoder_hidden_states,
        document_padding_mask,
        cache=cache,
        document_cache=TSRTDocumentCache(),
        retrieve_top_k=2,
    )

    usefulness_score = torch.rand_like(usefulness_score)

    encoder_hidden_states = torch.randn_like(
        encoder_hidden_states,
    )

    output2 = head(
        usefulness_score,
        retrieval_decision,
        encoder_hidden_states,
        document_padding_mask,
        cache=cache,
        document_cache=TSRTDocumentCache(),
        retrieve_top_k=2,
    )

    print("Cache updated?")

    print(
        not torch.equal(
            output1.encoder_hidden_states,
            output2.encoder_hidden_states,
        )
    )


################################################################################


def main():

    test_first_retrieval()

    test_no_retrieval_without_cache()

    test_reuse_cache()

    test_overwrite_cache()


if __name__ == "__main__":
    main()