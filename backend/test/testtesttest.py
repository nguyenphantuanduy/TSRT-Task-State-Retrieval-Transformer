import torch

from models.tsrt.modeling_tsrt import (
    TSRTRetrievalMemoryHead,
    RETRIEVAL_DECISION_THRESHOLD,
)


class DummyCache:
    def __init__(self):
        self.chosen_document = None
        self.document_padding_mask = None
        self.retrieval_memory = None

    def update(
        self,
        chosen_document,
        document_padding_mask,
        retrieval_memory,
    ):
        self.chosen_document = chosen_document
        self.document_padding_mask = document_padding_mask
        self.retrieval_memory = retrieval_memory

    def get(self):
        return (
            self.chosen_document,
            self.document_padding_mask,
            self.retrieval_memory,
        )


def print_output(title, output):
    print("=" * 80)
    print(title)

    if output.encoder_hidden_states is None:
        print("No document selected.")
        return

    print("encoder_hidden_states :", output.encoder_hidden_states.shape)
    print("document_padding_mask :", output.document_padding_mask.shape)
    print("retrieval_memory      :", output.retrieval_memory.shape)

    print()
    print("document_padding_mask")
    print(output.document_padding_mask)


def main():

    torch.manual_seed(0)

    B = 2
    L = 5
    D = 4
    L_doc = 45
    H = 16

    head = TSRTRetrievalMemoryHead()

    usefulness_score = torch.rand(B, L, D)

    retrieval_decision = torch.tensor(
        [
            [[0.2], [0.9], [0.3], [0.8], [0.9]],
            [[0.1], [0.4], [0.9], [0.9], [0.2]],
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

    # giả sử document cuối chỉ dài 20 token
    document_padding_mask[:, 3, 20:] = 0

    cache = DummyCache()

    ###################################################################
    # TRAIN
    ###################################################################

    out = head(
        usefulness_score=usefulness_score,
        retrieval_decision=retrieval_decision,
        encoder_hidden_states=encoder_hidden_states,
        document_padding_mask=document_padding_mask,
        cache=cache,
    )

    print_output("TRAIN", out)

    ###################################################################
    # GENERATION
    ###################################################################

    out = head(
        usefulness_score=usefulness_score,
        retrieval_decision=retrieval_decision,
        encoder_hidden_states=encoder_hidden_states,
        document_padding_mask=document_padding_mask,
        cache=cache,
        retrieve_top_k=2,
    )

    print_output("GENERATION", out)


if __name__ == "__main__":
    main()