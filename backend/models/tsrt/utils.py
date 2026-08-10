import torch


def prepare_document_attention_mask(
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Convert document token attention mask for Transformer attention.

    Args:
        attention_mask:
            Tensor of shape (B, D, L)

            Values:
                1 -> valid token
                0 -> padding token

    Returns:
        Tensor of shape (B*D, 1, 1, L)

        Values:
            0    -> attendable
            -inf -> masked
    """

    B, D, L = attention_mask.shape

    # (B, D, L) -> (B*D, L)
    attention_mask = attention_mask.view(
        B * D,
        L,
    )

    # Convert:
    # 1 -> 0
    # 0 -> -inf
    attention_mask = (
        1.0 - attention_mask
    ) * torch.finfo(torch.float32).min

    # (B*D, L) -> (B*D, 1, 1, L)
    attention_mask = attention_mask[:, None, None, :]

    return attention_mask

def prepare_cross_attention_mask(
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Prepare cross-attention mask for document tokens.

    Args:
        attention_mask:
            Tensor of shape (B, D, L)

            Values:
                1 -> valid token
                0 -> padding token

    Returns:
        Tensor of shape (B, 1, 1, D*L)

        Values:
                0    -> attendable
                -inf -> masked
    """

    B, D, L = attention_mask.shape

    # (B, D, L) -> (B, D*L)
    attention_mask = attention_mask.reshape(
        B,
        D * L,
    )

    # Convert:
    # 1 -> 0
    # 0 -> -inf
    attention_mask = (
        1.0 - attention_mask
    ) * torch.finfo(torch.float32).min

    # (B, D*L) -> (B, 1, 1, D*L)
    attention_mask = attention_mask[:, None, None, :]

    return attention_mask

def prepare_projection_mask(
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Prepare attention mask for projection weighting.

    Args:
        attention_mask:
            Tensor of shape (B, L)
            or
            (B, D, L)

            Values:
                1 -> valid token
                0 -> padding token

    Returns:
        Tensor with the same shape as input.

        Values:
            0    -> valid token
            -inf -> masked token
    """

    return (
        1.0 - attention_mask
    ) * torch.finfo(torch.float32).min

import torch
import torch.nn.functional as F


def compute_retrieval_decision_loss(
    retrieval_decision_logits: torch.Tensor,   # (B, L), raw logits
    retrieval_decision_scores: torch.Tensor,   # (B, L), sigmoid scores in [0, 1]
    retrieval_decision_labels: torch.Tensor,   # (B, L), {0, 1, -1}
    num_items_in_batch: int | None = None,
) -> tuple[
    torch.Tensor,  # training loss
    torch.Tensor,  # logging loss
    torch.Tensor,  # decision_predict
    torch.Tensor,  # non_decision_predict
]:
    """
    Binary cross entropy loss for retrieval decision.

    Ignore:
        - label == -1

    Returns:
        training_loss:
            Loss used for backward.

        logging_loss:
            Mean BCE loss (independent of num_items_in_batch).

        decision_predict:
            Mean sigmoid score on positive labels.

        non_decision_predict:
            Mean sigmoid score on negative labels.
    """

    valid_mask = retrieval_decision_labels != -1

    if not valid_mask.any():
        zero = retrieval_decision_logits.new_zeros(())
        return zero, zero, zero, zero

    logits = retrieval_decision_logits[valid_mask]
    scores = retrieval_decision_scores[valid_mask]
    targets = retrieval_decision_labels[valid_mask].float()

    # --------------------------------------------------------
    # Logging loss (always mean)
    # --------------------------------------------------------

    logging_loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="mean",
    )

    # --------------------------------------------------------
    # Training loss
    # --------------------------------------------------------

    if num_items_in_batch is None:
        loss = logging_loss
    else:
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="sum",
        )
        loss = loss / num_items_in_batch

    # --------------------------------------------------------
    # Logging statistics
    # --------------------------------------------------------

    positive_mask = targets == 1
    negative_mask = targets == 0

    if positive_mask.any():
        decision_predict = scores[positive_mask].mean()
    else:
        decision_predict = retrieval_decision_scores.new_zeros(())

    if negative_mask.any():
        non_decision_predict = scores[negative_mask].mean()
    else:
        non_decision_predict = retrieval_decision_scores.new_zeros(())

    return (
        loss,
        logging_loss,
        decision_predict,
        non_decision_predict,
    )

import torch


def compute_retrieval_ranking_loss(
    usefulness_scores: torch.Tensor,      # (B, L, D), scores in [-1, 1]
    usefulness_score_matrix: torch.Tensor,# (B, L, D), {1, 0, -1}
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Multi-positive InfoNCE ranking loss.

    Each positive document is treated as one positive sample and
    competes against all negative documents of the same (B, L).

    Ignore:
        - label == -1
        - (B, L) positions without positive or without negative docs
    """

    pos_mask = usefulness_score_matrix == 1
    neg_mask = usefulness_score_matrix == 0

    # (B, L)
    has_pos = pos_mask.any(dim=-1)
    has_neg = neg_mask.any(dim=-1)
    valid_query = has_pos & has_neg

    if not valid_query.any():
        # print("Have a not valid query")
        # print("usefulness score matrix: ", usefulness_score_matrix)
        return usefulness_scores.sum() * 0.0

    # Keep only valid (B, L)
    scores = usefulness_scores[valid_query]          # (Q, D)
    pos_mask = pos_mask[valid_query]                 # (Q, D)
    neg_mask = neg_mask[valid_query]                 # (Q, D)

    scores = scores / temperature

    losses = []

    for query_scores, query_pos_mask, query_neg_mask in zip(scores, pos_mask, neg_mask):

        pos_scores = query_scores[query_pos_mask]    # (P,)
        neg_scores = query_scores[query_neg_mask]    # (N,)

        logits = torch.cat([
            pos_scores[:, None],
            neg_scores.expand(pos_scores.size(0), -1),
        ], dim=1)

        positive = logits[:, 0]

        loss = -(positive - torch.logsumexp(logits, dim=1))

        losses.append(loss.mean())

    return torch.stack(losses).mean()


import torch
import torch.nn.functional as F


def compute_retrieval_scoring_loss(
    usefulness_scores: torch.Tensor,        # (B, L, D), cosine scores [-1,1]
    usefulness_score_matrix: torch.Tensor,  # (B, L, D), {1,0,-1}
    margin: float = -0.8,
) -> torch.Tensor:
    """
    Cosine similarity retrieval loss.

    Positive documents:
        Pull cosine similarity closer to 1.

    Negative documents:
        Push cosine similarity below margin.

    Ignore:
        - label == -1
    """

    pos_mask = usefulness_score_matrix == 1
    neg_mask = usefulness_score_matrix == 0

    losses = []

    # --------------------------------------------------------
    # Positive documents
    # Maximize cosine similarity
    #
    # loss = 1 - cosine
    # --------------------------------------------------------
    if pos_mask.any():
        positive_scores = usefulness_scores[pos_mask]

        positive_loss = 1.0 - positive_scores

        losses.append(
            positive_loss.mean()
        )

    # --------------------------------------------------------
    # Negative documents
    # Minimize cosine similarity
    #
    # loss = max(cosine - margin, 0)
    # --------------------------------------------------------
    if neg_mask.any():
        negative_scores = usefulness_scores[neg_mask]

        negative_loss = F.relu(
            negative_scores - margin
        )

        losses.append(
            negative_loss.mean()
        )

    # Không có positive hoặc negative hợp lệ
    if len(losses) == 0:
        return usefulness_scores.new_zeros(())

    return torch.stack(losses).mean()


import torch

def compute_positive_negative_scores(
    usefulness_scores: torch.Tensor,        # (B, L, D), scores in [-1, 1]
    usefulness_score_matrix: torch.Tensor,  # (B, L, D), {1, 0, -1}
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean usefulness score for positive and negative documents.

    Ignore:
        - label == -1

    Returns:
        positive_score:
            Scalar tensor containing the mean score of all positive documents.

        negative_score:
            Scalar tensor containing the mean score of all negative documents.
    """

    pos_mask = usefulness_score_matrix == 1
    neg_mask = usefulness_score_matrix == 0

    if pos_mask.any():
        positive_score = usefulness_scores[pos_mask].mean()
    else:
        positive_score = usefulness_scores.new_zeros(())

    if neg_mask.any():
        negative_score = usefulness_scores[neg_mask].mean()
    else:
        negative_score = usefulness_scores.new_zeros(())

    return positive_score, negative_score