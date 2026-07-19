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

def multi_positive_info_nce(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    temperature: float = 0.07,
):
    # pos_scores: (P,)
    # neg_scores: (N,)

    logits = torch.cat([
        pos_scores[:, None],                     # (P,1)
        neg_scores.expand(len(pos_scores), -1),  # (P,N)
    ], dim=1)

    logits = logits / temperature

    positive = logits[:, 0]

    loss = -(positive - torch.logsumexp(logits, dim=1))

    return loss.mean()


import torch
import torch.nn.functional as F


def compute_retrieval_decision_loss(
    retrieval_decision_scores: torch.Tensor,    # (B, L), sigmoid scores in [0, 1]
    retrieval_decision_labels: torch.Tensor,    # (B, L), {0, 1, -1}
) -> torch.Tensor:
    # Ignore positions with label = -1
    valid_mask = retrieval_decision_labels != -1

    if not valid_mask.any():
        return retrieval_decision_scores.new_zeros(())

    scores = retrieval_decision_scores[valid_mask]
    targets = retrieval_decision_labels[valid_mask].float()

    loss = F.binary_cross_entropy(
        scores,
        targets,
        reduction="mean",
    )

    return loss


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
        return usefulness_scores.new_zeros(())

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
    usefulness_scores: torch.Tensor,       # (B, L, D), scores in [-1, 1]
    usefulness_score_matrix: torch.Tensor, # (B, L, D), {1, 0, -1}
) -> torch.Tensor:
    """
    Binary Cross Entropy loss for usefulness scores.

    Ignore:
        - label == -1
    """

    # Convert scores from [-1, 1] -> [0, 1]
    usefulness_scores = (usefulness_scores + 1.0) / 2.0

    # Ignore padded / skipped documents
    valid_mask = usefulness_score_matrix != -1

    if not valid_mask.any():
        return usefulness_scores.new_zeros(())

    scores = usefulness_scores[valid_mask]
    targets = usefulness_score_matrix[valid_mask].float()

    loss = F.binary_cross_entropy(
        scores,
        targets,
        reduction="mean",
    )

    return loss