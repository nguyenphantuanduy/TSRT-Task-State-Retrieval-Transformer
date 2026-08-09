from dataclasses import dataclass
from typing import Dict
import torch

from transformers import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import ModelOutput


@dataclass
class TSRTModelOutputWithPast(BaseModelOutputWithPast):
    """
    Base model output for TSRT.
    """
    retrieval_decision_logits: torch.FloatTensor | None = None
    retrieval_decision: torch.FloatTensor | None = None
    usefulness_score: torch.FloatTensor | None = None
    all_cross_stats: Dict[str, Dict[str, torch.Tensor]] | None = None

@dataclass
class TSRTRetrieverOutput(ModelOutput):
    loss: torch.FloatTensor | None = None

    # Sentence-level task/query embeddings
    task_embeddings: torch.FloatTensor | None = None

    # Sentence-level document embeddings
    document_embeddings: torch.FloatTensor | None = None
    