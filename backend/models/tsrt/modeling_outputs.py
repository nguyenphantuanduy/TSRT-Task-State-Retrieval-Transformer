from dataclasses import dataclass
from typing import Dict
import torch

from transformers import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast


@dataclass
class TSRTModelOutputWithPast(BaseModelOutputWithPast):
    """
    Base model output for TSRT.
    """
    retrieval_decision_logits: torch.FloatTensor | None = None
    retrieval_decision: torch.FloatTensor | None = None
    usefulness_score: torch.FloatTensor | None = None
    all_cross_stats: Dict[str, Dict[str, torch.Tensor]] | None = None