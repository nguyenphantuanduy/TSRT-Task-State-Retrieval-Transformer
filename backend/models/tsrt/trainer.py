from __future__ import annotations

import torch
from transformers import Trainer


class TSRTTrainer(Trainer):
    """
    Hugging Face Trainer with automatic logging of
    TSRT auxiliary losses stored in model.logged_losses.
    """

    def log(
        self,
        logs: dict[str, float],
        start_time: float | None = None,
    ) -> None:
        """
        Append model.logged_losses to Trainer logs.

        The model is expected to expose:

            model.logged_losses: dict[str, Tensor | float | None]
        """

        model = self.model

        # unwrap if needed (Accelerate / DDP / DeepSpeed / FSDP ...)
        if hasattr(model, "module"):
            model = model.module

        if hasattr(model, "logged_losses"):

            for key, value in model.logged_losses.items():
                if value is None:
                    continue

                logs[key] = value.item() if torch.is_tensor(value) else value

        super().log(
            logs,
            start_time=start_time,
        )