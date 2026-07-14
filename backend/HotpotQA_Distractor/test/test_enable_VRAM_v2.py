import torch
import torch.nn as nn


class FakeSelfAttention(nn.Module):

    def __init__(self, hidden_size=2560):
        super().__init__()

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)


class FakeMLP(nn.Module):

    def __init__(
        self,
        hidden_size=2560,
        intermediate_size=6912,
    ):
        super().__init__()

        self.gate_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )

        self.up_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )

        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
        )


class FakeLayer(nn.Module):

    def __init__(self):
        super().__init__()

        self.self_attn = FakeSelfAttention()

        self.mlp = FakeMLP()

        self.input_layernorm = nn.Parameter(
            torch.ones(2560)
        )

        self.post_attention_layernorm = nn.Parameter(
            torch.ones(2560)
        )


class FakeQwen4B(nn.Module):

    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList(
            [FakeLayer() for _ in range(36)]
        )


def freeze_all(model):

    for p in model.parameters():
        p.requires_grad = False


def unfreeze_attention_ffn_norm_qwen3_4b(model):

    for name, param in model.named_parameters():

        lower_name = name.lower()

        # Attention 0-23
        for layer_idx in range(24):

            prefix = f"layers.{layer_idx}."

            if prefix not in lower_name:
                continue

            if any(
                x in lower_name
                for x in [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                ]
            ):
                param.requires_grad = True
                break

        # FFN + Norm 24-35
        for layer_idx in range(24, 36):

            prefix = f"layers.{layer_idx}."

            if prefix not in lower_name:
                continue

            if any(
                x in lower_name
                for x in [
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]
            ):
                param.requires_grad = True
                break

            if any(
                x in lower_name
                for x in [
                    "input_layernorm",
                    "post_attention_layernorm",
                ]
            ):
                param.requires_grad = True
                break


def count_parameters(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


model = FakeQwen4B()

freeze_all(model)
unfreeze_attention_ffn_norm_qwen3_4b(model)

total, trainable = count_parameters(model)

print(f"Total     : {total:,}")
print(f"Trainable : {trainable:,}")