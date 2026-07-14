import copy
import torch

from transformers import (
    AutoModelForCausalLM,
)

MODEL_NAME = "Qwen/Qwen3-1.7B"


def freeze_all(model):

    for p in model.parameters():
        p.requires_grad = False


def unfreeze_attention_ffn_36layer(model):

    for name, param in model.named_parameters():

        lower_name = name.lower()

        # ==================================================
        # Attention layers 0 -> 23
        # ==================================================
        for layer_idx in range(24):

            prefix = f"model.layers.{layer_idx}."

            if prefix not in lower_name:
                continue

            if any(
                k in lower_name
                for k in [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                ]
            ):
                param.requires_grad = True

            break

        # ==================================================
        # FFN layers 24 -> 35
        # ==================================================
        for layer_idx in range(24, 36):

            prefix = f"model.layers.{layer_idx}."

            if prefix not in lower_name:
                continue

            if any(
                k in lower_name
                for k in [
                    "gate_proj",
                    "up_proj",
                    "down_proj",
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


def print_breakdown(model):

    attn = 0
    ffn = 0

    for name, p in model.named_parameters():

        if not p.requires_grad:
            continue

        if any(
            k in name
            for k in [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ]
        ):
            attn += p.numel()

        elif any(
            k in name
            for k in [
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        ):
            ffn += p.numel()

    print()
    print("Breakdown")
    print("-" * 80)
    print(f"Attention params : {attn:,}")
    print(f"FFN params       : {ffn:,}")
    print("-" * 80)


def expand_to_36_layers(model):

    layers = model.model.layers

    original_num_layers = len(layers)

    print()
    print(
        f"Original layers : "
        f"{original_num_layers}"
    )

    while len(layers) < 36:

        src_idx = len(layers) % original_num_layers

        new_layer = copy.deepcopy(
            layers[src_idx]
        )

        layers.extend(
            [new_layer]
        )

    print(
        f"Expanded layers : "
        f"{len(layers)}"
    )

    return model


def main():

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print(
        f"Before expand: "
        f"{len(model.model.layers)}"
    )

    model = expand_to_36_layers(model)

    print(
        f"After expand : "
        f"{len(model.model.layers)}"
    )

    freeze_all(model)

    unfreeze_attention_ffn_36layer(model)

    total, trainable = count_parameters(model)

    print()
    print("=" * 80)
    print(
        f"Total params     : "
        f"{total:,}"
    )
    print(
        f"Trainable params : "
        f"{trainable:,}"
    )
    print("=" * 80)

    print_breakdown(model)

    print()
    print("First trainable tensors:")
    print("-" * 80)

    shown = 0

    for name, p in model.named_parameters():

        if not p.requires_grad:
            continue

        print(name)

        shown += 1

        if shown >= 30:
            break


if __name__ == "__main__":
    main()