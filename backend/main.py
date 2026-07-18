# from transformers import AutoModelForCausalLM, AutoTokenizer

# MODEL_NAME = "Qwen/Qwen3-1.7B"
# CACHE_DIR = "weights/Qwen3-1.7B"

# tokenizer = AutoTokenizer.from_pretrained(
#     MODEL_NAME,
#     cache_dir=CACHE_DIR
# )

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     cache_dir=CACHE_DIR
# )

# import inspect
# from transformers import GenerationMixin

# print(
#     "stop_strings" in inspect.signature(
#         GenerationMixin.generate
#     ).parameters
# )

# import time
# from datasets import load_dataset

# DATASET_REPO = "nguyenphantuanduy/temp-dataset"

# # ==========================================================
# # DOWNLOAD
# # ==========================================================

# start = time.time()

# dataset = load_dataset(
#     "hotpotqa/hotpot_qa",
#     "distractor",
# )

# download_time = time.time() - start

# print(
#     f"Download + load: "
#     f"{download_time:.2f}s"
# )

# print(dataset)

# # ==========================================================
# # UPLOAD
# # ==========================================================

# start = time.time()

# dataset.push_to_hub(
#     DATASET_REPO,
#     private=False,
# )

# upload_time = time.time() - start

# print(
#     f"Upload time: "
#     f"{upload_time:.2f}s"
# )

# print(
#     f"Total time: "
#     f"{download_time + upload_time:.2f}s"
# )

# import json

# FILE_PATH = "train_00017.jsonl"


# def load_jsonl(path):
#     samples = []

#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             samples.append(json.loads(line))

#     return samples


# def main():
#     data = load_jsonl(FILE_PATH)

#     print("=" * 100)
#     print(f"Total samples: {len(data)}")
#     print("=" * 100)

#     for i, sample in enumerate(data[:5]):
#         print()
#         print("#" * 100)
#         print(f"SAMPLE {i}")
#         print("#" * 100)

#         print(json.dumps(sample, indent=2, ensure_ascii=False))


# if __name__ == "__main__":
#     main()

# from huggingface_hub import hf_hub_download
# import json

# config_path = hf_hub_download(
#     repo_id="Qwen/Qwen3-1.7B",
#     filename="config.json"
# )

# with open(config_path, "r", encoding="utf-8") as f:
#     config = json.load(f)

# print(json.dumps(config, indent=2, ensure_ascii=False))

# import sys

# sys.path.append("models/tsrt")

# from configuration_tsrt import TSRTConfig

# config = TSRTConfig.from_json_file(
#     "models/tsrt/config.json"
# )

# print(config)
# print(config.model_type)
# print(config.num_hidden_layers)
# print(config.num_encoder_layers)
# print(config.num_decoder_layers)
# print(config.num_tsrt_layers)

# from transformers import AutoConfig

# config = AutoConfig.from_pretrained(
#     "models/tsrt",
#     trust_remote_code=True
# )

# print(type(config))
# print(config.num_hidden_layers)
# print(config.num_encoder_layers)
# print(config.num_decoder_layers)
# print(config.num_tsrt_layers)




'''
test decoder cache
'''
# import torch
# from transformers import AutoConfig

# # import class của bạn
# from models.tsrt.cache_utils import TSRTDecoderCache


# def make_dummy_kv(batch=1, seq=4, dim=64, device="cpu"):
#     k = torch.randn(batch, seq, dim, device=device)
#     v = torch.randn(batch, seq, dim, device=device)
#     return (k, v)


# def make_ddp_cache_data(num_layers, dim=64):
#     data = []
#     for _ in range(num_layers):
#         data.append(make_dummy_kv(dim=dim))
#     return data


# def test_init_with_config():
#     print("=== test_init_with_config ===")

#     config = AutoConfig.from_pretrained(
#         "models/tsrt",
#         trust_remote_code=True
#     )

#     cache = TSRTDecoderCache(config=config)

#     expected_layers = config.num_decoder_layers + config.num_tsrt_layers

#     print("num layers:", len(cache.layers))
#     print("expected :", expected_layers)

#     assert len(cache.layers) == expected_layers

#     print("OK\n")


# def test_layer_types_slice():
#     print("=== test_layer_types_slice ===")

#     config = AutoConfig.from_pretrained(
#         "models/tsrt",
#         trust_remote_code=True
#     )

#     cache = TSRTDecoderCache(config=config)

#     if config.layer_types is not None:
#         expected = config.layer_types[config.num_encoder_layers:]

#         print("expected layer_types:", expected)
#         print("actual   layer_types:", cache.cache_config.layer_types)

#         assert cache.cache_config.layer_types == expected

#     print("OK\n")


# def test_ddp_cache_loading():
#     print("=== test_ddp_cache_loading ===")

#     config = AutoConfig.from_pretrained(
#         "models/tsrt",
#         trust_remote_code=True
#     )

#     num_layers = config.num_decoder_layers + config.num_tsrt_layers
#     ddp_data = make_ddp_cache_data(num_layers)

#     cache = TSRTDecoderCache(
#         config=config,
#         ddp_cache_data=ddp_data,
#     )

#     for i in range(num_layers):
#         layer = cache.layers[i]
#         k = layer.keys
#         v = layer.values

#         print(f"layer {i}: shape =", k.shape)

#         assert k.shape == v.shape
#         assert k.shape[1] == 4  # seq length dummy

#     print("OK\n")


# def test_append_behavior():
#     print("=== test_append_behavior ===")

#     config = AutoConfig.from_pretrained(
#         "models/tsrt",
#         trust_remote_code=True
#     )

#     cache = TSRTDecoderCache(config=config)

#     layer_id = 0
#     layer = cache.layers[layer_id]

#     k1, v1 = make_dummy_kv(seq=2)
#     k2, v2 = make_dummy_kv(seq=3)

#     # set initial
#     layer.keys = k1
#     layer.values = v1
#     layer.is_initialized = True

#     # append (giống KV cache)
#     layer.keys = torch.cat([layer.keys, k2], dim=1)
#     layer.values = torch.cat([layer.values, v2], dim=1)

#     final_seq = layer.keys.shape[1]

#     print("final seq length:", final_seq)

#     assert final_seq == 5

#     print("OK\n")


# if __name__ == "__main__":
#     test_init_with_config()
#     test_layer_types_slice()
#     test_ddp_cache_loading()
#     test_append_behavior()

'''
test doc cache
'''
import torch
from transformers import AutoConfig

from models.tsrt.cache_utils import TSRTDocumentCache


# =========================
# Helpers
# =========================

def make_dummy_kv(batch=1, docs=2, seq=4, dim=64, device="cpu", dtype=torch.float32):
    k = torch.randn(batch, docs, seq, dim, device=device, dtype=dtype)
    v = torch.randn(batch, docs, seq, dim, device=device, dtype=dtype)
    return (k, v)


def make_dummy_encoder_state(
    batch=1,
    docs=2,
    seq=8,
    dim=64,
    device="cpu",
    dtype=torch.float32,
):
    return torch.randn(
        batch,
        docs,
        seq,
        dim,
        device=device,
        dtype=dtype,
    )


def make_ddp_cache_data(num_layers, dim=64):
    data = []

    for _ in range(num_layers):
        data.append(make_dummy_kv(dim=dim))

    return data


# =========================
# Tests
# =========================


def test_init_with_config():

    print("=== test_init_with_config ===")

    config = AutoConfig.from_pretrained(
        "models/tsrt",
        trust_remote_code=True,
    )

    cache = TSRTDocumentCache(config=config)

    expected_layers = config.num_tsrt_layers

    print("num layers:", len(cache.layers))
    print("expected :", expected_layers)

    assert len(cache.layers) == expected_layers

    assert cache.encoder_state is None

    print("OK\n")



def test_layer_types_slice():

    print("=== test_layer_types_slice ===")

    config = AutoConfig.from_pretrained(
        "models/tsrt",
        trust_remote_code=True,
    )

    cache = TSRTDocumentCache(config=config)

    if config.layer_types is not None:

        expected = config.layer_types[
            :config.num_tsrt_layers
        ]

        print("expected:", expected)
        print("actual  :", cache.cache_config.layer_types)

        assert cache.cache_config.layer_types == expected

    print("OK\n")



def test_ddp_cache_loading():

    print("=== test_ddp_cache_loading ===")

    config = AutoConfig.from_pretrained(
        "models/tsrt",
        trust_remote_code=True,
    )

    num_layers = config.num_tsrt_layers

    ddp_data = make_ddp_cache_data(num_layers)

    cache = TSRTDocumentCache(
        config=config,
        ddp_cache_data=ddp_data,
    )


    for i in range(num_layers):

        layer = cache.layers[i]

        print(
            f"layer {i}:",
            layer.keys.shape
        )

        assert layer.keys.shape == layer.values.shape
        assert layer.keys.dim() == 4
        assert layer.is_initialized


    print("OK\n")



def test_has_kv_and_get_kv():

    print("=== test_has_kv_and_get_kv ===")

    config = AutoConfig.from_pretrained(
        "models/tsrt",
        trust_remote_code=True,
    )

    cache = TSRTDocumentCache(config=config)

    layer_id = 0


    assert cache.has_kv(layer_id) is False


    k, v = make_dummy_kv(seq=3)


    cache.layers[layer_id].keys = k
    cache.layers[layer_id].values = v
    cache.layers[layer_id].is_initialized = True


    assert cache.has_kv(layer_id)


    k_out, v_out = cache.get_kv(layer_id)


    assert torch.equal(k, k_out)
    assert torch.equal(v, v_out)


    print("OK\n")



def test_get_kv_empty():

    print("=== test_get_kv_empty ===")

    config = AutoConfig.from_pretrained(
        "models/tsrt",
        trust_remote_code=True,
    )

    cache = TSRTDocumentCache(config=config)


    k, v = cache.get_kv(0)

    assert k is None
    assert v is None


    print("OK\n")



def test_reset_kv():

    print("=== test_reset_kv ===")


    config = AutoConfig.from_pretrained(
        "models/tsrt",
        trust_remote_code=True,
    )

    cache = TSRTDocumentCache(config=config)


    k, v = make_dummy_kv(seq=3)


    cache.layers[0].keys = k
    cache.layers[0].values = v
    cache.layers[0].is_initialized = True


    cache.reset_kv(0)


    assert cache.layers[0].keys is None
    assert cache.layers[0].values is None
    assert cache.layers[0].is_initialized is False


    print("OK\n")



def test_reset_all():

    print("=== test_reset_all ===")


    config = AutoConfig.from_pretrained(
        "models/tsrt",
        trust_remote_code=True,
    )


    cache = TSRTDocumentCache(config=config)


    for layer in cache.layers:

        k, v = make_dummy_kv()

        layer.keys = k
        layer.values = v
        layer.is_initialized = True


    cache.reset_all()


    for layer in cache.layers:

        assert layer.keys is None
        assert layer.values is None
        assert layer.is_initialized is False


    print("OK\n")



# =========================
# New update() test
# =========================


def test_update_kv():

    print("=== test_update_kv ===")


    config = AutoConfig.from_pretrained(
        "models/tsrt",
        trust_remote_code=True,
    )

    cache = TSRTDocumentCache(config=config)


    layer = 0


    k1, v1 = make_dummy_kv(seq=3)

    cache.update(
        k1,
        v1,
        layer,
    )


    assert cache.layers[layer].keys.shape[-2] == 3



    k2, v2 = make_dummy_kv(seq=4)

    cache.update(
        k2,
        v2,
        layer,
    )


    assert cache.layers[layer].keys.shape[-2] == 7


    print("OK\n")



def test_update_kv_dtype_fix():

    print("=== test_update_kv_dtype_fix ===")


    config = AutoConfig.from_pretrained(
        "models/tsrt",
        trust_remote_code=True,
    )


    cache = TSRTDocumentCache(config=config)


    k1, v1 = make_dummy_kv(
        seq=2,
        dtype=torch.float32,
    )

    cache.update(k1, v1, 0)


    k2, v2 = make_dummy_kv(
        seq=2,
        dtype=torch.float16,
    )


    cache.update(k2, v2, 0)


    assert cache.layers[0].keys.dtype == torch.float32


    print("OK\n")



# =========================
# Encoder state tests
# =========================


def test_encoder_state_init():

    print("=== test_encoder_state_init ===")


    config = AutoConfig.from_pretrained(
        "models/tsrt",
        trust_remote_code=True,
    )


    cache = TSRTDocumentCache(config=config)


    assert cache.has_encoder_state() is False
    assert cache.encoder_state is None


    print("OK\n")



def test_update_encoder_state():

    print("=== test_update_encoder_state ===")


    config = AutoConfig.from_pretrained(
        "models/tsrt",
        trust_remote_code=True,
    )


    cache = TSRTDocumentCache(config=config)


    state1 = make_dummy_encoder_state(
        docs=2
    )


    cache.update_encoder_state(state1)


    assert cache.has_encoder_state()

    assert cache.encoder_state.shape == (
        1,
        2,
        8,
        64
    )



    state2 = make_dummy_encoder_state(
        docs=3
    )


    cache.update_encoder_state(state2)


    assert cache.encoder_state.shape == (
        1,
        5,
        8,
        64
    )


    print("OK\n")



def test_update_encoder_state_dtype_fix():

    print("=== test_update_encoder_state_dtype_fix ===")


    config = AutoConfig.from_pretrained(
        "models/tsrt",
        trust_remote_code=True,
    )


    cache = TSRTDocumentCache(config=config)


    state1 = make_dummy_encoder_state(
        dtype=torch.float32
    )


    cache.update_encoder_state(state1)



    state2 = make_dummy_encoder_state(
        dtype=torch.float16
    )


    cache.update_encoder_state(state2)


    assert cache.encoder_state.dtype == torch.float32


    print("OK\n")



def test_reset_encoder_state():

    print("=== test_reset_encoder_state ===")


    config = AutoConfig.from_pretrained(
        "models/tsrt",
        trust_remote_code=True,
    )


    cache = TSRTDocumentCache(config=config)


    state = make_dummy_encoder_state()

    cache.update_encoder_state(state)


    assert cache.has_encoder_state()



    cache.reset_encoder_state()


    assert cache.encoder_state is None
    assert cache.has_encoder_state() is False


    print("OK\n")



# =========================
# Run
# =========================


if __name__ == "__main__":

    test_init_with_config()
    test_layer_types_slice()

    test_ddp_cache_loading()

    test_has_kv_and_get_kv()
    test_get_kv_empty()

    test_reset_kv()
    test_reset_all()

    test_update_kv()
    test_update_kv_dtype_fix()

    test_encoder_state_init()
    test_update_encoder_state()
    test_update_encoder_state_dtype_fix()
    test_reset_encoder_state()
'''
test emb cache
'''
# import torch

# from models.tsrt.cache_utils import TSRTEmbeddingCache


# # =========================
# # Helpers
# # =========================
# def make_tensor(batch=1, seq=3, dim=64, device="cpu", dtype=torch.float32):
#     return torch.randn(batch, seq, dim, device=device, dtype=dtype)


# # =========================
# # Tests
# # =========================
# def test_init_empty():
#     print("=== test_init_empty ===")

#     cache = TSRTEmbeddingCache()

#     assert cache.weights is None
#     assert cache.hidden_embs is None
#     assert cache.doc_embs is None
#     assert cache.dtype is None
#     assert cache.device is None

#     print("OK\n")


# def test_init_with_data():
#     print("=== test_init_with_data ===")

#     w = make_tensor()
#     h = make_tensor()

#     cache = TSRTEmbeddingCache(weights=w, hidden_embs=h)

#     assert cache.weights is w
#     assert cache.hidden_embs is h
#     assert cache.dtype == w.dtype
#     assert cache.device == w.device

#     print("OK\n")


# def test_update_concat():
#     print("=== test_update_concat ===")

#     cache = TSRTEmbeddingCache()

#     w1 = make_tensor(seq=3)
#     h1 = make_tensor(seq=3)

#     cache.update(w1, h1)

#     assert cache.weights.shape[1] == 3
#     assert cache.hidden_embs.shape[1] == 3

#     w2 = make_tensor(seq=4)
#     h2 = make_tensor(seq=4)

#     cache.update(w2, h2)

#     print("final seq:", cache.weights.shape[1])

#     assert cache.weights.shape[1] == 7
#     assert cache.hidden_embs.shape[1] == 7

#     # prefix giữ nguyên
#     assert torch.equal(cache.weights[:, :3, :], w1)
#     assert torch.equal(cache.hidden_embs[:, :3, :], h1)

#     print("OK\n")


# def test_update_dtype_device_consistency():
#     print("=== test_update_dtype_device_consistency ===")

#     cache = TSRTEmbeddingCache()

#     w1 = make_tensor(dtype=torch.float32)
#     h1 = make_tensor(dtype=torch.float32)

#     cache.update(w1, h1)

#     # update với dtype khác
#     w2 = make_tensor(dtype=torch.float16)
#     h2 = make_tensor(dtype=torch.float16)

#     cache.update(w2, h2)

#     assert cache.weights.dtype == torch.float32
#     assert cache.hidden_embs.dtype == torch.float32

#     print("OK\n")


# def test_update_doc_embs():
#     print("=== test_update_doc_embs ===")

#     cache = TSRTEmbeddingCache()

#     d1 = make_tensor(seq=2)

#     cache.update_doc_embs(d1)

#     assert cache.doc_embs.shape[1] == 2

#     d2 = make_tensor(seq=5)

#     cache.update_doc_embs(d2)

#     # overwrite, không concat
#     assert cache.doc_embs.shape[1] == 5
#     assert torch.equal(cache.doc_embs, d2)

#     print("OK\n")


# def test_has_doc_embs():
#     print("=== test_has_doc_embs ===")

#     cache = TSRTEmbeddingCache()

#     assert cache.has_doc_embs() is False

#     d = make_tensor(seq=2)
#     cache.update_doc_embs(d)

#     assert cache.has_doc_embs() is True

#     print("OK\n")


# def test_reset_doc_embs():
#     print("=== test_reset_doc_embs ===")

#     cache = TSRTEmbeddingCache()

#     d = make_tensor(seq=2)
#     cache.update_doc_embs(d)

#     cache.reset_doc_embs()

#     assert cache.doc_embs is None
#     assert cache.has_doc_embs() is False

#     print("OK\n")


# def test_reset_all():
#     print("=== test_reset_all ===")

#     cache = TSRTEmbeddingCache()

#     w = make_tensor()
#     h = make_tensor()
#     d = make_tensor()

#     cache.update(w, h)
#     cache.update_doc_embs(d)

#     cache.reset_all()

#     assert cache.weights is None
#     assert cache.hidden_embs is None
#     assert cache.doc_embs is None
#     assert cache.dtype is None
#     assert cache.device is None

#     print("OK\n")


# def test_device_alignment():
#     print("=== test_device_alignment ===")

#     cache = TSRTEmbeddingCache()

#     w = make_tensor(device="cpu")
#     h = make_tensor(device="cpu")

#     cache.update(w, h)

#     w2 = make_tensor(device="cpu")
#     h2 = make_tensor(device="cpu")

#     cache.update(w2, h2)

#     assert cache.weights.device == cache.hidden_embs.device

#     print("OK\n")


# # =========================
# # Run
# # =========================
# if __name__ == "__main__":
#     test_init_empty()
#     test_init_with_data()
#     test_update_concat()
#     test_update_dtype_device_consistency()
#     test_update_doc_embs()
#     test_has_doc_embs()
#     test_reset_doc_embs()
#     test_reset_all()
#     test_device_alignment()