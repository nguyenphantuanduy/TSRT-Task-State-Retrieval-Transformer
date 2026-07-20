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
# import torch
# from transformers import AutoConfig

# from models.tsrt.cache_utils import TSRTDocumentCache


# # =========================
# # Helpers
# # =========================
# def make_dummy_multidoc_kv(
#     batch=1,
#     docs=3,
#     heads=8,
#     seq=4,
#     head_dim=64,
#     device="cpu",
#     dtype=torch.float32,
# ):
#     k = torch.randn(
#         batch,
#         docs,
#         heads,
#         seq,
#         head_dim,
#         device=device,
#         dtype=dtype,
#     )

#     v = torch.randn(
#         batch,
#         docs,
#         heads,
#         seq,
#         head_dim,
#         device=device,
#         dtype=dtype,
#     )

#     return k, v

# def test_update_multidoc_kv():

#     print("=== test_update_multidoc_kv ===")

#     config = AutoConfig.from_pretrained(
#         "models/tsrt",
#         trust_remote_code=True,
#     )

#     cache = TSRTDocumentCache(config=config)

#     layer = 0

#     k1, v1 = make_dummy_multidoc_kv(
#         batch=2,
#         docs=3,
#         heads=4,
#         seq=5,
#         head_dim=32,
#     )

#     key_out, value_out = cache.update(
#         k1,
#         v1,
#         layer,
#     )

#     assert key_out.shape == (
#         2,
#         3,
#         4,
#         5,
#         32,
#     )

#     assert value_out.shape == (
#         2,
#         3,
#         4,
#         5,
#         32,
#     )

#     assert cache.layers[layer].keys.shape == (
#         2,
#         3,
#         4,
#         5,
#         32,
#     )

#     # update lần 2

#     k2, v2 = make_dummy_multidoc_kv(
#         batch=2,
#         docs=3,
#         heads=4,
#         seq=7,
#         head_dim=32,
#     )

#     key_out, value_out = cache.update(
#         k2,
#         v2,
#         layer,
#     )

#     assert key_out.shape == (
#         2,
#         3,
#         4,
#         12,
#         32,
#     )

#     assert value_out.shape == (
#         2,
#         3,
#         4,
#         12,
#         32,
#     )

#     assert cache.layers[layer].keys.shape == (
#         2,
#         3,
#         4,
#         12,
#         32,
#     )

#     assert cache.layers[layer].values.shape == (
#         2,
#         3,
#         4,
#         12,
#         32,
#     )

#     print("OK\n")

# def make_dummy_kv(batch=1, docs=2, seq=4, dim=64, device="cpu", dtype=torch.float32):
#     k = torch.randn(batch, docs, seq, dim, device=device, dtype=dtype)
#     v = torch.randn(batch, docs, seq, dim, device=device, dtype=dtype)
#     return (k, v)


# def make_dummy_encoder_state(
#     batch=1,
#     docs=2,
#     seq=8,
#     dim=64,
#     device="cpu",
#     dtype=torch.float32,
# ):
#     return torch.randn(
#         batch,
#         docs,
#         seq,
#         dim,
#         device=device,
#         dtype=dtype,
#     )


# def make_ddp_cache_data(num_layers, dim=64):
#     data = []

#     for _ in range(num_layers):
#         data.append(make_dummy_kv(dim=dim))

#     return data


# # =========================
# # Tests
# # =========================


# def test_init_with_config():

#     print("=== test_init_with_config ===")

    # config = AutoConfig.from_pretrained(
    #     "models/tsrt",
    #     trust_remote_code=True,
    # )

#     cache = TSRTDocumentCache(config=config)

#     expected_layers = config.num_tsrt_layers

#     print("num layers:", len(cache.layers))
#     print("expected :", expected_layers)

#     assert len(cache.layers) == expected_layers

#     assert cache.encoder_state is None

#     print("OK\n")



# def test_layer_types_slice():

#     print("=== test_layer_types_slice ===")

#     config = AutoConfig.from_pretrained(
#         "models/tsrt",
#         trust_remote_code=True,
#     )

#     cache = TSRTDocumentCache(config=config)

#     if config.layer_types is not None:

#         expected = config.layer_types[
#             :config.num_tsrt_layers
#         ]

#         print("expected:", expected)
#         print("actual  :", cache.cache_config.layer_types)

#         assert cache.cache_config.layer_types == expected

#     print("OK\n")



# def test_ddp_cache_loading():

#     print("=== test_ddp_cache_loading ===")

#     config = AutoConfig.from_pretrained(
#         "models/tsrt",
#         trust_remote_code=True,
#     )

#     num_layers = config.num_tsrt_layers

#     ddp_data = make_ddp_cache_data(num_layers)

#     cache = TSRTDocumentCache(
#         config=config,
#         ddp_cache_data=ddp_data,
#     )


#     for i in range(num_layers):

#         layer = cache.layers[i]

#         print(
#             f"layer {i}:",
#             layer.keys.shape
#         )

#         assert layer.keys.shape == layer.values.shape
#         assert layer.keys.dim() == 4
#         assert layer.is_initialized


#     print("OK\n")



# def test_has_kv_and_get_kv():

#     print("=== test_has_kv_and_get_kv ===")

#     config = AutoConfig.from_pretrained(
#         "models/tsrt",
#         trust_remote_code=True,
#     )

#     cache = TSRTDocumentCache(config=config)

#     layer_id = 0


#     assert cache.has_kv(layer_id) is False


#     k, v = make_dummy_kv(seq=3)


#     cache.layers[layer_id].keys = k
#     cache.layers[layer_id].values = v
#     cache.layers[layer_id].is_initialized = True


#     assert cache.has_kv(layer_id)


#     k_out, v_out = cache.get_kv(layer_id)


#     assert torch.equal(k, k_out)
#     assert torch.equal(v, v_out)


#     print("OK\n")



# def test_get_kv_empty():

#     print("=== test_get_kv_empty ===")

#     config = AutoConfig.from_pretrained(
#         "models/tsrt",
#         trust_remote_code=True,
#     )

#     cache = TSRTDocumentCache(config=config)


#     k, v = cache.get_kv(0)

#     assert k is None
#     assert v is None


#     print("OK\n")



# def test_reset_kv():

#     print("=== test_reset_kv ===")


#     config = AutoConfig.from_pretrained(
#         "models/tsrt",
#         trust_remote_code=True,
#     )

#     cache = TSRTDocumentCache(config=config)


#     k, v = make_dummy_kv(seq=3)


#     cache.layers[0].keys = k
#     cache.layers[0].values = v
#     cache.layers[0].is_initialized = True


#     cache.reset_kv(0)


#     assert cache.layers[0].keys is None
#     assert cache.layers[0].values is None
#     assert cache.layers[0].is_initialized is False


#     print("OK\n")



# def test_reset_all():

#     print("=== test_reset_all ===")


#     config = AutoConfig.from_pretrained(
#         "models/tsrt",
#         trust_remote_code=True,
#     )


#     cache = TSRTDocumentCache(config=config)


#     for layer in cache.layers:

#         k, v = make_dummy_kv()

#         layer.keys = k
#         layer.values = v
#         layer.is_initialized = True


#     cache.reset_all()


#     for layer in cache.layers:

#         assert layer.keys is None
#         assert layer.values is None
#         assert layer.is_initialized is False


#     print("OK\n")



# # =========================
# # New update() test
# # =========================


# def test_update_kv():

#     print("=== test_update_kv ===")


#     config = AutoConfig.from_pretrained(
#         "models/tsrt",
#         trust_remote_code=True,
#     )

#     cache = TSRTDocumentCache(config=config)


#     layer = 0


#     k1, v1 = make_dummy_kv(seq=3)

#     cache.update(
#         k1,
#         v1,
#         layer,
#     )


#     assert cache.layers[layer].keys.shape[-2] == 3



#     k2, v2 = make_dummy_kv(seq=4)

#     cache.update(
#         k2,
#         v2,
#         layer,
#     )


#     assert cache.layers[layer].keys.shape[-2] == 7


#     print("OK\n")



# def test_update_kv_dtype_fix():

#     print("=== test_update_kv_dtype_fix ===")


#     config = AutoConfig.from_pretrained(
#         "models/tsrt",
#         trust_remote_code=True,
#     )


#     cache = TSRTDocumentCache(config=config)


#     k1, v1 = make_dummy_kv(
#         seq=2,
#         dtype=torch.float32,
#     )

#     cache.update(k1, v1, 0)


#     k2, v2 = make_dummy_kv(
#         seq=2,
#         dtype=torch.float16,
#     )


#     cache.update(k2, v2, 0)


#     assert cache.layers[0].keys.dtype == torch.float32


#     print("OK\n")



# # =========================
# # Encoder state tests
# # =========================


# def test_encoder_state_init():

#     print("=== test_encoder_state_init ===")


#     config = AutoConfig.from_pretrained(
#         "models/tsrt",
#         trust_remote_code=True,
#     )


#     cache = TSRTDocumentCache(config=config)


#     assert cache.has_encoder_state() is False
#     assert cache.encoder_state is None


#     print("OK\n")



# def test_update_encoder_state():

#     print("=== test_update_encoder_state ===")


#     config = AutoConfig.from_pretrained(
#         "models/tsrt",
#         trust_remote_code=True,
#     )


#     cache = TSRTDocumentCache(config=config)


#     state1 = make_dummy_encoder_state(
#         docs=2
#     )


#     cache.update_encoder_state(state1)


#     assert cache.has_encoder_state()

#     assert cache.encoder_state.shape == (
#         1,
#         2,
#         8,
#         64
#     )



#     state2 = make_dummy_encoder_state(
#         docs=3
#     )


#     cache.update_encoder_state(state2)


#     assert cache.encoder_state.shape == (
#         1,
#         5,
#         8,
#         64
#     )


#     print("OK\n")



# def test_update_encoder_state_dtype_fix():

#     print("=== test_update_encoder_state_dtype_fix ===")


#     config = AutoConfig.from_pretrained(
#         "models/tsrt",
#         trust_remote_code=True,
#     )


#     cache = TSRTDocumentCache(config=config)


#     state1 = make_dummy_encoder_state(
#         dtype=torch.float32
#     )


#     cache.update_encoder_state(state1)



#     state2 = make_dummy_encoder_state(
#         dtype=torch.float16
#     )


#     cache.update_encoder_state(state2)


#     assert cache.encoder_state.dtype == torch.float32


#     print("OK\n")



# def test_reset_encoder_state():

#     print("=== test_reset_encoder_state ===")


#     config = AutoConfig.from_pretrained(
#         "models/tsrt",
#         trust_remote_code=True,
#     )


#     cache = TSRTDocumentCache(config=config)


#     state = make_dummy_encoder_state()

#     cache.update_encoder_state(state)


#     assert cache.has_encoder_state()



#     cache.reset_encoder_state()


#     assert cache.encoder_state is None
#     assert cache.has_encoder_state() is False


#     print("OK\n")

# def test_update_multidoc_kv_shape():

#     print("=== test_update_multidoc_kv_shape ===")

#     config = AutoConfig.from_pretrained(
#         "models/tsrt",
#         trust_remote_code=True,
#     )

#     cache = TSRTDocumentCache(config=config)

#     layer = 0

#     k1, v1 = make_dummy_multidoc_kv(
#         batch=2,
#         docs=3,
#         heads=4,
#         seq=5,
#         head_dim=32,
#     )

#     cache.update(k1, v1, layer)

#     k2, v2 = make_dummy_multidoc_kv(
#         batch=2,
#         docs=3,
#         heads=4,
#         seq=7,
#         head_dim=32,
#     )

#     cache.update(k2, v2, layer)

#     k_out, v_out = cache.get_kv(layer)

#     assert k_out.shape == (2, 3, 4, 12, 32)
#     assert v_out.shape == (2, 3, 4, 12, 32)

#     print("OK\n")

# def test_update_multidoc_kv_values():

#     print("=== test_update_multidoc_kv_values ===")

#     config = AutoConfig.from_pretrained(
#         "models/tsrt",
#         trust_remote_code=True,
#     )

#     cache = TSRTDocumentCache(config=config)

#     layer = 0

#     k1, v1 = make_dummy_multidoc_kv(
#         batch=2,
#         docs=3,
#         heads=4,
#         seq=5,
#         head_dim=32,
#     )

#     cache.update(k1, v1, layer)

#     k2, v2 = make_dummy_multidoc_kv(
#         batch=2,
#         docs=3,
#         heads=4,
#         seq=7,
#         head_dim=32,
#     )

#     cache.update(k2, v2, layer)

#     expected_k = torch.cat(
#         [k1, k2],
#         dim=-2,
#     )

#     expected_v = torch.cat(
#         [v1, v2],
#         dim=-2,
#     )

#     k_out, v_out = cache.get_kv(layer)

#     assert torch.equal(k_out, expected_k)
#     assert torch.equal(v_out, expected_v)

#     print("OK\n")

# def test_update_multidoc_kv_order():

#     print("=== test_update_multidoc_kv_order ===")

#     config = AutoConfig.from_pretrained(
#         "models/tsrt",
#         trust_remote_code=True,
#     )

#     cache = TSRTDocumentCache(config=config)

#     layer = 0

#     k1 = torch.arange(
#         2 * 3 * 4 * 5 * 8,
#         dtype=torch.float32,
#     ).reshape(
#         2, 3, 4, 5, 8,
#     )

#     v1 = k1 + 10000

#     cache.update(
#         k1,
#         v1,
#         layer,
#     )

#     k2 = (
#         torch.arange(
#             2 * 3 * 4 * 7 * 8,
#             dtype=torch.float32,
#         ).reshape(
#             2, 3, 4, 7, 8,
#         )
#         + 100000
#     )

#     v2 = k2 + 10000

#     cache.update(
#         k2,
#         v2,
#         layer,
#     )

#     expected_k = torch.cat(
#         [k1, k2],
#         dim=-2,
#     )

#     expected_v = torch.cat(
#         [v1, v2],
#         dim=-2,
#     )

#     k_out, v_out = cache.get_kv(layer)

#     assert torch.equal(k_out, expected_k)
#     assert torch.equal(v_out, expected_v)

#     # kiểm tra vài vị trí cụ thể

#     assert k_out[0, 0, 0, 0, 0] == k1[0, 0, 0, 0, 0]

#     assert k_out[0, 0, 0, 4, 0] == k1[0, 0, 0, 4, 0]

#     assert k_out[0, 0, 0, 5, 0] == k2[0, 0, 0, 0, 0]

#     assert k_out[0, 0, 0, 11, 0] == k2[0, 0, 0, 6, 0]

#     print("OK\n")

# # =========================
# # Run
# # =========================


# if __name__ == "__main__":

#     test_init_with_config()
#     test_layer_types_slice()

#     test_ddp_cache_loading()

#     test_has_kv_and_get_kv()
#     test_get_kv_empty()

#     test_reset_kv()
#     test_reset_all()

#     test_update_kv()
#     test_update_kv_dtype_fix()

#     test_encoder_state_init()
#     test_update_encoder_state()
#     test_update_encoder_state_dtype_fix()
#     test_reset_encoder_state()
#     test_update_multidoc_kv()
#     test_update_multidoc_kv_order()
#     test_update_multidoc_kv_values()
#     test_update_multidoc_kv_shape()
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

# from transformers import AutoConfig

# from models.tsrt.utils import tsrt_config_to_qwen3_config


# def main():

#     tsrt_config = AutoConfig.from_pretrained(
#         "models/tsrt",
#         trust_remote_code=True,
#     )

#     print("=== TSRT Config ===")
#     print(type(tsrt_config))
#     print("hidden_size:", tsrt_config.hidden_size)
#     print("num_hidden_layers:", tsrt_config.num_hidden_layers)
#     print("num_encoder_layers:", tsrt_config.num_encoder_layers)
#     print("num_decoder_layers:", tsrt_config.num_decoder_layers)
#     print("num_tsrt_layers:", tsrt_config.num_tsrt_layers)

#     qwen_config = tsrt_config_to_qwen3_config(
#         tsrt_config
#     )

#     print("\n=== Qwen Config ===")
#     print(type(qwen_config))
#     print("hidden_size:", qwen_config.hidden_size)
#     print("num_hidden_layers:", qwen_config.num_hidden_layers)
#     print("num_attention_heads:", qwen_config.num_attention_heads)
#     print("num_key_value_heads:", qwen_config.num_key_value_heads)

#     print("\n=== Equality Check ===")

#     fields = [
#         "vocab_size",
#         "hidden_size",
#         "intermediate_size",
#         "num_hidden_layers",
#         "num_attention_heads",
#         "num_key_value_heads",
#         "head_dim",
#         "hidden_act",
#         "max_position_embeddings",
#         "initializer_range",
#         "rms_norm_eps",
#         "attention_bias",
#         "attention_dropout",
#     ]

#     for field in fields:
#         tsrt_value = getattr(tsrt_config, field)
#         qwen_value = getattr(qwen_config, field)

#         ok = tsrt_value == qwen_value

#         print(
#             f"{field:30s} : "
#             f"{'OK' if ok else 'FAIL'}"
#         )


# if __name__ == "__main__":
#     main()

# import torch
# from transformers import Qwen3Config
# from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding


# def main():

#     config = Qwen3Config(
#         hidden_size=2048,
#         num_attention_heads=16,
#         max_position_embeddings=4096,
#     )

#     rotary = Qwen3RotaryEmbedding(config)

#     B = 4
#     L = 128

#     hidden_states = torch.randn(
#         B,
#         L,
#         config.hidden_size,
#     )

#     position_ids = torch.arange(L).unsqueeze(0)

#     print("hidden_states:", hidden_states.shape)
#     print("position_ids :", position_ids.shape)

#     cos, sin = rotary(hidden_states, position_ids)

#     print()
#     print("cos:", cos.shape)
#     print("sin:", sin.shape)


# if __name__ == "__main__":
#     main()

# import torch
# from transformers import AutoConfig

# from models.tsrt.modeling_tsrt import TSRTCrossAttention
# from models.tsrt.cache_utils import TSRTDocumentCache


# torch.manual_seed(42)


# # ==========================================================
# # Helpers
# # ==========================================================

# def make_config():
#     config = AutoConfig.from_pretrained(
#         "models/tsrt",
#         trust_remote_code=True,
#     )

#     # giảm size để chạy CPU
#     config.hidden_size = 128
#     config.num_attention_heads = 4
#     config.num_key_value_heads = 2
#     config.head_dim = 32

#     config.attention_bias = False
#     config.attention_dropout = 0.0
#     config._attn_implementation = "eager"

#     config.rms_norm_eps = 1e-6

#     return config



# def make_inputs(
#     batch=2,
#     seq_q=5,
#     docs=3,
#     seq_k=7,
#     hidden=128,
# ):

#     decoder_hidden_states = torch.randn(
#         batch,
#         seq_q,
#         hidden,
#     )

#     encoder_hidden_states = torch.randn(
#         batch,
#         docs,
#         seq_k,
#         hidden,
#     )

#     retrieval_memory = torch.randn(
#         batch,
#         seq_q,
#         docs,
#     ).tanh()


#     decoder_cos = torch.randn(
#         1,
#         seq_q,
#         32,
#     )

#     decoder_sin = torch.randn(
#         1,
#         seq_q,
#         32,
#     )


#     encoder_cos = torch.randn(
#         1,
#         seq_k,
#         32,
#     )

#     encoder_sin = torch.randn(
#         1,
#         seq_k,
#         32,
#     )


#     return (
#         decoder_hidden_states,
#         encoder_hidden_states,
#         retrieval_memory,
#         (decoder_cos, decoder_sin),
#         (encoder_cos, encoder_sin),
#     )



# # ==========================================================
# # Tests
# # ==========================================================


# def test_forward_shape():

#     print("=== test_forward_shape ===")


#     config = make_config()

#     attn = TSRTCrossAttention(
#         config,
#         layer_idx=0,
#     )


#     inputs = make_inputs()


#     out, weights = attn(
#         *inputs,
#         attention_mask=None,
#     )


#     print("output :", out.shape)
#     print("weight :", weights.shape)


#     assert out.shape == (
#         2,
#         5,
#         128,
#     )


#     assert weights.shape == (
#         2,
#         4,
#         5,
#         21,     # docs * seq_k = 3*7
#     )


#     print("OK\n")



# def test_forward_single_doc():

#     print("=== test_forward_single_doc ===")


#     config = make_config()

#     attn = TSRTCrossAttention(
#         config,
#         layer_idx=0,
#     )


#     inputs = make_inputs(
#         docs=1,
#         seq_k=4,
#     )


#     out, weights = attn(
#         *inputs,
#         attention_mask=None,
#     )


#     print(out.shape)
#     print(weights.shape)


#     assert weights.shape[-1] == 4


#     print("OK\n")



# def test_cache_path():

#     print("=== test_cache_path ===")


#     config = make_config()


#     attn = TSRTCrossAttention(
#         config,
#         layer_idx=0,
#     )


#     cache = TSRTDocumentCache(
#         config=config,
#     )


#     inputs = make_inputs(
#         batch=1,
#         seq_q=3,
#         docs=2,
#         seq_k=4,
#     )


#     # lần đầu:
#     out1, w1 = attn(
#         *inputs,
#         attention_mask=None,
#         past_key_values=cache,
#     )


#     assert cache.has_kv(0)


#     k, v = cache.get_kv(0)


#     print(
#         "cached k:",
#         k.shape
#     )

#     print(
#         "cached v:",
#         v.shape
#     )


#     # lần hai:
#     out2, w2 = attn(
#         *inputs,
#         attention_mask=None,
#         past_key_values=cache,
#     )


#     assert out2.shape == out1.shape


#     print("OK\n")



# def test_gradient():

#     print("=== test_gradient ===")


#     config = make_config()


#     attn = TSRTCrossAttention(
#         config,
#         layer_idx=0,
#     )


#     inputs = make_inputs(
#         batch=1,
#         seq_q=2,
#         docs=2,
#         seq_k=3,
#     )


#     decoder = inputs[0]
#     decoder.requires_grad_(True)


#     inputs = (
#         decoder,
#         *inputs[1:]
#     )


#     out, _ = attn(
#         *inputs,
#         attention_mask=None,
#     )


#     loss = out.mean()

#     loss.backward()


#     print(
#         "decoder grad:",
#         decoder.grad.abs().mean()
#     )


#     assert decoder.grad is not None


#     print("OK\n")



# def test_retrieval_bias_shape():

#     print("=== test_retrieval_bias_shape ===")


#     B = 2
#     L = 3
#     D = 4
#     Lk = 5


#     retrieval_memory = torch.rand(
#         B,
#         L,
#         D
#     ) * 2 - 1


#     bias = (
#         1
#         + torch.log(
#             torch.clamp(
#                 (retrieval_memory + 1) / 2,
#                 min=1e-6,
#             )
#         )
#     )


#     bias = (
#         bias
#         .unsqueeze(-1)
#         .expand(-1,-1,-1,Lk)
#         .reshape(B,L,D*Lk)
#         .unsqueeze(1)
#     )


#     print(
#         "bias:",
#         bias.shape
#     )


#     assert bias.shape == (
#         B,
#         1,
#         L,
#         D*Lk
#     )


#     print("OK\n")

# def test_retrieval_gradient():

#     print("=== test_retrieval_gradient ===")

#     config = make_config()

#     attn = TSRTCrossAttention(
#         config,
#         layer_idx=0,
#     )

#     inputs = make_inputs(
#         batch=1,
#         seq_q=3,
#         docs=2,
#         seq_k=4,
#     )

#     retrieval_memory = inputs[2]
#     retrieval_memory.requires_grad_(True)

#     inputs = (
#         inputs[0],
#         inputs[1],
#         retrieval_memory,
#         inputs[3],
#         inputs[4],
#     )

#     out, _ = attn(
#         *inputs,
#         attention_mask=None,
#     )

#     loss = out.mean()

#     loss.backward()


#     print(
#         "retrieval grad:",
#         retrieval_memory.grad
#     )


#     assert retrieval_memory.grad is not None

#     print("OK\n")

# # ==========================================================
# # Run
# # ==========================================================

# if __name__ == "__main__":

#     test_forward_shape()

#     test_forward_single_doc()

#     test_cache_path()

#     test_gradient()

#     test_retrieval_bias_shape()
#     test_retrieval_gradient()

# import torch

# from transformers.cache_utils import DynamicCache
# from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

# from models.tsrt.configuration_tsrt import TSRTConfig
# from models.tsrt.modeling_tsrt import TSRTLayer
# from models.tsrt.cache_utils import TSRTDocumentCache


# torch.manual_seed(42)


# def build_config():
#     return TSRTConfig(
#         hidden_size=128,
#         intermediate_size=256,
#         num_attention_heads=4,
#         num_key_value_heads=2,
#         head_dim=32,
#         num_hidden_layers=2,
#         attention_dropout=0.0,
#         rms_norm_eps=1e-6,
#     )


# def build_pos(config, seq_len):
#     rotary = Qwen3RotaryEmbedding(config)

#     x = torch.zeros(
#         1,
#         seq_len,
#         config.hidden_size,
#     )

#     pos = torch.arange(seq_len).unsqueeze(0)

#     return rotary(x, pos)


# def test_forward():

#     print("=" * 60)
#     print("Forward")

#     config = build_config()

#     layer = TSRTLayer(
#         config,
#         0,
#         1,
#     )

#     B = 2
#     L = 5
#     D = 3
#     Lk = 4

#     decoder = torch.randn(B, L, config.hidden_size)

#     encoder = torch.randn(
#         B,
#         D,
#         Lk,
#         config.hidden_size,
#     )

#     retrieval = torch.rand(B, L, D)

#     out = layer(
#         decoder_hidden_states=decoder,
#         encoder_hidden_states=encoder,
#         retrieval_memory=retrieval,
#         decoder_position_embeddings=build_pos(config, L),
#         encoder_position_embeddings=build_pos(config, Lk),
#     )

#     print(out.shape)

#     assert out.shape == decoder.shape

#     print("OK")


# def test_single_doc():

#     print("=" * 60)
#     print("Single doc")

#     config = build_config()

#     layer = TSRTLayer(config, 0, 1)

#     B = 1
#     L = 4
#     D = 1
#     Lk = 3

#     out = layer(
#         decoder_hidden_states=torch.randn(B, L, config.hidden_size),
#         encoder_hidden_states=torch.randn(
#             B,
#             D,
#             Lk,
#             config.hidden_size,
#         ),
#         retrieval_memory=torch.rand(B, L, D),
#         decoder_position_embeddings=build_pos(config, L),
#         encoder_position_embeddings=build_pos(config, Lk),
#     )

#     print(out.shape)

#     print("OK")


# def test_cache():

#     print("=" * 60)
#     print("Document cache")

#     config = build_config()

#     layer = TSRTLayer(config, 0, 1)

#     cache = TSRTDocumentCache(config=config)

#     B = 1
#     L = 5
#     D = 2
#     Lk = 4

#     decoder = torch.randn(B, L, config.hidden_size)

#     encoder = torch.randn(
#         B,
#         D,
#         Lk,
#         config.hidden_size,
#     )

#     retrieval = torch.rand(B, L, D)

#     layer(
#         decoder_hidden_states=decoder,
#         encoder_hidden_states=encoder,
#         retrieval_memory=retrieval,
#         decoder_position_embeddings=build_pos(config, L),
#         encoder_position_embeddings=build_pos(config, Lk),
#         cross_attn_past_key_values=cache,
#     )

#     assert cache.has_kv(1)

#     k, v = cache.get_kv(1)

#     print(k.shape)
#     print(v.shape)

#     print("OK")


# def test_gradient():

#     print("=" * 60)
#     print("Gradient")

#     config = build_config()

#     layer = TSRTLayer(config, 0, 1)

#     B = 2
#     L = 5
#     D = 2
#     Lk = 3

#     decoder = torch.randn(
#         B,
#         L,
#         config.hidden_size,
#         requires_grad=True,
#     )

#     encoder = torch.randn(
#         B,
#         D,
#         Lk,
#         config.hidden_size,
#         requires_grad=True,
#     )

#     retrieval = torch.rand(
#         B,
#         L,
#         D,
#         requires_grad=True,
#     )

#     out = layer(
#         decoder_hidden_states=decoder,
#         encoder_hidden_states=encoder,
#         retrieval_memory=retrieval,
#         decoder_position_embeddings=build_pos(config, L),
#         encoder_position_embeddings=build_pos(config, Lk),
#     )

#     out.mean().backward()

#     print(decoder.grad.abs().mean())
#     print(encoder.grad.abs().mean())
#     print(retrieval.grad.abs().mean())

#     print("OK")


# def test_mask():

#     print("=" * 60)
#     print("Mask")

#     config = build_config()

#     layer = TSRTLayer(config, 0, 1)

#     B = 1
#     L = 4
#     D = 2
#     Lk = 3

#     self_mask = torch.zeros(B, 1, L, L)

#     cross_mask = torch.zeros(
#         B,
#         1,
#         L,
#         D * Lk,
#     )

#     out = layer(
#         decoder_hidden_states=torch.randn(B, L, config.hidden_size),
#         encoder_hidden_states=torch.randn(
#             B,
#             D,
#             Lk,
#             config.hidden_size,
#         ),
#         retrieval_memory=torch.rand(B, L, D),
#         decoder_position_embeddings=build_pos(config, L),
#         encoder_position_embeddings=build_pos(config, Lk),
#         self_attention_mask=self_mask,
#         cross_attention_mask=cross_mask,
#     )

#     print(out.shape)

#     print("OK")


# def test_train_eval():

#     print("=" * 60)
#     print("Train / Eval")

#     config = build_config()

#     layer = TSRTLayer(config, 0, 1)

#     layer.train()

#     print(layer.training)

#     layer.eval()

#     print(layer.training)

#     print("OK")


# def test_no_nan():

#     print("=" * 60)
#     print("NaN")

#     config = build_config()

#     layer = TSRTLayer(config, 0, 1)

#     B = 2
#     L = 5
#     D = 2
#     Lk = 3

#     out = layer(
#         decoder_hidden_states=torch.randn(B, L, config.hidden_size),
#         encoder_hidden_states=torch.randn(
#             B,
#             D,
#             Lk,
#             config.hidden_size,
#         ),
#         retrieval_memory=torch.rand(B, L, D),
#         decoder_position_embeddings=build_pos(config, L),
#         encoder_position_embeddings=build_pos(config, Lk),
#     )

#     assert torch.isfinite(out).all()

#     print("No NaN")

#     print("OK")


# if __name__ == "__main__":

#     test_forward()

#     test_single_doc()

#     test_cache()

#     test_gradient()

#     test_mask()

#     test_train_eval()

#     test_no_nan()

#     print("=" * 60)
#     print("ALL TESTS PASSED")

# import torch

# from transformers import AutoTokenizer

# from utils.utils import (
#     batch_tokenize_documents,
# )

# from models.tsrt.utils import (
#     prepare_document_attention_mask,
#     prepare_cross_attention_mask,
# )


# if __name__ == "__main__":

#     torch.set_printoptions(
#         linewidth=200,
#         sci_mode=True,
#     )

#     # =====================
#     # TOKENIZER
#     # =====================

#     tokenizer = AutoTokenizer.from_pretrained(
#         "Qwen/Qwen3-1.7B",
#         trust_remote_code=True,
#     )

#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token


#     # =====================
#     # DOCUMENT BATCH
#     # =====================

#     samples = [
#         [
#             "Paris is the capital of France.",
#             "France is located in Europe.",
#         ],
#         [
#             "The Moon is Earth's only natural satellite.",
#         ],
#         [
#             "PyTorch is a deep learning framework.",
#             "Transformers are neural network architectures.",
#             "HotpotQA is a multi-hop question answering dataset.",
#         ],
#     ]


#     # =====================
#     # TOKENIZE
#     # =====================

#     outputs = batch_tokenize_documents(
#         samples=samples,
#         tokenizer=tokenizer,
#         max_length=16,
#     )


#     input_ids = outputs["input_ids"]
#     attention_mask = outputs["attention_mask"]


#     print("\n====================")
#     print("TOKENIZED OUTPUT")
#     print("====================")

#     print("input_ids shape:")
#     print(input_ids.shape)

#     print("attention_mask shape:")
#     print(attention_mask.shape)


#     B, D, L = attention_mask.shape

#     print()
#     print(f"B = {B}")
#     print(f"D = {D}")
#     print(f"L = {L}")


#     print("\nOriginal attention mask")
#     print(attention_mask)


#     # =====================
#     # DOCUMENT SELF ATTENTION MASK
#     # =====================

#     document_mask = prepare_document_attention_mask(
#         attention_mask
#     )


#     print("\n====================")
#     print("DOCUMENT ATTENTION MASK")
#     print("====================")

#     print("shape:")
#     print(document_mask.shape)

#     print(document_mask)


#     # =====================
#     # CROSS ATTENTION MASK
#     # =====================

#     cross_mask = prepare_cross_attention_mask(
#         attention_mask
#     )


#     print("\n====================")
#     print("CROSS ATTENTION MASK")
#     print("====================")

#     print("shape:")
#     print(cross_mask.shape)

#     print(cross_mask)


#     # =====================
#     # CHECK EXPECTED SHAPE
#     # =====================

#     assert document_mask.shape == (
#         B * D,
#         1,
#         1,
#         L,
#     )

#     assert cross_mask.shape == (
#         B,
#         1,
#         1,
#         D * L,
#     )


#     print("\nAll shape checks passed!")

# import torch


# def compute_retrieval_ranking_loss(
#     usefulness_scores: torch.Tensor,
#     usefulness_score_matrix: torch.Tensor,
#     temperature: float = 0.07,
# ):
#     pos_mask = usefulness_score_matrix == 1
#     neg_mask = usefulness_score_matrix == 0

#     has_pos = pos_mask.any(dim=-1)
#     has_neg = neg_mask.any(dim=-1)
#     valid_query = has_pos & has_neg

#     if not valid_query.any():
#         return usefulness_scores.new_zeros(())

#     scores = usefulness_scores[valid_query]
#     pos_mask = pos_mask[valid_query]
#     neg_mask = neg_mask[valid_query]

#     scores = scores / temperature

#     losses = []

#     for query_scores, query_pos_mask, query_neg_mask in zip(scores, pos_mask, neg_mask):

#         pos_scores = query_scores[query_pos_mask]
#         neg_scores = query_scores[query_neg_mask]

#         logits = torch.cat([
#             pos_scores[:, None],
#             neg_scores.expand(pos_scores.size(0), -1),
#         ], dim=1)

#         positive = logits[:, 0]

#         loss = -(positive - torch.logsumexp(logits, dim=1))

#         losses.append(loss.mean())

#     return torch.stack(losses).mean()


# def run_case(name, scores, labels):
#     loss = compute_retrieval_ranking_loss(scores, labels)
#     print(f"{name:35s}: {loss.item():.6f}")


# if __name__ == "__main__":

#     # ==========================================================
#     # Case 1: Positive >> Negative
#     # ==========================================================
#     scores = torch.tensor([
#         [
#             [0.99, -0.99, -0.99]
#         ]
#     ])

#     labels = torch.tensor([
#         [
#             [1, 0, 0]
#         ]
#     ])

#     run_case("Positive >> Negative", scores, labels)

#     # ==========================================================
#     # Case 2: Positive == Negative
#     # ==========================================================
#     scores = torch.tensor([
#         [
#             [0.5, 0.5, 0.5]
#         ]
#     ])

#     labels = torch.tensor([
#         [
#             [1, 0, 0]
#         ]
#     ])

#     run_case("Positive == Negative", scores, labels)

#     # ==========================================================
#     # Case 3: Positive << Negative
#     # ==========================================================
#     scores = torch.tensor([
#         [
#             [-0.9, 0.9, 0.8]
#         ]
#     ])

#     labels = torch.tensor([
#         [
#             [1, 0, 0]
#         ]
#     ])

#     run_case("Positive << Negative", scores, labels)

#     # ==========================================================
#     # Case 4: Multi Positive
#     # ==========================================================
#     scores = torch.tensor([
#         [
#             [0.9, 0.8, 0.1, -0.2]
#         ]
#     ])

#     labels = torch.tensor([
#         [
#             [1, 1, 0, 0]
#         ]
#     ])

#     run_case("Multi Positive", scores, labels)

#     # ==========================================================
#     # Case 5: Skip entire query
#     # ==========================================================
#     scores = torch.tensor([
#         [
#             [0.9, 0.8, 0.1, -0.2],
#             [0.3, 0.4, 0.5, 0.6],
#         ]
#     ])

#     labels = torch.tensor([
#         [
#             [1, 1, 0, 0],
#             [-1, -1, -1, -1],
#         ]
#     ])

#     run_case("Skip Query", scores, labels)

#     # ==========================================================
#     # Case 6a: Skip documents inside a query
#     # ==========================================================
#     scores = torch.tensor([
#         [
#             [0.9, 0.8, 0.1, -0.2, 0.7]
#         ]
#     ])

#     labels = torch.tensor([
#         [
#             [1, -1, 0, -1, 1]
#         ]
#     ])

#     run_case("Skip Docs", scores, labels)

#     # ==========================================================
#     # Case 6b: Remove skipped docs completely
#     # Expected:
#     # Loss phải giống hệt Case 6a
#     # ==========================================================
#     scores = torch.tensor([
#         [
#             [0.9, 0.1, 0.7]
#         ]
#     ])

#     labels = torch.tensor([
#         [
#             [1, 0, 1]
#         ]
#     ])

#     run_case("Skip Docs (Reference)", scores, labels)

# import torch
# import torch.nn.functional as F


# def compute_retrieval_scoring_loss(
#     usefulness_scores: torch.Tensor,
#     usefulness_score_matrix: torch.Tensor,
# ):
#     """
#     usefulness_scores: (B, L, D), values in [-1, 1]
#     usefulness_score_matrix: (B, L, D), {1, 0, -1}
#     """

#     # Normalize to [0, 1]
#     usefulness_scores = (usefulness_scores + 1.0) / 2.0

#     valid_mask = usefulness_score_matrix != -1

#     if not valid_mask.any():
#         return usefulness_scores.new_zeros(())

#     scores = usefulness_scores[valid_mask]
#     targets = usefulness_score_matrix[valid_mask].float()

#     return F.binary_cross_entropy(
#         scores,
#         targets,
#         reduction="mean",
#     )


# def run_case(name, scores, labels):
#     loss = compute_retrieval_scoring_loss(scores, labels)
#     print(f"{name:35s}: {loss.item():.6f}")


# if __name__ == "__main__":

#     # ==========================================================
#     # Case 1: Perfect prediction
#     # ==========================================================
#     scores = torch.tensor([
#         [
#             [1.0, -1.0, 1.0, -1.0]
#         ]
#     ])

#     labels = torch.tensor([
#         [
#             [1, 0, 1, 0]
#         ]
#     ])

#     run_case("Perfect Prediction", scores, labels)

#     # ==========================================================
#     # Case 2: Completely wrong prediction
#     # ==========================================================
#     scores = torch.tensor([
#         [
#             [-1.0, 1.0, -1.0, 1.0]
#         ]
#     ])

#     labels = torch.tensor([
#         [
#             [1, 0, 1, 0]
#         ]
#     ])

#     run_case("Completely Wrong", scores, labels)

#     # ==========================================================
#     # Case 3: Random prediction
#     # ==========================================================
#     scores = torch.tensor([
#         [
#             [0.2, -0.3, 0.7, -0.5]
#         ]
#     ])

#     labels = torch.tensor([
#         [
#             [1, 0, 1, 0]
#         ]
#     ])

#     run_case("Random Prediction", scores, labels)

#     # ==========================================================
#     # Case 4: Skip documents
#     # ==========================================================
#     scores = torch.tensor([
#         [
#             [1.0, 0.5, -1.0, 0.3]
#         ]
#     ])

#     labels = torch.tensor([
#         [
#             [1, -1, 0, -1]
#         ]
#     ])

#     run_case("Skip Docs", scores, labels)

#     # ==========================================================
#     # Case 5: Reference (remove skipped docs)
#     # ==========================================================
#     scores = torch.tensor([
#         [
#             [1.0, -1.0]
#         ]
#     ])

#     labels = torch.tensor([
#         [
#             [1, 0]
#         ]
#     ])

#     run_case("Skip Docs (Reference)", scores, labels)

#     # ==========================================================
#     # Case 6: All skip
#     # ==========================================================
#     scores = torch.tensor([
#         [
#             [0.2, 0.5, -0.3]
#         ]
#     ])

#     labels = torch.tensor([
#         [
#             [-1, -1, -1]
#         ]
#     ])

#     run_case("All Skip", scores, labels)

#     # ==========================================================
#     # Case 7: Gradient check
#     # ==========================================================
#     print("\nGradient Check")

#     # Random scores in [-1, 1]
#     scores = torch.empty(
#         2,
#         3,
#         4,
#     ).uniform_(-1.0, 1.0)

#     scores.requires_grad_()

#     labels = torch.tensor([
#         [
#             [1, 0, 1, -1],
#             [0, 1, 0, -1],
#             [1, 0, -1, -1],
#         ],
#         [
#             [1, 0, 0, 0],
#             [0, 1, -1, -1],
#             [-1, -1, -1, -1],
#         ],
#     ])

#     loss = compute_retrieval_scoring_loss(scores, labels)

#     loss.backward()

#     print("Loss            :", loss.item())
#     print("Gradient exists :", scores.grad is not None)
#     print("Has NaN         :", torch.isnan(scores.grad).any().item())
#     print("Gradient norm   :", scores.grad.norm().item())

#     print("\nGradient:")
#     print(scores.grad)


from transformers import AutoConfig, AutoModelForCausalLM
config = AutoConfig.from_pretrained(
    "models/tsrt",
    trust_remote_code=True,
)

config.torch_dtype = "bfloat16"

model = AutoModelForCausalLM.from_config(
    config,
    trust_remote_code=True,
)