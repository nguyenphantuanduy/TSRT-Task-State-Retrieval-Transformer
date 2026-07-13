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

import time
from datasets import load_dataset

DATASET_REPO = "nguyenphantuanduy/temp-dataset"

# ==========================================================
# DOWNLOAD
# ==========================================================

start = time.time()

dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor",
)

download_time = time.time() - start

print(
    f"Download + load: "
    f"{download_time:.2f}s"
)

print(dataset)

# ==========================================================
# UPLOAD
# ==========================================================

start = time.time()

dataset.push_to_hub(
    DATASET_REPO,
    private=False,
)

upload_time = time.time() - start

print(
    f"Upload time: "
    f"{upload_time:.2f}s"
)

print(
    f"Total time: "
    f"{download_time + upload_time:.2f}s"
)