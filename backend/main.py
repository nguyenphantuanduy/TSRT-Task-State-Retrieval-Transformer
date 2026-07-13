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

start_time = time.time()

dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor",
)

print(
    f"Download + load: "
    f"{time.time() - start_time:.2f}s"
)

count = 0

start_iter = time.time()

for sample in dataset["train"]:
    count += 1

    if count % 10000 == 0:
        print(
            f"{count} samples | "
            f"{count/(time.time()-start_iter):.2f} samples/s"
        )

iter_time = time.time() - start_iter

print(f"Train samples: {count}")
print(f"Iteration time: {iter_time:.2f}s")
print(f"Samples/sec: {count/iter_time:.2f}")