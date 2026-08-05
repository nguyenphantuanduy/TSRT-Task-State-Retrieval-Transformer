import json
import os

from huggingface_hub import hf_hub_download

# ==========================================================
# CONFIG
# ==========================================================

MODEL_ID = "Qwen/Qwen3-1.7B"
OUTPUT_DIR = "models/tsrt"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "config.json"
)

# ==========================================================
# DOWNLOAD CONFIG
# ==========================================================

config_path = hf_hub_download(
    repo_id=MODEL_ID,
    filename="config.json"
)

with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

# ==========================================================
# MODIFY CONFIG
# ==========================================================

# TSRT architecture
config["num_hidden_layers"] = 42

config["num_encoder_layers"] = 14
config["num_decoder_layers"] = 7
config["num_tsrt_layers"] = 21
config["retrieval_embedding_size"] = 1024

# Retrieval bias
config["retrieval_bias_gamma"] = 4.0

# Custom model metadata
config["model_type"] = "tsrt"

config["architectures"] = [
    "TSRTForCausalLM"
]

config["auto_map"] = {
    "AutoConfig": "configuration_tsrt.TSRTConfig",
    "AutoModel": "modeling_tsrt.TSRTModel",
    "AutoModelForCausalLM": "modeling_tsrt.TSRTForCausalLM"
}

# ==========================================================
# SAVE
# ==========================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        config,
        f,
        indent=2,
        ensure_ascii=False
    )

print("=" * 80)
print("TSRT config saved")
print(f"Source : {MODEL_ID}")
print(f"Output : {OUTPUT_FILE}")
print("=" * 80)

print(json.dumps(
    {
        "model_type": config["model_type"],
        "num_hidden_layers": config["num_hidden_layers"],
        "num_encoder_layers": config["num_encoder_layers"],
        "num_decoder_layers": config["num_decoder_layers"],
        "num_tsrt_layers": config["num_tsrt_layers"],
    },
    indent=2
))