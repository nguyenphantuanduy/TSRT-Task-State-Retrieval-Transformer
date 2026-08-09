import os
import json

from huggingface_hub import HfApi, upload_folder
from transformers import AutoTokenizer

# ==========================================
# Config
# ==========================================

LOCAL_MODEL_PATH = "./models/tsrt"

HF_REPO = "tsrt-lab/TSRT-Qwen3-1.7B"

TOKENIZER_NAME = "Qwen/Qwen3-1.7B"


# ==========================================
# Check files
# ==========================================

def check_required_files():

    print("=" * 60)
    print("Checking files")
    print("=" * 60)

    required = [
        "config.json",
        "model.safetensors",
        "modeling_tsrt.py",
        "configuration_tsrt.py",
    ]

    for f in required:

        path = os.path.join(
            LOCAL_MODEL_PATH,
            f
        )

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Missing: {path}"
            )

        print(
            f"[OK] {f}"
        )


# ==========================================
# Check retriever files
# ==========================================

def check_retriever_files():

    print("=" * 60)
    print("Checking retriever files")
    print("=" * 60)

    retriever_path = os.path.join(
        LOCAL_MODEL_PATH,
        "retriever",
    )

    if not os.path.exists(retriever_path):

        raise FileNotFoundError(
            f"Missing retriever folder: {retriever_path}"
        )

    required = [
        "config.json",
        "model.safetensors",
    ]

    for f in required:

        path = os.path.join(
            retriever_path,
            f
        )

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Missing retriever file: {path}"
            )

        print(
            f"[OK] retriever/{f}"
        )


# ==========================================
# Add auto map
# ==========================================

def check_auto_map():

    config_path = os.path.join(
        LOCAL_MODEL_PATH,
        "config.json"
    )

    with open(config_path, "r") as f:
        config = json.load(f)


    if "auto_map" not in config:

        print(
            "[WARNING] auto_map missing, adding..."
        )


        config["auto_map"] = {

            "AutoConfig":
                "configuration_tsrt.TSRTConfig",

            "AutoModelForCausalLM":
                "modeling_tsrt.TSRTForCausalLM",
        }


        with open(config_path, "w") as f:

            json.dump(
                config,
                f,
                indent=2
            )


        print(
            "[OK] auto_map added"
        )


    else:

        print(
            "[OK] auto_map exists"
        )


# ==========================================
# Add tokenizer
# ==========================================

def save_tokenizer():

    print("=" * 60)
    print("Loading Qwen3 tokenizer")
    print("=" * 60)


    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        trust_remote_code=True,
    )


    tokenizer.save_pretrained(
        LOCAL_MODEL_PATH
    )


    print(
        "[OK] tokenizer saved"
    )


# ==========================================
# Upload
# ==========================================

def upload():

    print("=" * 60)
    print("Uploading model")
    print("=" * 60)


    api = HfApi()


    api.create_repo(
        repo_id=HF_REPO,
        exist_ok=True,
    )


    upload_folder(
        folder_path=LOCAL_MODEL_PATH,
        repo_id=HF_REPO,
        commit_message=
        "Upload TSRT Qwen3-1.7B with tokenizer and remote code",
    )


    print("=" * 60)
    print("DONE")
    print("=" * 60)


    print(
        f"https://huggingface.co/{HF_REPO}"
    )


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":

    check_required_files()

    check_retriever_files()

    check_auto_map()

    save_tokenizer()

    upload()
