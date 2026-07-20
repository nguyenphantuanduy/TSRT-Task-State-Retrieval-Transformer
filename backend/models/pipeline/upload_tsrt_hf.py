import os
from huggingface_hub import HfApi, upload_folder


# ==========================================
# Config
# ==========================================

LOCAL_MODEL_PATH = "./models/tsrt"

HF_REPO = "nguyenphantuanduy/TSRT-Qwen3-1.7B"



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



def check_auto_map():

    import json


    config_path = os.path.join(
        LOCAL_MODEL_PATH,
        "config.json"
    )


    with open(config_path) as f:
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
        "Upload TSRT Qwen3-1.7B model with remote code",
    )


    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print(
        f"https://huggingface.co/{HF_REPO}"
    )



if __name__ == "__main__":

    check_required_files()

    check_auto_map()

    upload()