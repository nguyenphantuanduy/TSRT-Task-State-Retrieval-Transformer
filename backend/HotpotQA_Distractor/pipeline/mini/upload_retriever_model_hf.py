import os

from huggingface_hub import HfApi, login


# =====================================================
# CONFIGURATION
# =====================================================

REPO_ID = "tsrt-lab/TSRT-Qwen3-1.7B"

LOCAL_FOLDER = "./best_mini_retriever"

REPO_SUBFOLDER = "mini/retriever"


# =====================================================
# ALLOWED FILE EXTENSIONS
# =====================================================

ALLOWED_EXTENSIONS = (
    ".safetensors",
    ".bin",
    ".json",
    ".pth",
    ".pt",
)


def upload_mini_retriever():
    """
    Upload Mini TSRT Retriever weights and configuration
    to the `mini/retriever/` subfolder of the
    Hugging Face Hub repository.
    """

    # =================================================
    # 1. CHECK SOURCE FOLDER
    # =================================================

    if not os.path.exists(LOCAL_FOLDER):
        raise FileNotFoundError(
            f"Thư mục '{LOCAL_FOLDER}' không tồn tại. "
            "Hãy đảm bảo bạn đã hoàn thành quá trình "
            "huấn luyện/lưu Mini Retriever."
        )

    # =================================================
    # 2. LOGIN
    # =================================================

    login()

    api = HfApi()

    # =================================================
    # 3. FIND WEIGHT / CONFIG FILES
    # =================================================

    all_files = os.listdir(LOCAL_FOLDER)

    files_to_upload = [
        file_name
        for file_name in all_files
        if file_name.endswith(ALLOWED_EXTENSIONS)
        and not file_name.startswith("tokenizer")
    ]

    if not files_to_upload:
        print(
            "Không tìm thấy file weight/config nào hợp lệ "
            "trong thư mục Mini Retriever!"
        )
        return

    print(
        f"Phát hiện {len(files_to_upload)} file cần upload:"
    )

    for file_name in files_to_upload:
        print(f" - {file_name}")

    # =================================================
    # 4. UPLOAD
    # =================================================

    print(
        "\nBắt đầu upload Mini Retriever lên "
        "Hugging Face Hub..."
    )

    for file_name in files_to_upload:

        local_filepath = os.path.join(
            LOCAL_FOLDER,
            file_name,
        )

        repo_filepath = os.path.join(
            REPO_SUBFOLDER,
            file_name,
        ).replace("\\", "/")

        print(
            f"Đang tải lên: "
            f"{repo_filepath} ..."
        )

        api.upload_file(
            path_or_fileobj=local_filepath,
            path_in_repo=repo_filepath,
            repo_id=REPO_ID,
            repo_type="model",
            commit_message=(
                f"Update Mini Retriever: {file_name}"
            ),
        )

    # =================================================
    # 5. DONE
    # =================================================

    print(
        "\nHoàn tất upload Mini Retriever!"
    )

    print(
        f"Mini Retriever nằm tại: "
        f"https://huggingface.co/{REPO_ID}/tree/main/"
        f"{REPO_SUBFOLDER}"
    )


if __name__ == "__main__":
    upload_mini_retriever()
