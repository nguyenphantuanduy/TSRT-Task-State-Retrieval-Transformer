import os
from huggingface_hub import HfApi, login

# =====================================================
# CONFIGURATION
# =====================================================
REPO_ID = "tsrt-lab/TSRT-Qwen3-1.7B"
LOCAL_FOLDER = "./best_model"

# Danh sách các đuôi file liên quan đến weight & config của mô hình
ALLOWED_EXTENSIONS = (
    ".safetensors",
    ".bin",
    ".json",
    ".pth",
    ".pt",
)


def upload_weights_only():
    """Chỉ upload các file weight và configuration của model lên Hugging Face Hub."""
    # 1. Kiểm tra thư mục nguồn
    if not os.path.exists(LOCAL_FOLDER):
        raise FileNotFoundError(
            f"Thư mục '{LOCAL_FOLDER}' không tồn tại. "
            "Hãy đảm bảo bạn đã hoàn thành quá trình huấn luyện/lưu mô hình."
        )

    # 2. Đăng nhập Hugging Face (nếu chưa lưu token trong môi trường)
    # LƯU Ý: Bạn có thể truyền thẳng token vào login(token="hf_xxx") 
    # hoặc thiết lập biến môi trường HUGGING_FACE_HUB_TOKEN trước khi chạy.
    login()

    api = HfApi()

    # 3. Lọc danh sách file chỉ lấy Weights và Config
    all_files = os.listdir(LOCAL_FOLDER)
    files_to_upload = [
        f for f in all_files 
        if f.endswith(ALLOWED_EXTENSIONS) and not f.startswith("tokenizer")
    ]

    if not files_to_upload:
        print("Không tìm thấy file weight/config nào hợp lệ trong thư mục!")
        return

    print(f"Phát hiện {len(files_to_upload)} file cần upload:")
    for file_name in files_to_upload:
        print(f" - {file_name}")

    # 4. Thực hiện upload từng file lên repo
    print("\nBắt đầu upload lên Hugging Face Hub...")
    for file_name in files_to_upload:
        local_filepath = os.path.join(LOCAL_FOLDER, file_name)
        
        print(f"Đang tải lên: {file_name} ...")
        api.upload_file(
            path_or_fileobj=local_filepath,
            path_in_repo=file_name,
            repo_id=REPO_ID,
            repo_type="model",
            commit_message=f"Update model weight: {file_name}",
        )

    print(f"\n Hoàn tất upload! Kiểm tra tại: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    upload_weights_only()