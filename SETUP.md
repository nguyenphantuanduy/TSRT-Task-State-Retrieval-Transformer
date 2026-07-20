# TSRT - Virtual Machine Setup

Hướng dẫn cài đặt môi trường và chạy dự án trên máy ảo Ubuntu.

## 1. Cập nhật hệ thống

```bash
sudo apt update
sudo apt upgrade -y
```

## 2. Cài đặt các gói cần thiết

```bash
sudo apt install -y build-essential
sudo apt install -y python3-dev
sudo apt install -y python3-venv
sudo apt install -y git
```

## 3. Clone repository

```bash
mkdir -p ~/projects
cd ~/projects

git clone https://github.com/nguyenphantuanduy/TSRT-Task-State-Retrieval-Transformer.git
cd TSRT-Task-State-Retrieval-Transformer
```

## 4. Tạo môi trường ảo

```bash
cd backend

python3 -m venv .venv
source .venv/bin/activate
```

## 5. Nâng cấp pip

```bash
pip install --upgrade pip
```

## 6. Cài đặt PyTorch

```bash
pip install torch torchvision torchaudio
```

> **Lưu ý:** Nếu máy ảo sử dụng GPU NVIDIA với CUDA, hãy cài đặt phiên bản PyTorch phù hợp với phiên bản CUDA của hệ thống.

## 7. Cài đặt các thư viện còn lại

```bash
pip install -r requirements.txt
```

## Hoàn tất

Sau khi hoàn thành các bước trên, môi trường đã sẵn sàng để chạy và phát triển dự án TSRT.
