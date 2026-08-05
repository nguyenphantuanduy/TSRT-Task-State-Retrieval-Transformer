# ==============================

# Update + Install system dependencies

# ==============================

sudo apt update && \
sudo apt upgrade -y && \
sudo apt install -y \
 build-essential \
 git \
 software-properties-common \
 curl && \
sudo add-apt-repository ppa:deadsnakes/ppa -y && \
sudo apt update && \
sudo apt install -y \
 python3.13 \
 python3.13-dev \
 python3.13-venv

# ==============================

# Verify Python

# ==============================

python3.13 --version

# ==============================

# Clone project

# ==============================

mkdir -p ~/projects && \
cd ~/projects && \
git clone https://github.com/nguyenphantuanduy/TSRT-Task-State-Retrieval-Transformer.git

cd ~/projects/TSRT-Task-State-Retrieval-Transformer/backend

# ==============================

# Create Python 3.13 environment

# ==============================

python3.13 -m venv .venv

source .venv/bin/activate

# ==============================

# Install Python dependencies

# ==============================

python --version

pip install --upgrade pip setuptools wheel

pip install torch torchvision torchaudio

pip install -r requirements.txt
