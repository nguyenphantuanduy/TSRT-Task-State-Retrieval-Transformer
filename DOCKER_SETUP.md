# TSRT - Docker Deployment & Testing

This project uses Docker and Docker Compose profiles to provide two distinct environments:
1. **GPU Environment**: For full model training and heavy inference (requires an NVIDIA GPU).
2. **CPU Environment**: A lightweight image for local debugging, unit testing, and data pipeline verification.

## 1. Prerequisites
- [Docker](https://docs.docker.com/get-docker/) installed.
- [Docker Compose](https://docs.docker.com/compose/install/) (v2+ recommended, accessed via `docker compose`).
- *Optional (For GPU only)*: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed and configured so Docker can access your GPU.

## 2. Running the Environments

The project uses Docker Compose profiles (`cpu` and `gpu`) to separate the setups. 

### Start the CPU Environment (Lightweight Debugging)
Use this profile if you don't have a GPU or just want to run data processing tests without downloading gigabytes of CUDA binaries.

```bash
docker compose --profile cpu up -d --build
```

### Start the GPU Environment (Full Training/Inference)
Use this profile for the complete TSRT model lifecycle.

```bash
docker compose --profile gpu up -d --build
```

## 3. Executing Commands Inside the Container

Once your chosen container is running in the background (`-d`), you can execute tests or open a shell inside it.

**Run a specific test script:**
```bash
# On CPU container
docker compose --profile cpu exec tsrt-cpu python backend/test/test_retrieval_mem.py

# On GPU container
docker compose --profile gpu exec tsrt-gpu python backend/test/test_retrieval_mem.py
```

**Open an interactive bash shell inside the container:**
```bash
# On CPU container
docker compose --profile cpu exec tsrt-cpu bash

# On GPU container
docker compose --profile gpu exec tsrt-gpu bash
```

## 4. Stopping the Environments

To stop the running containers and remove them:

```bash
# Stop CPU container
docker compose --profile cpu down

# Stop GPU container
docker compose --profile gpu down
```
