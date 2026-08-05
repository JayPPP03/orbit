# ORBIT Environment Setup Guide

## Hardware Requirements
- **Mac:** Apple Silicon (M1/M2/M3/M5), 16GB RAM minimum
- **Storage:** ~10GB free space
- **OS:** macOS 13+ (Ventura or later)

---

## Phase 1: System Tools

### 1. Install Homebrew
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Install Git
```bash
brew install git
git config --global user.name "Jay Patel"
git config --global user.email "pjay205335@gmail.com"
```

### 3. Install Docker Desktop
Download from: https://docs.docker.com/desktop/setup/install/mac-install/
- Choose **Apple Silicon** version
- Allocate **8GB RAM** in Docker Preferences → Resources

### 4. Install Python 3.11+
```bash
brew install python@3.11
brew install pyenv
```

---

## Phase 2: Python Environment

### 5. Install Poetry
```bash
brew install poetry
```

### 6. Create Project & Virtual Environment
```bash
mkdir ~/orbit && cd ~/orbit
poetry init --name orbit --python "^3.11"
poetry shell
```

---

## Phase 3: Core Dependencies

### Data & Processing
```bash
poetry add polars duckdb pyarrow
```

### Machine Learning
```bash
poetry add xgboost scikit-learn optuna
```

### Deep Learning (CPU-optimized for M5)
```bash
poetry add torch torchvision torchaudio
```
*PyTorch auto-detects MPS (Metal Performance Shaders) on Apple Silicon.*

### API & Serving
```bash
poetry add fastapi uvicorn redis pydantic
```

### Experiment Tracking
```bash
poetry add mlflow
```

### Causal ML
```bash
poetry add econml
```

---

## Phase 4: LLM / Explanation Engine

### Local LLM Inference (Apple Silicon GPU)
```bash
CMAKE_ARGS="-DLLAMA_METAL=on" poetry add llama-cpp-python
```

### QLoRA Fine-Tuning
```bash
poetry add unsloth
```

### HuggingFace Tools
```bash
poetry add transformers peft accelerate bitsandbytes
```

---

## Phase 5: MLOps & Development

### Data Versioning
```bash
brew install dvc
```

### Code Quality & Testing
```bash
poetry add --group dev ruff black pytest pytest-cov locust
```

### Jupyter (for EDA)
```bash
poetry add --group dev jupyter
```

---

## Phase 6: Docker Services

These run as containers — do NOT install natively:

```bash
# Start all services
docker-compose up -d redis duckdb prometheus grafana
```

| Service | Port | Purpose |
|---------|------|---------|
| Redis | 6379 | Feature store cache |
| Prometheus | 9090 | Metrics collection |
| Grafana | 3000 | Dashboards |
| DuckDB | - | Embedded OLAP |

---

## Verification

Run this script to verify everything:

```bash
python -c "import polars; print('Polars:', polars.__version__)"
python -c "import duckdb; print('DuckDB:', duckdb.__version__)"
python -c "import torch; print('PyTorch MPS:', torch.backends.mps.is_available())"
python -c "import xgboost; print('XGBoost:', xgboost.__version__)"
python -c "import mlflow; print('MLflow:', mlflow.__version__)"
python -c "from llama_cpp import Llama; print('llama-cpp: OK')"
docker --version
docker-compose --version
git --version
poetry --version
dvc --version
```

---

## Disk Space Budget

| Category | Size |
|----------|------|
| Docker Desktop | ~2GB |
| Python + packages | ~3-4GB |
| Docker images | ~1GB |
| Datasets | ~500MB |
| Models | ~2GB |
| **Total** | **~8-9GB** |

---

## Next Steps

After setup is complete, proceed to **Week 1, Day 1: Data Ingestion**.

See the [30-Day Roadmap in ORBIT_Project_Blueprint.md](ORBIT_Project_Blueprint.md#17-30-day-build-roadmap) for day-by-day execution.
