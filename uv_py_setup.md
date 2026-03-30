# Parking Lot Prediction -- Setup Guide

## Prerequisites

- Python 3.12+ managed by [uv](https://docs.astral.sh/uv/)
- NVIDIA GPU with CUDA support (optional but recommended)
- PKLot dataset already extracted at `../PKLot/`

## 1. Install packages (GPU-accelerated)

From the repo root (`BSAN_765_AI_for_Business/`), add the required dependencies with CUDA-enabled PyTorch:

```bash
# Add the PyTorch CUDA index and core dependencies
uv add ultralytics pillow notebook torch torchvision
```

Then configure `pyproject.toml` to pull PyTorch from the CUDA 12.8 index instead of CPU-only PyPI. Add these sections:

```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
torchvision = { index = "pytorch-cu128" }
```

Then sync:

```bash
uv sync
```

### Verify GPU is detected

```bash
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'CPU only')"
```

## 2. Run the notebook

Open `main.ipynb` in VS Code and select the **bsan-765-ai-for-business (Python 3.12)** kernel.

Run all cells. The notebook will:
1. Skip zip extraction (data is already at `../PKLot/`)
2. Merge all lot/weather folders into `data/total-content/`
3. Convert XML labels to YOLO txt format
4. Split into train/test/val (70/15/15)
5. Fine-tune YOLOv8s for the configured number of epochs
6. Run prediction on a random test image

Training runs are saved to `runs/detect/train/` inside this directory.

## 3. Export notebook as HTML

After running the notebook, export it for sharing:

```bash
uv run jupyter nbconvert --to html main.ipynb --output main_output.html
```

Or to export with a custom name (e.g., with date):

```bash
uv run jupyter nbconvert --to html main.ipynb --output main_29_march.html
```

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'ultralytics'` | Run `uv add ultralytics` then `uv sync` |
| `GPU_mem: 0G` in training output | PyTorch is CPU-only. Reconfigure with the CUDA index (see step 1) |
| `FileExistsError` on XML rename | The notebook was run before. The `shutil.move` version of cell-12 handles this |
| `milvus-lite` install error on Windows | Add `; sys_platform != 'win32'` to the milvus-lite dependency in `pyproject.toml` |
| VS Code asks to install pip/notebook | Run `uv add notebook` instead of clicking Install |
