# Environment Check

Checked on 2026-07-15.

## System

- OS: Windows 11
- System Python: 3.13.3 at `python`
- Conda: 25.7.0 at `D:\miniconda`
- D drive free space: about 562 GB

## GPU

- Detected GPU: Intel Arc 130T GPU (16GB)
- Driver version: 32.0.101.6554
- NVIDIA CUDA: not detected
- `nvidia-smi`: not available in PATH

## Conda environments

- `dl-py312`: Python 3.12.13, PyTorch 2.13.0+cpu
- `dl-xpu-py312`: Python 3.12.13, PyTorch 2.13.0+xpu
- `openvino-py312`: Python 3.12.13, OpenVINO 2026.2.1

## Verified commands

```powershell
conda run -n dl-xpu-py312 python src/check_env.py
conda run -n dl-xpu-py312 python src/train_smoke.py
```

Both commands completed successfully on the `xpu` device.

## Recommendation

Use `dl-xpu-py312` for PyTorch projects on this machine.
Use `openvino-py312` when you specifically want OpenVINO inference or deployment tests.
