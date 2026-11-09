# dl-starter

面向 Intel Arc XPU 的可复用图像分类训练工程。它既能用合成数据离线验证完整训练链路，也能下载 CIFAR-10 进行正式训练。

## 已有能力

- 自动选择 Intel XPU、NVIDIA CUDA 或 CPU
- JSON 实验配置与参数校验
- TinyResNet 和 CIFAR-10 版 ResNet-18
- 自动混合精度、梯度缩放与梯度裁剪
- AdamW 优化器与余弦学习率调度
- `best.pt` / `last.pt` 原子保存与断点续训
- CSV 指标、JSON 配置和训练摘要归档
- 独立评估、单图 Top-K 推理和单元测试

## 环境准备

```powershell
Set-Location <repo>
conda activate dl-xpu-py312
python -m pip install -e . --no-deps
```

## 快速验证

这条命令不下载数据，约几十秒完成两轮 XPU 训练：

```powershell
python -m dl_starter.train --config configs/smoke.json
```

训练产物保存在 `outputs/runs/<运行名称-时间>/`：

```text
best.pt       验证集表现最好的检查点
last.pt       最后一轮检查点，可用于续训
metrics.csv   每轮损失、准确率和学习率
config.json   本次运行的完整配置快照
summary.json  最佳结果与设备信息
```

## 正式训练 CIFAR-10

首次运行会自动下载 CIFAR-10：

```powershell
python -m dl_starter.train --config configs/cifar10-resnet18.json
```

可以编辑 `configs/cifar10-resnet18.json` 调整 epoch、batch size、模型和学习率。

## 评估、续训与推理

```powershell
python -m dl_starter.evaluate outputs/runs/<run>/best.pt
python -m dl_starter.train --config configs/cifar10-resnet18.json --resume outputs/runs/<run>/last.pt
python -m dl_starter.predict example.jpg --checkpoint outputs/runs/<run>/best.pt --top-k 3
```

## 测试与环境检查

```powershell
python -m unittest discover -s tests -v
python src/check_env.py
python src/train_smoke.py
```

## 目录结构

```text
configs/          实验配置
data/             数据集缓存（Git 忽略）
outputs/runs/     检查点和训练记录（Git 忽略）
src/dl_starter/   可复用训练包
tests/            核心单元测试
```
