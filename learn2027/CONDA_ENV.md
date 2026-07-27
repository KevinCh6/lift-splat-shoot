# Conda 环境说明

已创建本地学习环境：

```bash
conda activate lss2027
```

Python：

```text
Python 3.10.20
```

核心包：

```text
torch 2.5.1
torchvision 0.20.1
jupyter / jupyterlab
matplotlib
efficientnet_pytorch
fire
pyquaternion
tqdm
```

Jupyter kernel：

```text
Python (lss2027)
```

## 启动 notebook

在仓库根目录运行：

```bash
conda activate lss2027
jupyter lab
```

然后打开：

```text
learn2027/lss_step_through.ipynb
```

在右上角 kernel 选择：

```text
Python (lss2027)
```

如果 VS Code/Jupyter 的当前工作目录不是仓库根目录，可以在启动前设置：

```bash
export LSS_REPO_ROOT=/path/to/lift-splat-shoot
```

不要把真实个人路径写进 notebook 源码。需要文件配置时，参考：

```text
learn2027/local_config.example.py
```

## 当前学习范围

这个环境已经可以用于：

```text
系统学习 learn2027/src/models.py
运行 PyTorch 张量实验
运行 create_frustum() 相关可视化
逐行理解 LSS 模型结构
```

暂时不强求本地跑：

```text
nuScenes 数据读取
完整 train.py 训练
lidar_check 可视化
```

这些需要 `nuscenes-devkit`、OpenCV、TensorBoardX 等数据/训练依赖。它们后面在租 GPU 或接真实 mini 数据时再补更合适。

## 学习副本改动

为了让新手阶段可以专注读 `models.py`，只对学习副本做了两个轻量调整：

```text
learn2027/src/__init__.py
learn2027/src/tools.py
```

目的：

```text
from src.models import LiftSplatShoot
```

可以直接运行，不会因为暂时没安装 nuScenes/OpenCV 而失败。

根目录原始源码 `src/` 没有改动。
