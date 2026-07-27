# LSS 完整复现计划

目标：在 1000 元人民币以内租用云 GPU，完整复现 Lift-Splat-Shoot 的 nuScenes BEV vehicle segmentation 结果，并产出可检查的训练日志、模型权重、验证 IoU、可视化样例和复现记录。

## 0. 复现定义

本计划里的“完整复现”包含：

- 使用 nuScenes `v1.0-trainval` 数据集训练 LSS vehicle BEV segmentation。
- 使用仓库默认核心设置：`final_dim=(128, 352)`、`dbound=[4.0, 45.0, 1.0]`、`x/ybound=[-50, 50, 0.5]`。
- 在 validation split 上运行 `eval_model_iou`。
- 保存训练 checkpoint、TensorBoard 日志、最终 IoU、若干预测可视化结果。
- 复现结果目标：优先达到仓库 README 附近的 vehicle IoU 水平；若因为训练时长不足，至少给出训练曲线、已达到 IoU、剩余差距和原因分析。

## 1. 推荐机器

首选：

```text
GPU: RTX 4090 24GB
RAM: >= 60GB
CPU: >= 8 cores
Disk: >= 600GB SSD/data disk
Billing: 按量计费，先短测再长跑
```

备选：

```text
GPU: RTX 3090 24GB
RAM: >= 60GB
Disk: >= 600GB
```

不建议用于完整复现：

```text
8GB / 10GB / 12GB GPU
```

这些卡可以学习和跑 mini，但完整 trainval 会更容易被 batch size、速度和 OOM 拖慢。

## 2. 预算

预估按 RTX 4090 约 1.5-2 元/小时计算。

```text
环境验证:       2-6 小时     5-15 元
mini 跑通:      2-8 小时     5-20 元
trainval 训练:  80-250 小时  120-500 元
eval/viz:       5-20 小时    10-40 元
返工缓冲:       100-250 元
总预算:         300-900 元
```

1000 元以内可行。关键是不要把 GPU 时间浪费在下载、解压、环境安装和路径调试上。

## 3. 本地准备

在租 GPU 前，本地先完成：

- 注册 nuScenes，确认能下载 `v1.0-trainval`、`v1.0-mini`、map expansion。
- 准备云平台账号和充值方式。
- 明确要保存的云盘/持久化存储机制。
- 阅读当前仓库入口：
  - `main.py`
  - `src/train.py`
  - `src/explore.py`
  - `src/models.py`
  - `src/data.py`

本地学习笔记继续记录到：

```text
learn2027/LEARNING_HISTORY.md
```

## 4. 云端目录规划

建议云端使用：

```text
/root/autodl-tmp/lift-splat-shoot
/root/autodl-tmp/nuscenes
/root/autodl-tmp/runs/lss-trainval
/root/autodl-tmp/outputs/lss-viz
```

nuScenes 目标结构：

```text
nuscenes/
  samples/
  sweeps/
  maps/
  v1.0-mini/
  v1.0-trainval/
```

注意：`samples` 和 `sweeps` 很大，必须放在数据盘或持久化盘，避免实例释放后丢失。

## 5. 阶段 A：环境冒烟测试

目标：确认代码能 import，GPU 可用，依赖版本可跑。

安装依赖：

```bash
pip install nuscenes-devkit tensorboardX efficientnet_pytorch==0.7.0 fire
```

检查：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -c "from src.models import compile_model; print('model import ok')"
```

验收：

```text
torch.cuda.is_available() == True
model import ok
```

预算控制：如果 1 小时内环境没跑通，停止实例，先整理错误，不要硬烧 GPU 时间。

## 6. 阶段 B：mini 数据集跑通

目标：用 `v1.0-mini` 证明数据路径、nuScenes devkit、模型 forward、loss、eval 都正常。

先做数据可视化检查：

```bash
python main.py lidar_check mini --dataroot=/root/autodl-tmp/nuscenes --viz_train=False
```

再做短训练：

```bash
python main.py train mini --dataroot=/root/autodl-tmp/nuscenes --logdir=/root/autodl-tmp/runs/lss-mini --gpuid=0
```

观察：

```bash
tensorboard --logdir=/root/autodl-tmp/runs --bind_all
```

验收：

```text
能成功读取 mini 数据。
训练 loss 开始下降。
能保存 model*.pt。
能运行一次 validation。
```

预算控制：mini 阶段只用于通路验证，不追最终指标。

## 7. 阶段 C：trainval 正式训练

目标：在完整 trainval 上训练 vehicle segmentation。

默认命令：

```bash
python main.py train trainval \
  --dataroot=/root/autodl-tmp/nuscenes \
  --logdir=/root/autodl-tmp/runs/lss-trainval \
  --gpuid=0
```

默认关键参数来自 `src/train.py`：

```text
bsz=4
ncams=5
final_dim=(128, 352)
dbound=[4.0, 45.0, 1.0]
lr=1e-3
```

如果 OOM，按顺序处理：

1. 把 `bsz` 从 4 降到 2。
2. 把 `nworkers` 从 10 降到 4 或 2。
3. 确认没有多个训练进程占 GPU。
4. 保持 `final_dim`、`dbound`、`x/ybound` 不动，避免复现目标漂移。

可用命令示例：

```bash
python main.py train trainval \
  --dataroot=/root/autodl-tmp/nuscenes \
  --logdir=/root/autodl-tmp/runs/lss-trainval-bsz2 \
  --gpuid=0 \
  --bsz=2 \
  --nworkers=4
```

训练期间每隔一段时间记录：

```text
step
train/loss
train/iou
val/loss
val/iou
checkpoint path
GPU 型号
batch size
训练累计小时
累计费用
```

## 8. 阶段 D：评估

目标：用保存的 checkpoint 在 validation split 上报告 IoU。

命令：

```bash
python main.py eval_model_iou trainval \
  --modelf=/root/autodl-tmp/runs/lss-trainval/modelXXXX.pt \
  --dataroot=/root/autodl-tmp/nuscenes
```

验收：

```text
记录最终 checkpoint。
记录 validation IoU。
和 README 中 reported/repository vehicle IoU 做比较。
```

## 9. 阶段 E：可视化

目标：保存模型预测图，检查不是只追数字。

命令：

```bash
python main.py viz_model_preds trainval \
  --modelf=/root/autodl-tmp/runs/lss-trainval/modelXXXX.pt \
  --dataroot=/root/autodl-tmp/nuscenes \
  --map_folder=/root/autodl-tmp/nuscenes
```

验收：

```text
保存若干预测可视化。
挑选 5-10 个样例写观察：好在哪里、错在哪里。
```

## 10. 产出物

最终至少保存：

```text
runs/lss-trainval/
  events.out.tfevents.*
  modelXXXX.pt

outputs/lss-viz/
  prediction samples

learn2027/
  LSS_REPRODUCTION_REPORT.md
```

报告建议包含：

```text
机器配置
数据版本
训练命令
超参数
训练时长
费用
最终 IoU
README/论文指标对比
可视化样例
遇到的问题和解决方式
```

## 11. 时间表

最快节奏：

```text
Day 1: 云端环境、mini 跑通、确认数据路径
Day 2-4: trainval 训练
Day 5: eval、viz、写报告
```

稳妥节奏：

```text
Day 1: 本地准备、云端环境
Day 2: mini 跑通、修路径和依赖
Day 3-6: trainval 训练
Day 7: eval、viz、报告、代码理解复盘
```

## 12. 学习同步路线

训练跑着的时候不要干等，按这个顺序吃代码：

1. `src/models.py::CamEncode.get_depth_feat`
2. `src/models.py::LiftSplatShoot.create_frustum`
3. `src/models.py::LiftSplatShoot.get_geometry`
4. `src/models.py::LiftSplatShoot.voxel_pooling`
5. `src/data.py::NuscData.get_image_data`
6. `src/data.py::NuscData.get_binimg`
7. `src/train.py::train`
8. `src/tools.py::QuickCumsum`

每学完一个函数，在 `learn2027/LEARNING_HISTORY.md` 里写：

```text
输入 shape
输出 shape
数学含义
对应论文位置
我还不懂的问题
```

## 13. 停止条件

达到以下任一条件即可停止正式训练：

- validation IoU 接近仓库 README 的复现指标。
- train/val 曲线已经明显收敛，继续训练收益很小。
- 预算达到 800 元，需要保留 200 元做 eval、viz 和返工。
- 出现持续 OOM 或数据错误，且 2 小时内无法解决。

停止训练后必须先保存：

```text
checkpoint
TensorBoard logs
训练命令
当前 git commit/hash 或代码压缩包
```

## 14. 风险清单

常见风险：

- 数据没放在持久化盘，释放实例后丢失。
- 下载/解压占用 GPU 计费时间。
- `nworkers=10` 导致系统内存吃紧。
- `bsz=4` 在部分 24GB 机器上仍可能 OOM。
- map folder 路径错误导致可视化失败。
- 训练中断后没有 checkpoint。

处理原则：

```text
先 mini，后 trainval。
先按量短测，后长跑。
先保日志和 checkpoint，再释放机器。
优先保持论文/仓库核心参数不变。
```
