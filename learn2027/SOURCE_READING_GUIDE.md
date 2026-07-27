# LSS 源码阅读地图

目标：完全按照 `src/` 里的源码来学习 LSS。每一步都要回答两个问题：

```text
这段代码把数据从什么样子变成什么样子？
为什么 LSS 需要这样变？
```

## 0. 先不要从哪里开始

新手不要一开始从这些地方开始：

```text
src/train.py
src/explore.py
src/tools.py::QuickCumsum
```

原因：

- `train.py` 是训练流程，里面会一次性牵出模型、数据、loss、optimizer。
- `explore.py` 是实验和可视化入口，适合后面验证理解。
- `QuickCumsum` 是优化版自定义反向传播，先看会很绕。

## 1. 第一站：`src/models.py`

最重要文件：

```text
src/models.py
```

阅读顺序：

```text
1. LiftSplatShoot.__init__
2. LiftSplatShoot.create_frustum
3. CamEncode.get_depth_feat
4. LiftSplatShoot.get_cam_feats
5. LiftSplatShoot.get_geometry
6. LiftSplatShoot.voxel_pooling
7. BevEncode.forward
8. LiftSplatShoot.forward
```

核心主线：

```text
图片
-> 图像特征
-> 每个像素位置的深度概率
-> frustum 3D 特征
-> ego 坐标系中的点
-> BEV 网格
-> BEV 分割结果
```

先记住默认 shape：

```text
B: batch size
N: camera 数量
D: depth 候选数量，默认 41
C: camera feature 通道数，默认 64

imgs:
B x N x 3 x 128 x 352

cam feature:
B x N x D x 8 x 22 x C

geometry:
B x N x D x 8 x 22 x 3

BEV feature:
B x C x 200 x 200

prediction:
B x 1 x 200 x 200
```

## 2. 第二站：`src/data.py`

等 `models.py` 的主线有感觉后，再看：

```text
src/data.py
```

阅读顺序：

```text
1. NuscData.__getitem__ 的两个子类
2. NuscData.get_image_data
3. NuscData.sample_augmentation
4. NuscData.get_binimg
5. compile_data
```

这里主要回答：

```text
模型需要的 imgs、rots、trans、intrins、post_rots、post_trans、binimg 是怎么来的？
```

变量含义：

```text
imgs: 多相机图片
rots: camera 到 ego 的旋转
trans: camera 到 ego 的平移
intrins: 相机内参
post_rots: resize/crop/flip/rotate 后的图像坐标旋转
post_trans: resize/crop/flip/rotate 后的图像坐标平移
binimg: BEV 上车辆占用标签
```

## 3. 第三站：`src/train.py`

训练文件只回答一个问题：

```text
模型输出 preds 后，怎么和 BEV 标签 binimgs 算 loss 并更新参数？
```

重点读：

```text
compile_data(...)
compile_model(...)
preds = model(...)
loss = loss_fn(preds, binimgs)
loss.backward()
opt.step()
get_val_info(...)
```

先不用深究 optimizer 细节，只要知道：

```text
loss 越小，模型预测的 BEV 车辆区域越接近标签。
```

## 4. 第四站：`src/tools.py`

这里是工具函数。

优先看：

```text
gen_dx_bx
ego_to_cam
cam_to_ego
img_transform
SimpleLoss
get_batch_iou
```

后看：

```text
cumsum_trick
QuickCumsum
```

`QuickCumsum` 的目的不是改变模型思想，而是让 voxel pooling 更快、更省显存。

## 5. 第五站：`src/explore.py`

这里是可视化和评估入口。

优先用：

```bash
python main.py lidar_check mini --dataroot=... --viz_train=False
```

它能帮助你看：

```text
相机图像
LiDAR 投影
BEV 标签
frustum 点投到 ego 坐标后的分布
```

## 6. 每次读源码的固定模板

看到任何一段代码，都按这个格式记笔记：

```text
函数名：

输入：
shape：

关键代码：

输出：
shape：

目的：

为什么不能省：

我还不懂：
```

## 7. 建议第一天只看这三个函数

第一天不要贪多，只看：

```text
LiftSplatShoot.__init__
LiftSplatShoot.create_frustum
CamEncode.get_depth_feat
```

这三个函数搞懂后，你就会知道 LSS 的 “Lift” 到底是怎么开始的。

