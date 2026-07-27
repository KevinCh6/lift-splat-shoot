# LSS 学习历史

## 2026-07-24

### 当前理解进度

已经建立了 Lift-Splat-Shoot 的模糊整体概念：

- 输入是多路相机图像。
- 输出是 BEV 鸟瞰图上的车辆分割结果。
- 单张 2D 图像本身不知道深度，所以 LSS 给每个图像特征点准备多个深度假设。
- 这些候选深度点之后会通过相机参数转换到车体坐标系，再聚合到 BEV 网格。

### 已理解的第一个关键方法：`create_frustum()`

位置：[src/models.py](src/models.py)

`create_frustum()` 的作用是创建一个固定的 frustum 坐标模板。

它使用的配置包括：

```python
self.data_aug_conf['final_dim']
self.downsample
self.grid_conf['dbound']
```

默认配置下：

```text
final_dim = (128, 352)
downsample = 16
dbound = [4.0, 45.0, 1.0]
```

因此：

```text
fH = 128 // 16 = 8
fW = 352 // 16 = 22
D = 41
```

最终输出：

```text
frustum.shape = D x H x W x 3
              = 41 x 8 x 22 x 3
```

最后一维的 `3` 表示：

```text
[图像 x 坐标, 图像 y 坐标, depth 深度]
```

也就是说，`create_frustum()` 不是在预测东西，而是在提前准备：

```text
每个图像特征点在每个假设深度上的候选坐标
```

### 关于 `ds.view(-1, 1, 1)`

`ds` 原本是一维深度列表，例如：

```python
tensor([4., 5., 6., ..., 44.])
```

`ds.view(-1, 1, 1)` 把它变成：

```text
D x 1 x 1
```

这样后面才能通过：

```python
.expand(-1, fH, fW)
```

把每个深度值铺满整张 `H x W` 的特征图网格，得到：

```text
D x H x W
```

### 关于 `torch.stack((xs, ys, ds), -1)`

在 stack 之前：

```text
xs.shape = D x H x W
ys.shape = D x H x W
ds.shape = D x H x W
```

`torch.stack((xs, ys, ds), -1)` 会把同一个位置上的 `x、y、depth` 合成一个长度为 3 的小向量。

所以：

```text
frustum[d, h, w] = [x, y, depth]
```

### `create_frustum()` 带注释版代码

下面这段不是要改源码，而是把代码和理解写在一起，方便复习。

```python
def create_frustum(self):
    # 这个函数创建一个固定的 frustum 采样模板。
    #
    # 它不是在预测深度，也不是在提取图像特征。
    # 它只是提前准备一个坐标表：
    #
    #   frustum[d, h, w] = [图像x坐标, 图像y坐标, 假设depth]
    #
    # 后面的 get_geometry() 会用这个表，把图像坐标里的候选点
    # 转换到车体 ego 坐标系中。

    # final_dim 是图像增强后的最终输入图像尺寸，格式是 (H, W)。
    #
    # 默认：
    #   final_dim = (128, 352)
    #
    # 所以：
    #   ogfH = 128
    #   ogfW = 352
    ogfH, ogfW = self.data_aug_conf['final_dim']

    # CNN 会把图像下采样 self.downsample 倍。
    #
    # 默认：
    #   self.downsample = 16
    #
    # 所以：
    #   fH = 128 // 16 = 8
    #   fW = 352 // 16 = 22
    #
    # 注意：frustum 是基于特征图位置建立的，不是基于原图每个像素建立的。
    fH, fW = ogfH // self.downsample, ogfW // self.downsample

    # dbound 控制深度采样范围。
    #
    # 默认：
    #   dbound = [4.0, 45.0, 1.0]
    #
    # torch.arange(4.0, 45.0, 1.0) 会得到：
    #   [4, 5, 6, ..., 44]
    #
    # 注意：不包含 45。
    #
    # view(-1, 1, 1):
    #   把一维深度列表变成 D x 1 x 1。
    #
    # expand(-1, fH, fW):
    #   把每个深度值铺满整张 H x W 特征图。
    #
    # 最终：
    #   ds.shape = D x H x W
    ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)

    # D 是深度层数。
    #
    # 默认 dbound=[4.0, 45.0, 1.0] 时：
    #   D = 41
    D, _, _ = ds.shape

    # xs 是图像宽度方向的采样坐标。
    #
    # 例子：
    #   ogfW = 6
    #   fW = 3
    #
    # torch.linspace(0, 5, 3) 会得到：
    #   [0, 2.5, 5]
    #
    # view(1, 1, fW):
    #   变成 1 x 1 x W。
    #
    # expand(D, fH, fW):
    #   扩展成 D x H x W。
    #
    # 含义：
    #   每个深度层、每一行，都有同样的 x 坐标网格。
    xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)

    # ys 是图像高度方向的采样坐标。
    #
    # 例子：
    #   ogfH = 4
    #   fH = 2
    #
    # torch.linspace(0, 3, 2) 会得到：
    #   [0, 3]
    #
    # view(1, fH, 1):
    #   变成 1 x H x 1。
    #
    # expand(D, fH, fW):
    #   扩展成 D x H x W。
    #
    # 含义：
    #   每个深度层、每一列，都有同样的 y 坐标网格。
    ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

    # stack 前：
    #   xs.shape = D x H x W
    #   ys.shape = D x H x W
    #   ds.shape = D x H x W
    #
    # torch.stack((xs, ys, ds), -1) 会把同一个位置上的：
    #   x
    #   y
    #   depth
    #
    # 合成一个长度为 3 的小向量。
    #
    # stack 后：
    #   frustum.shape = D x H x W x 3
    #
    # 最后一维 3 表示：
    #   [图像x坐标, 图像y坐标, depth]
    frustum = torch.stack((xs, ys, ds), -1)

    # frustum 是固定几何模板，不是网络要学习的参数。
    #
    # nn.Parameter:
    #   把 frustum 注册到模型中，让它能跟着模型一起 .to(device)、保存、加载。
    #
    # requires_grad=False:
    #   训练时不更新它。
    return nn.Parameter(frustum, requires_grad=False)
```

### `create_frustum()` 小例子

假设配置是：

```python
self.data_aug_conf['final_dim'] = (4, 6)
self.downsample = 2
self.grid_conf['dbound'] = [10.0, 40.0, 5.0]
```

则：

```text
ogfH = 4
ogfW = 6
fH = 4 // 2 = 2
fW = 6 // 2 = 3
D = 6
```

深度：

```text
ds = [10, 15, 20, 25, 30, 35]
```

x 坐标：

```text
xs = [0, 2.5, 5]
```

y 坐标：

```text
ys = [0, 3]
```

最终：

```text
frustum.shape = 6 x 2 x 3 x 3
```

其中一层：

```text
depth = 10:
[
  [[0,   0, 10], [2.5, 0, 10], [5, 0, 10]],
  [[0,   3, 10], [2.5, 3, 10], [5, 3, 10]]
]
```

### 下一步

接下来要看：

```python
get_geometry()
```

它会把 `create_frustum()` 生成的：

```text
[图像 x 坐标, 图像 y 坐标, depth]
```

转换成车体坐标系中的：

```text
[前后位置, 左右位置, 高度]
```

一句话记忆：

```text
create_frustum: 图像坐标里的候选点
get_geometry: 车体坐标里的候选点
```

## 项目整体类、方法和调用关系

### 项目文件分工

```text
main.py          命令入口
src/train.py     训练流程
src/data.py      nuScenes 数据读取和标签生成
src/models.py    LSS 模型核心
src/tools.py     几何、loss、IoU、可视化工具函数
src/explore.py   评估和可视化脚本
```

### 命令入口

`main.py` 使用 `Fire` 把命令行命令映射到不同函数：

```python
Fire({
    'lidar_check': src.explore.lidar_check,
    'cumsum_check': src.explore.cumsum_check,
    'train': src.train.train,
    'eval_model_iou': src.explore.eval_model_iou,
    'viz_model_preds': src.explore.viz_model_preds,
})
```

对应关系：

```text
python main.py train             -> 训练模型
python main.py eval_model_iou    -> 评估 IoU
python main.py viz_model_preds   -> 可视化预测
python main.py lidar_check       -> 检查相机/雷达几何是否正确
python main.py cumsum_check      -> 检查 voxel pooling 加速实现
```

### 训练调用链路

训练入口在 `src/train.py`：

```python
def train(...):
    # 1. 先整理 BEV 网格配置。
    #
    # xbound / ybound / zbound 控制 BEV 空间范围和网格大小。
    # dbound 控制图像 frustum 的深度采样范围。
    #
    # 这些配置后面会给：
    #   compile_data()
    #   compile_model()
    #
    # 数据集要用它生成 BEV 标签；
    # 模型要用它生成 frustum 和 BEV voxel 网格。
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }

    # 2. 整理图像增强和相机配置。
    #
    # final_dim:
    #   图像 resize/crop 后输入网络的尺寸。
    #
    # cams:
    #   nuScenes 的 6 个环视相机。
    #
    # Ncams:
    #   每次训练实际使用几个相机。
    #
    # 这些配置后面会给 data.py 读取和增强图像，
    # 也会给 models.py 创建 frustum。
    data_aug_conf = {
        'resize_lim': resize_lim,
        'final_dim': final_dim,
        'rot_lim': rot_lim,
        'H': H,
        'W': W,
        'rand_flip': rand_flip,
        'bot_pct_lim': bot_pct_lim,
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'Ncams': ncams,
    }

    # 3. 编译数据。
    #
    # compile_data() 在 src/data.py 中。
    #
    # 内部大致做：
    #   NuScenes(...)
    #   -> SegmentationData(...)
    #       -> NuscData(...)
    #   -> DataLoader(...)
    #
    # trainloader / valloader 每次返回：
    #   imgs
    #   rots
    #   trans
    #   intrins
    #   post_rots
    #   post_trans
    #   binimgs
    #
    # 其中：
    #   imgs 是多相机图像；
    #   rots/trans/intrins 是相机几何参数；
    #   post_rots/post_trans 是图像增强后的几何修正；
    #   binimgs 是 BEV 车辆分割标签。
    trainloader, valloader = compile_data(
        version,
        dataroot,
        data_aug_conf=data_aug_conf,
        grid_conf=grid_conf,
        bsz=bsz,
        nworkers=nworkers,
        parser_name='segmentationdata',
    )

    # 4. 选择设备。
    #
    # gpuid < 0 时使用 CPU；
    # 否则使用 cuda:{gpuid}。
    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    # 5. 构建模型。
    #
    # compile_model() 在 src/models.py 中。
    #
    # 内部实际返回：
    #   LiftSplatShoot(grid_conf, data_aug_conf, outC)
    #
    # LiftSplatShoot.__init__() 内部会继续创建：
    #   create_frustum()
    #   CamEncode()
    #   BevEncode()
    #
    # outC=1 表示输出 1 个通道：
    #   每个 BEV grid cell 是不是 vehicle。
    model = compile_model(grid_conf, data_aug_conf, outC=1)
    model.to(device)

    # 6. 创建优化器。
    #
    # Adam 负责更新模型参数。
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 7. 创建 loss。
    #
    # SimpleLoss 是 src/tools.py 里的 BCEWithLogitsLoss 封装。
    #
    # preds 是模型输出的 logits；
    # binimgs 是 BEV 二值标签。
    loss_fn = SimpleLoss(pos_weight).cuda(gpuid)

    # 8. 开始训练循环。
    model.train()
    counter = 0
    for epoch in range(nepochs):
        for batchi, batch in enumerate(trainloader):
            # DataLoader 给出一个 batch。
            imgs, rots, trans, intrins, post_rots, post_trans, binimgs = batch

            # 清空上一轮梯度。
            opt.zero_grad()

            # 9. 前向传播。
            #
            # 这里会调用：
            #   LiftSplatShoot.forward()
            #     -> get_voxels()
            #         -> get_geometry()
            #         -> get_cam_feats()
            #         -> voxel_pooling()
            #     -> bevencode()
            #
            # 输入：
            #   多相机图像 + 相机参数
            #
            # 输出：
            #   BEV 车辆分割 logits
            preds = model(
                imgs.to(device),
                rots.to(device),
                trans.to(device),
                intrins.to(device),
                post_rots.to(device),
                post_trans.to(device),
            )

            # 10. 计算 loss。
            #
            # preds:
            #   模型预测的 BEV logits。
            #
            # binimgs:
            #   BEV 车辆标签。
            binimgs = binimgs.to(device)
            loss = loss_fn(preds, binimgs)

            # 11. 反向传播。
            #
            # loss.backward() 会把梯度从 BEV loss 一路传回：
            #   BevEncode
            #   voxel_pooling
            #   CamEncode
            #   depthnet
            #
            # 所以深度分布虽然没有直接深度监督，
            # 也会通过最终 BEV 分割 loss 被训练。
            loss.backward()

            # 12. 梯度裁剪，避免梯度过大。
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # 13. 更新模型参数。
            opt.step()
            counter += 1

            # 14. 训练日志。
            #
            # 每 10 step 记录一次 train/loss。
            # 每 50 step 计算一次 train/iou。
            # 每 val_step 做一次验证和保存模型。
            if counter % 10 == 0:
                writer.add_scalar('train/loss', loss, counter)

            if counter % 50 == 0:
                _, _, iou = get_batch_iou(preds, binimgs)
                writer.add_scalar('train/iou', iou, counter)

            if counter % val_step == 0:
                # get_val_info() 会在验证集上跑 model，
                # 计算 val loss 和 val IoU。
                val_info = get_val_info(model, valloader, loss_fn, device)

                # 保存当前模型权重。
                torch.save(model.state_dict(), mname)
```

### 数据相关类和方法

核心文件是 `src/data.py`。这一块的目的：

```text
nuScenes 原始数据
  -> 多相机图像
  -> 相机内参/外参
  -> 图像增强后的几何修正
  -> BEV 车辆分割标签
```

带注释版调用骨架：

```python
def compile_data(version, dataroot, data_aug_conf, grid_conf, bsz,
                 nworkers, parser_name):
    # 1. 创建 nuScenes API 对象。
    #
    # version:
    #   mini 或 trainval。
    #
    # dataroot:
    #   nuScenes 数据根目录。
    #
    # 这里会读 nuScenes metadata，例如 sample、sample_data、
    # calibrated_sensor、ego_pose、annotation 等。
    nusc = NuScenes(
        version='v1.0-{}'.format(version),
        dataroot=os.path.join(dataroot, version),
        verbose=False,
    )

    # 2. 根据 parser_name 选择数据集类型。
    #
    # segmentationdata:
    #   训练/评估使用，返回图像、相机参数、BEV 标签。
    #
    # vizdata:
    #   可视化使用，额外返回 LiDAR 点云。
    parser = {
        'vizdata': VizData,
        'segmentationdata': SegmentationData,
    }[parser_name]

    # 3. 创建训练集和验证集。
    #
    # parser(...) 会进入：
    #   SegmentationData.__init__()
    #     -> NuscData.__init__()
    #         -> get_scenes()
    #         -> prepro()
    #         -> fix_nuscenes_formatting()
    traindata = parser(nusc, is_train=True, data_aug_conf=data_aug_conf,
                       grid_conf=grid_conf)
    valdata = parser(nusc, is_train=False, data_aug_conf=data_aug_conf,
                     grid_conf=grid_conf)

    # 4. 用 PyTorch DataLoader 包起来。
    #
    # 训练时 shuffle=True；
    # 验证时 shuffle=False。
    trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
                                              shuffle=True,
                                              num_workers=nworkers,
                                              drop_last=True,
                                              worker_init_fn=worker_rnd_init)
    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers)

    return trainloader, valloader
```

`NuscData` 是基础数据集类：

```python
class NuscData(torch.utils.data.Dataset):
    def __init__(self, nusc, is_train, data_aug_conf, grid_conf):
        # 保存 nuScenes API、训练/验证标记、图像增强配置、BEV 网格配置。
        self.nusc = nusc
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf

        # 选择当前 split 的 scene 名称。
        # 例如：
        #   mini_train / mini_val
        #   train / val
        self.scenes = self.get_scenes()

        # 从 nusc.sample 里筛选属于当前 split 的 sample。
        self.ixes = self.prepro()

        # 根据 xbound/ybound/zbound 生成 BEV 网格参数。
        #
        # 后面的 get_binimg() 会用这些参数，
        # 把车辆 box 画到 BEV 标签图上。
        dx, bx, nx = gen_dx_bx(
            grid_conf['xbound'],
            grid_conf['ybound'],
            grid_conf['zbound'],
        )
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        # 兼容不同 nuScenes 文件目录结构。
        self.fix_nuscenes_formatting()

    def get_scenes(self):
        # 根据 nusc.version 和 is_train 选择 split。
        #
        # v1.0-mini + is_train=True  -> mini_train
        # v1.0-mini + is_train=False -> mini_val
        # v1.0-trainval + True       -> train
        # v1.0-trainval + False      -> val
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[self.nusc.version][self.is_train]

        scenes = create_splits_scenes()[split]
        return scenes

    def prepro(self):
        # 取出全部 sample。
        samples = [samp for samp in self.nusc.sample]

        # 只保留当前 split 里的 sample。
        samples = [
            samp for samp in samples
            if self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes
        ]

        # 按 scene 和时间排序，主要方便可视化。
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))
        return samples

    def sample_augmentation(self):
        # 训练时随机 resize/crop/flip/rotate。
        # 验证时使用固定 resize/crop，不随机 flip/rotate。
        #
        # 这个方法只生成增强参数；
        # 真正应用图像变换在 get_image_data() 里。
        return resize, resize_dims, crop, flip, rotate

    def get_image_data(self, rec, cams):
        # 对一个 sample，读取多个相机图像和相机参数。
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []

        for cam in cams:
            # 1. 找到当前相机对应的 sample_data。
            samp = self.nusc.get('sample_data', rec['data'][cam])

            # 2. 读取图像。
            img = Image.open(imgname)

            # 3. 读取相机内参。
            # intrins: 相机坐标 -> 像素坐标用的内参矩阵。
            intrin = torch.Tensor(sens['camera_intrinsic'])

            # 4. 读取相机外参。
            # rot/trans: 相机坐标 -> ego 车体坐标。
            rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)
            tran = torch.Tensor(sens['translation'])

            # 5. 图像增强。
            #
            # img_transform() 不只改图像，还会返回 post_rot/post_tran。
            # post_rot/post_tran 记录 resize/crop/flip/rotate 对像素坐标造成的影响。
            img, post_rot2, post_tran2 = img_transform(...)

            # 6. 保存模型需要的输入。
            imgs.append(normalize_img(img))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        # 返回一个 sample 的所有相机数据。
        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans))

    def get_binimg(self, rec):
        # 生成 BEV 车辆分割标签。
        #
        # 做法：
        #   遍历当前 sample 的所有 annotation；
        #   只保留 category_name 以 vehicle 开头的目标；
        #   把 3D box 转到当前 ego 坐标系；
        #   取 box 底面四个角；
        #   映射到 BEV grid 坐标；
        #   用 cv2.fillPoly 在标签图上填成 1。
        #
        # 输出：
        #   binimg.shape = 1 x X x Y
        return torch.Tensor(img).unsqueeze(0)

    def choose_cams(self):
        # 训练时如果 Ncams 小于总相机数，会随机选 Ncams 个相机。
        # 验证时使用固定相机列表。
        return cams
```

训练/评估真正用的是 `SegmentationData`：

```python
class SegmentationData(NuscData):
    def __getitem__(self, index):
        # 1. 取出一个 nuScenes sample。
        rec = self.ixes[index]

        # 2. 选择使用哪些相机。
        cams = self.choose_cams()

        # 3. 读取多相机图像和相机参数。
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)

        # 4. 生成 BEV 车辆标签。
        binimg = self.get_binimg(rec)

        # 5. DataLoader 每次拿到的就是这些。
        return imgs, rots, trans, intrins, post_rots, post_trans, binimg
```

### 模型相关类和方法

核心文件是 `src/models.py`。这一块的目的：

```text
多相机图像 + 相机参数
  -> Lift: 图像特征扩展到多个深度层
  -> Splat: 根据几何位置聚合到 BEV 网格
  -> BEV CNN: 输出车辆分割 logits
```

主模型带注释版骨架：

```python
class LiftSplatShoot(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC):
        # 保存配置。
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf

        # 1. 生成 BEV voxel 网格参数。
        #
        # dx: 每个格子的实际大小。
        # bx: 第一个格子的中心位置。
        # nx: 每个方向有多少格。
        dx, bx, nx = gen_dx_bx(
            self.grid_conf['xbound'],
            self.grid_conf['ybound'],
            self.grid_conf['zbound'],
        )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        # 2. 图像特征图相对输入图像下采样 16 倍。
        self.downsample = 16

        # 3. 每个 frustum 点携带 64 维图像特征。
        self.camC = 64

        # 4. 生成固定 frustum 模板。
        #
        # create_frustum() 只在初始化时调用一次。
        #
        # self.frustum.shape:
        #   D x H x W x 3
        #
        # 最后一维是：
        #   [图像x, 图像y, depth]
        self.frustum = self.create_frustum()

        # D 是深度层数。
        self.D, _, _, _ = self.frustum.shape

        # 5. 图像编码器。
        #
        # CamEncode 负责：
        #   输入单张相机图像；
        #   提取图像特征；
        #   预测深度分布；
        #   输出 D x H x W x C 的 frustum feature。
        self.camencode = CamEncode(self.D, self.camC, self.downsample)

        # 6. BEV 编码器。
        #
        # BevEncode 负责：
        #   输入 BEV feature；
        #   输出 BEV 车辆分割 logits。
        self.bevencode = BevEncode(inC=self.camC, outC=outC)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        # 输入：
        #   self.frustum:
        #     D x H x W x 3，图像坐标里的 [x, y, depth]。
        #
        #   rots/trans:
        #     相机坐标 -> ego 坐标的外参。
        #
        #   intrins:
        #     相机内参。
        #
        #   post_rots/post_trans:
        #     图像增强 resize/crop/flip/rotate 后的几何修正。
        #
        # 输出：
        #   points.shape = B x N x D x H x W x 3
        #
        # 含义：
        #   每个 batch、每个相机、每个深度层、每个特征点，
        #   在车体 ego 坐标系里的 3D 位置。

        # 1. 撤销图像增强对像素坐标的影响。
        #
        # self.frustum 本来是增强后图像坐标里的点。
        # 要投回真实相机几何，需要先 undo post transformation。
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # 2. 从像素坐标 [u, v, depth] 变成相机坐标。
        #
        # 像素坐标里：
        #   u = x / z
        #   v = y / z
        #
        # 所以这里先做：
        #   [u, v, d] -> [u*d, v*d, d]
        points = torch.cat((
            points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
            points[:, :, :, :, :, 2:3],
        ), 5)

        # 3. 使用相机内参逆矩阵和外参，把相机坐标转到 ego 坐标。
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)

        # 4. 加上相机在 ego 坐标系中的平移。
        points += trans.view(B, N, 1, 1, 1, 3)

        return points

    def get_cam_feats(self, x):
        # 输入：
        #   x.shape = B x N x 3 x imH x imW
        #
        # B: batch size
        # N: 相机数量

        B, N, C, imH, imW = x.shape

        # 1. 把 batch 和 camera 两个维度合并。
        #
        # CamEncode 只处理普通图片 batch：
        #   B*N x 3 x H x W
        x = x.view(B*N, C, imH, imW)

        # 2. 对每张图片做 Lift。
        #
        # self.camencode(x) 输出：
        #   B*N x C x D x H/16 x W/16
        x = self.camencode(x)

        # 3. 把 batch 和 camera 维度还原回来。
        x = x.view(B, N, self.camC, self.D,
                   imH//self.downsample, imW//self.downsample)

        # 4. 调整维度顺序，和 get_geometry() 的 geom 对齐。
        #
        # 输出：
        #   B x N x D x H x W x C
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x

    def voxel_pooling(self, geom_feats, x):
        # 输入：
        #   geom_feats.shape = B x N x D x H x W x 3
        #   x.shape          = B x N x D x H x W x C
        #
        # geom_feats 告诉每个 frustum feature 应该落在 ego 空间哪里；
        # x 是这些点携带的图像特征。

        # 1. 展平成一堆点。
        x = x.reshape(Nprime, C)
        geom_feats = geom_feats.view(Nprime, 3)

        # 2. 把 ego 坐标转换成 BEV voxel 下标。
        #
        # 真实坐标：
        #   米为单位的 x/y/z。
        #
        # voxel 下标：
        #   第几个 grid cell。
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()

        # 3. 过滤掉 BEV 范围外的点。
        #
        # 这里用 ... 表示省略完整布尔条件；
        # 源码里会分别判断 x/y/z 三个方向是否在 nx 范围内。
        kept = ...

        # 4. 把落到同一个 voxel 的点排在一起。
        #
        # 这里用 ... 表示省略完整 rank 公式；
        # 源码会把 voxel 的 x/y/z 下标和 batch 下标编码成一个整数 rank。
        ranks = ...
        sorts = ranks.argsort()

        # 5. 对同一个 voxel 内的 feature 求和。
        #
        # QuickCumsum 是更省内存/更快的自定义 autograd 实现。
        x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # 6. 把稀疏点重新放回规则 BEV feature map。
        #
        # final.shape:
        #   B x C x Z x X x Y
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)

        # 7. 把 Z 维压到通道维。
        #
        # 默认 zbound=[-10,10,20] 时，Z 只有 1 层。
        # 所以可以近似理解为输出：
        #   B x C x X x Y
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        # 这是 Lift-Splat 的总控。
        #
        # 1. 算几何位置。
        #    geom.shape = B x N x D x H x W x 3
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)

        # 2. 算图像 frustum feature。
        #    x.shape = B x N x D x H x W x C
        x = self.get_cam_feats(x)

        # 3. 根据 geom 把 x 聚合进 BEV 网格。
        #    x.shape = B x C x X x Y
        x = self.voxel_pooling(geom, x)

        return x

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        # 完整前向：
        #
        # 多相机图像
        #   -> Lift + Splat 成 BEV feature
        #   -> BEV CNN 输出车辆分割 logits
        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        x = self.bevencode(x)
        return x
```

### 模型内部子类

```python
class CamEncode(nn.Module)
```

`CamEncode` 负责单张相机图像的 Lift：

```python
class CamEncode(nn.Module):
    def __init__(self, D, C, downsample):
        # D: 深度层数，例如 41。
        # C: 每个 frustum 点的图像特征通道数，例如 64。
        self.D = D
        self.C = C

        # 1. 图像 backbone。
        #
        # EfficientNet-B0 从单张图中提取 2D 图像特征。
        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")

        # 2. 上采样融合模块。
        #
        # 把 EfficientNet 深层特征和浅层特征融合，
        # 得到空间尺寸更合适的图像 feature map。
        self.up1 = Up(320+112, 512)

        # 3. depthnet 同时输出两部分：
        #
        # 前 D 个通道：
        #   每个像素/特征点的深度 logits。
        #
        # 后 C 个通道：
        #   每个像素/特征点的图像语义特征。
        #
        # 所以输出通道数是 D + C。
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        # 对深度通道做 softmax。
        #
        # 输入：
        #   x.shape = B*N x D x H x W
        #
        # 输出：
        #   depth.shape = B*N x D x H x W
        #
        # 含义：
        #   每个图像特征点在 D 个深度上的概率分布。
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        # 1. 用 EfficientNet 提取图像特征。
        x = self.get_eff_depth(x)

        # 2. 用 1x1 conv 同时预测：
        #   depth logits
        #   image feature
        x = self.depthnet(x)

        # 3. 前 D 个通道变成深度概率。
        depth = self.get_depth_dist(x[:, :self.D])

        # 4. 后 C 个通道是图像特征。
        #
        # x[:, self.D:self.D + self.C].shape:
        #   B*N x C x H x W
        #
        # unsqueeze(2) 后：
        #   B*N x C x 1 x H x W
        image_feat = x[:, self.D:(self.D + self.C)]

        # 5. 把图像特征按深度概率分配到 D 个深度层。
        #
        # depth.unsqueeze(1).shape:
        #   B*N x 1 x D x H x W
        #
        # image_feat.unsqueeze(2).shape:
        #   B*N x C x 1 x H x W
        #
        # 相乘后广播得到：
        #   B*N x C x D x H x W
        #
        # 这就是 Lift：
        #   2D 图像特征 -> 带深度分布的 frustum feature。
        new_x = depth.unsqueeze(1) * image_feat.unsqueeze(2)

        return depth, new_x

    def forward(self, x):
        # CamEncode.forward() 只返回 frustum feature，
        # depth 概率中间参与计算，但不作为最终模型输出返回。
        depth, x = self.get_depth_feat(x)
        return x
```

`BevEncode` 负责 BEV 上的卷积预测：

```python
class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        # inC:
        #   voxel_pooling 后的 BEV feature 通道数。
        #
        # outC:
        #   输出通道数。车辆分割任务里 outC=1。

        # 使用 ResNet18 的部分结构做 BEV feature 编码。
        trunk = resnet18(pretrained=False, zero_init_residual=True)

        # 第一层把 BEV feature 转到 64 通道。
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)

        # 复用 ResNet 的 layer1/layer2/layer3。
        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        # 上采样并融合浅层特征。
        self.up1 = Up(64+256, 256, scale_factor=4)

        # 最后上采样回 BEV 输出分辨率，并输出 outC 通道 logits。
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        # 输入：
        #   x.shape = B x C x X x Y
        #
        # 输出：
        #   x.shape = B x outC x X x Y
        #
        # 对本项目：
        #   outC = 1
        #   表示每个 BEV 位置是不是 vehicle 的 logits。
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # x1 是浅层 BEV 特征，后面和深层特征做 skip connection。
        x1 = self.layer1(x)

        # 继续提取更深层的 BEV 语义特征。
        x = self.layer2(x1)
        x = self.layer3(x)

        # 把深层特征上采样，并和 x1 拼接融合。
        x = self.up1(x, x1)

        # 输出最终 BEV 分割 logits。
        x = self.up2(x)
        return x
```

`Up` 是辅助上采样模块：

```python
class Up(nn.Module):
    def forward(self, x1, x2):
        # x1: 深层特征，语义强但分辨率低。
        # x2: 浅层特征，语义弱但分辨率高。

        # 1. 把深层特征上采样。
        x1 = self.up(x1)

        # 2. 和浅层特征在通道维拼接。
        x1 = torch.cat([x2, x1], dim=1)

        # 3. 用卷积融合。
        return self.conv(x1)
```

### 工具函数

核心文件：

```text
src/tools.py
```

重要函数：

```python
gen_dx_bx()
```

生成 BEV 网格参数：

```text
dx: 每个格子的大小
bx: 第一个格子的中心位置
nx: 每个方向有多少格
```

```python
img_transform()
```

执行图像增强，同时记录增强对应的几何变换。

```python
ego_to_cam()
cam_to_ego()
```

相机坐标和车体坐标之间转换。

```python
cumsum_trick()
QuickCumsum
```

用于快速把同一个 voxel 里的特征求和。

```python
SimpleLoss
```

`BCEWithLogitsLoss` 的封装。

```python
get_batch_iou()
get_val_info()
```

计算 IoU 和验证集指标。

### 完整训练调用链

```text
main.py
  -> train()

train()
  -> compile_data()
      -> SegmentationData.__getitem__()
          -> get_image_data()
          -> get_binimg()

  -> compile_model()
      -> LiftSplatShoot.__init__()
          -> create_frustum()
          -> CamEncode()
          -> BevEncode()

  -> model(imgs, rots, trans, intrins, post_rots, post_trans)
      -> LiftSplatShoot.forward()
          -> get_voxels()
              -> get_geometry()
                  -> 使用 self.frustum
              -> get_cam_feats()
                  -> CamEncode.forward()
                      -> get_depth_feat()
                          -> get_eff_depth()
                          -> get_depth_dist()
              -> voxel_pooling()
          -> BevEncode.forward()

  -> SimpleLoss(preds, binimgs)
  -> backward()
  -> optimizer.step()
```

一句话总结：

```text
data.py 准备多相机图像、相机参数和 BEV 标签；
models.py 把图像 Lift 到深度 frustum，再 Splat 到 BEV 网格，最后预测车辆分割；
train.py 负责把这套流程训练起来；
explore.py 负责评估和可视化。
```
