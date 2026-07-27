# Jupyter 学习入口

这里的 `src/` 是根目录 `src/` 的学习副本：

```text
learn2027/src/
```

你可以在这个副本里随便加 `print()`、改单行、插入断点，不会影响原始源码。

## 推荐打开

```text
learn2027/lss_step_through.ipynb
```

## 启动方式

在仓库根目录运行：

```bash
jupyter lab
```

或：

```bash
jupyter notebook
```

然后打开：

```text
learn2027/lss_step_through.ipynb
```

## 注意

Notebook 第一格会执行：

```python
sys.path.insert(0, str(repo_root / "learn2027"))
```

这样：

```python
import src
```

导入的是 `learn2027/src`，不是根目录原始 `src`。

## 私有路径配置

Notebook 不应该写死个人电脑上的绝对路径。当前第一格会按顺序寻找仓库：

```text
1. 从当前工作目录向上寻找仓库根目录
2. 读取环境变量 LSS_REPO_ROOT
3. 读取环境变量 LSS_LOCAL_CONFIG 指向的配置文件
4. 读取 ~/.config/lss2027/local_config.py
5. 读取 learn2027/local_config.py
```

推荐方式一：从仓库根目录启动 Jupyter。

```bash
cd /path/to/lift-splat-shoot
conda activate lss2027
jupyter lab
```

推荐方式二：设置环境变量，不写进 Git。

```bash
export LSS_REPO_ROOT=/path/to/lift-splat-shoot
conda activate lss2027
jupyter lab
```

推荐方式三：使用本地配置文件。

```bash
mkdir -p ~/.config/lss2027
cp learn2027/local_config.example.py ~/.config/lss2027/local_config.py
```

然后把 `~/.config/lss2027/local_config.py` 里的 `REPO_ROOT` 改成你的本地路径。

也可以复制到：

```text
learn2027/local_config.py
```

这个文件已经被 `.gitignore` 忽略，不会提交到 GitHub。
