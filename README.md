# RetargetNet

**人体运动重定向神经网络** — 从人体动作捕捉数据到机器人关节控制目标的前馈神经网络映射。

---

## 概述

RetargetNet 旨在学习一个从人体姿态空间到机器人姿态空间的非线性映射函数。给定一帧人体运动捕捉数据（关节姿态、根节点朝向与平移），网络直接回归出对应的机器人控制目标（根节点位姿与各关节自由度角度），实现端到端的运动重定向（Motion Retargeting）。

当前阶段以全连接网络（MLP）为基线，验证数据管线与训练流程的可行性。后续将引入更强大的网络结构以提升映射精度。

### 核心特点

- **端到端回归** — 不依赖人工设计的运动学规则或优化迭代，单次前馈即可完成推理
- **文件级数据划分** — 按动作片段文件（而非帧）划分训练/验证/测试集，杜绝帧间信息泄露
- **可复现性** — 固定全局随机种子 (42)，确保实验可完全复现
- **模块化设计** — 模型、数据、训练/测试脚本解耦，便于替换网络结构或训练策略

---

## 项目结构

```
RetargetNet/
├── data/
│   └── vector_pairs/           # .npz 格式的帧级配对向量
│       ├── Female1General_c3d/ # 按受试者+动作类型组织的子目录
│       ├── Female1Gestures_c3d/
│       ├── Female1Running_c3d/
│       ├── Female1Walking_c3d/
│       ├── Male1General_c3d/
│       ├── Male1Running_c3d/
│       ├── Male1Walking_c3d/
│       ├── Male2General_c3d/
│       ├── Male2MartialArtsExtended_c3d/
│       ├── Male2MartialArtsKicks_c3d/
│       ├── Male2MartialArtsPunches_c3d/
│       ├── Male2MartialArtsStances_c3d/
│       ├── Male2Running_c3d/
│       ├── Male2Walking_c3d/
│       ├── MartialArtsWalksTurns_c3d/
│       ├── s001/  s007/  s008/  s009/  s011/
│       └── vector_pairs.tar    # 数据压缩包
│
├── models/
│   └── SimpleMLP.py            # 基线 MLP 模型
│
├── scripts/
│   ├── prepare_data.py         # 数据加载、划分与归一化
│   ├── train.py                # 训练入口
│   └── test.py                 # 测试入口
│
├── model_ckpt/                 # 模型权重与超参数存档
├── demo_load_pairs.py          # 数据加载示例
├── CLAUDE.md                   # AI 助手指令与项目上下文
└── README.md
```

---

## 数据格式

### 输入向量 X — 人体姿态 (69 维)

| 分量 | 维度 | 含义 |
|---|---|---|
| `input_pose_body` | 63 | 身体各关节的局部旋转 |
| `input_root_orient` | 3 | 人体根节点在世界坐标系下的朝向 |
| `input_trans` | 3 | 人体根节点在世界坐标系下的平移 |

### 输出向量 Y — 机器人目标姿态 (36 维)

| 分量 | 维度 | 含义 |
|---|---|---|
| `output_root_pos` | 3 | 机器人根节点的目标位置 |
| `output_root_rot` | 4 | 机器人根节点的目标旋转（四元数） |
| `output_dof_pos` | 29 | 机器人各关节自由度的目标角度 |

> 所有数据源于 CMU Mocap 数据库，经 C3D 格式导出后逐帧处理为上述配对向量。当前共 **252 个** `.npz` 文件，分布于 **20 个子目录**中。

### 数据划分与预处理

- **文件级划分**：70% 训练 / 15% 验证 / 15% 测试，保证同一 `.npz` 内的所有帧只出现在一个集合中
- **Z-score 归一化**：使用训练集的均值和标准差对所有划分进行标准化，`std` 下限为 `1e-8` 防止除零
- **固定随机种子** (42)：确保每次运行的文件排列结果一致

---

## 快速开始

### 环境要求

- Python ≥ 3.8
- PyTorch ≥ 1.10
- NumPy
- wandb（可选，用于训练日志记录）

### 安装

```bash
git clone <repo-url>
cd RetargetNet
conda create -n retargetnet python=3.14
conda activate retargetnet
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install numpy wandb tqdm
```

### 数据准备

将 `.npz` 数据文件按受试者/动作类型放置于 `data/vector_pairs/` 下的对应子目录中。加载示例：

```python
import numpy as np

data = np.load("data/vector_pairs/Male1General_c3d/General_A1_-_Stand_stageii_pairs.npz")
X = np.concatenate([data["input_pose_body"],
                    data["input_root_orient"],
                    data["input_trans"]], axis=1)  # (N, 69)

Y = np.concatenate([data["output_root_pos"],
                    data["output_root_rot"],
                    data["output_dof_pos"]], axis=1)  # (N, 36)
```

### 训练

```bash
python scripts/train.py --model SimpleMLP
```

训练过程中，wandb 会记录 loss 曲线与学习率变化。最优模型权重自动保存至 `model_ckpt/`。

### 测试

```bash
python scripts/test.py --model SimpleMLP
```

脚本将加载保存的超参数与最优权重，在测试集上计算并输出 MSE。

---

## 模型

### SimpleMLP（基线）

```
输入 (69) ──► Linear(69, 256) ──► ReLU ──► Dropout(0.1)
           ──► Linear(256, 256) ──► ReLU ──► Dropout(0.1)
           ──► Linear(256, 256) ──► ReLU ──► Dropout(0.1)
           ──► Linear(256, 36) ──► 输出
```

| 超参数 | 值 |
|---|---|
| `hidden_dim` | 256 |
| `num_layers` | 3 |
| `dropout_rate` | 0.1 |
| 激活函数 | ReLU |
| 总参数量 | ~119k |

### 添加新模型

1. 在 `models/` 下创建新文件（如 `Transformer.py`）
2. 实现 `.name()` 方法和 `forward()` 方法
3. 在 `train.py` 和 `test.py` 的 `if-else` 分支中注册模型及其超参数与优化器配置

---

## 训练配置

| 配置项 | 值 |
|---|---|
| 损失函数 | MSELoss |
| 优化器 | AdamW (`lr=1e-3`, `weight_decay=1e-2`) |
| 学习率调度 | CosineAnnealingLR (`T_max=75`) |
| Batch size | 128 |
| Epochs | 75 |
| 随机种子 | 42 |
| 设备 | CUDA（若可用）否则 CPU |
| 数据加载线程 | 4 (`num_workers`) |

---

## 当前局限与已知问题

| 局限 | 说明 |
|---|---|
| **模型表达能力弱** | 3 层 MLP 难以精确建模人体到机器人的复杂运动学映射，当前 MSE 较高 |
| **无时序建模** | 逐帧独立推理，未利用相邻帧的运动连续性，可能导致机器人轨迹抖动 |
| **数据规模有限** | 252 个文件仅覆盖有限的受试者和动作类型，泛化能力不足 |
| **损失函数单一** | 简单 MSE 同等对待所有输出维度，忽略了旋转分量与位置分量的语义差异 |
| **评估指标匮乏** | 仅使用 MSE，缺乏关节角度误差、末端执行器位置误差等机器人学相关指标 |

---

## 路线图

### 短期
- [ ] 收集更多高质量的动捕数据，丰富受试者与动作多样性
- [ ] 实现更深/更宽的全连接网络变体（含残差连接、LayerNorm）
- [ ] 引入学习率 warmup 策略

### 中期
- [ ] 引入时序模型架构：LSTM / GRU / Temporal Convolution / Transformer
- [ ] 设计运动学感知的损失函数（旋转几何距离、末端执行器加权等）
- [ ] 添加关节角度误差、末端位置误差等评估指标

### 长期
- [ ] 探索基于物理的约束正则化（如接触点、平衡约束）
- [ ] 实现从视频/深度传感器到网络输入的端到端 pipeline
- [ ] 将模型部署到真实机器人平台进行在线验证

---

## 引用

若本工作对您的研究有帮助，请引用：

```bibtex
@misc{retargetnet,
  title   = {RetargetNet: Feedforward Human-to-Robot Motion Retargeting},
  author  = {<Authors>},
  year    = {2026},
  note    = {Work in progress}
}
```