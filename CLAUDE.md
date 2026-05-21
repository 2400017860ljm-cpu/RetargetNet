# RetargetNet — 人体姿态到机器人姿态的神经网络映射

## 项目目标

训练一个前馈神经网络，将输入的人体姿态向量映射为机器人姿态向量，实现从人体运动捕捉数据到机器人关节控制目标的端到端回归。

## 数据

### 来源与格式

所有数据以 `.npz` 格式存储于 `data/vector_pairs/`，共 **252 个文件**，涵盖 **20 个子目录**（按 CMU Mocap 受试者与动作类型组织）。子目录命名规则为 `{Subject}{Action}_c3d`，来源为 C3D 格式的动捕数据经处理后的帧级配对向量。

| 受试者/来源 | 动作类型 |
|---|---|
| Female1 | General, Gestures, Running, Walking |
| Male1 | General, Running, Walking |
| Male2 | General, Running, Walking, MartialArtsKicks, MartialArtsPunches, MartialArtsStances, MartialArtsExtended |
| MartialArtsWalksTurns | 混合武术转身 |
| s001, s007, s008, s009, s011 | 额外受试者数据 |

### 输入向量 X — 人体姿态 (69 维)

```
X = concat([input_pose_body (63),    # 身体关节姿态
            input_root_orient (3),    # 根节点朝向
            input_trans (3)])         # 根节点平移
```

### 输出向量 Y — 机器人目标姿态 (36 维)

```
Y = concat([output_root_pos (3),     # 根节点位置
            output_root_rot (4),     # 根节点旋转 (四元数)
            output_dof_pos (29)])    # 机器人关节自由度角度
```

### 数据划分

- 按**文件级别**划分（避免同一动作片段的帧泄露到不同集合）
- 比例：70% 训练 / 15% 验证 / 15% 测试
- 固定随机种子 42 进行文件排列
- **归一化**：使用训练集统计量（均值和标准差）对所有划分进行 Z-score 标准化，`x_std` 和 `y_std` 以 `1e-8` 为下限防止除零

## 模型

### SimpleMLP (`models/SimpleMLP.py`)

当前唯一实现的基线模型。

```
输入 (69) → Linear(69, 256) → ReLU → Dropout(0.1)
          → Linear(256, 256) → ReLU → Dropout(0.1)
          → Linear(256, 256) → ReLU → Dropout(0.1)
          → Linear(256, 36) → 输出
```

| 超参数 | 值 |
|---|---|
| hidden_dim | 256 |
| num_layers | 3 |
| dropout_rate | 0.1 |
| 总参数量 | ~取决于 hidden_dim |

### 模型扩展点

`train.py` 和 `test.py` 通过 `--model` 参数选择模型，当前仅支持 `SimpleMLP`。若要添加新模型，需在 `models/` 下新建文件，并在 `train.py`/`test.py` 的 if-else 分支注册。

## 训练

### 命令

```
python scripts/train.py --model SimpleMLP
python scripts/test.py --model SimpleMLP
```

### 训练配置

| 配置项 | 值 |
|---|---|
| 损失函数 | MSELoss |
| 优化器 | AdamW (lr=1e-3, weight_decay=1e-2) |
| 学习率调度 | CosineAnnealingLR (T_max=75) |
| Batch size | 128 |
| Epochs | 75 |
| 随机种子 | 42 (全面复现性设置) |
| 设备 | CUDA (若可用) 否则 CPU |

### 日志与检查点

- 使用 **wandb** 记录训练曲线（项目名 `pose-mapping`，run 名同模型名）
- 每 epoch 记录：train_loss、valid_loss、学习率
- 每 step 记录：train_loss (step 级)
- 最优模型权重保存至 `model_ckpt/pose_{model_name}.pth`
- 超参数保存至 `model_ckpt/pose_{model_name}_arg.pickle`

### 测试

测试脚本加载保存的超参数 pickle → 重建模型 → 加载最优权重 → 在测试集上计算 MSE。

## 当前状态与局限

- **SimpleMLP 能力较差**：仅作为初步基线实验，MSE 较高，无法精确建模人体到机器人的复杂运动学映射。
- **数据量有限**：252 个文件覆盖的受试者和动作类型尚不充分。
- **逐帧独立预测**：当前 MLP 对每帧独立推理，未利用时序上下文（如相邻帧的运动连续性）。

## 下一步方向

1. **数据扩充**：收集更多高质量的动作捕捉数据，增加受试者多样性和动作覆盖范围
2. **网络结构优化**：
   - 引入时序模型（LSTM / GRU / Temporal Convolution / Transformer）
   - 尝试更深或更宽的全连接网络
   - 引入残差连接、LayerNorm 等现代架构组件
3. **训练策略改进**：
   - 学习率 warmup
   - 更复杂的优化器设置
   - 损失函数改进（如加权 MSE、运动学约束损失）
4. **评估指标丰富化**：除 MSE 外，加入关节角度误差、末端执行器位置误差等机器人学相关指标