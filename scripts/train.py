import sys
sys.path.append(".")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from argparse import ArgumentParser
import pickle
import os
import random
import numpy as np

from scripts.prepare_data import prepare_pose_datasets   # MLP 单帧数据
from models.TemporalModel import SimpleLSTM              # 新增 LSTM 模型
from prepare_sequence_data import create_dataloaders     # 新增序列数据加载
from models.ResMLP import ResMLP                      # 新增 ResMLP 模型
from models.RegLSTM import RegLSTM
from models.TransformerRegressor import TransformerRegressor  # 新增 Transformer 模型
# ======================== 新增：组合损失函数 ========================
class CombinedLoss(nn.Module):
    """
    位置+MSE / 旋转+L2距离 / 自由度+MSE 的组合损失。
    支持对目标旋转施加随机小角度扰动（数据增强），提升旋转泛化能力。
    """
    def __init__(self, mean, std, pos_weight=1.0, rot_weight=1.0, dof_weight=1.0, rot_augment_deg=0.0):
        super().__init__()
        self.mean = mean
        self.std = std
        self.pos_weight = pos_weight
        self.rot_weight = rot_weight
        self.dof_weight = dof_weight
        self.rot_augment_deg = rot_augment_deg
        self.mse = nn.MSELoss(reduction='mean')

        self.idx_pos = slice(0, 3)
        self.idx_rot = slice(3, 7)
        self.idx_dof = slice(7, 36)

    def forward(self, pred, target):
        if pred.dim() == 3:
            B, T, D = pred.shape
            pred = pred.reshape(-1, D)
            target = target.reshape(-1, D)

        # 1. 位置 MSE（标准化空间）
        loss_pos = self.mse(pred[:, self.idx_pos], target[:, self.idx_pos])

        # 2. 旋转损失（反标准化后计算 L2 距离）
        pred_rot = pred[:, self.idx_rot] * self.std[self.idx_rot] + self.mean[self.idx_rot]
        target_rot = target[:, self.idx_rot] * self.std[self.idx_rot] + self.mean[self.idx_rot]

        # 数据增强：仅训练时对目标四元数施加随机微小旋转
        if self.training and self.rot_augment_deg > 0:
            B = target_rot.size(0)
            device = target_rot.device
            max_rad = torch.deg2rad(torch.tensor(self.rot_augment_deg, device=device))
            axis = torch.randn(B, 3, device=device)
            axis = axis / axis.norm(dim=1, keepdim=True).clamp(min=1e-7)
            angle_rad = (torch.rand(B, device=device) * 2 - 1) * max_rad
            half = angle_rad / 2.0
            q_w = torch.cos(half)
            q_xyz = torch.sin(half).unsqueeze(1) * axis
            rand_quat = torch.cat([q_w.unsqueeze(1), q_xyz], dim=1)

            # 归一化目标四元数并旋转
            target_rot = target_rot / target_rot.norm(dim=1, keepdim=True).clamp(min=1e-7)
            w1, x1, y1, z1 = rand_quat[:,0], rand_quat[:,1], rand_quat[:,2], rand_quat[:,3]
            w2, x2, y2, z2 = target_rot[:,0], target_rot[:,1], target_rot[:,2], target_rot[:,3]
            new_w = w1*w2 - x1*x2 - y1*y2 - z1*z2
            new_x = w1*x2 + x1*w2 + y1*z2 - z1*y2
            new_y = w1*y2 - x1*z2 + y1*w2 + z1*x2
            new_z = w1*z2 + x1*y2 - y1*x2 + z1*w2
            target_rot = torch.stack([new_w, new_x, new_y, new_z], dim=1)
        else:
            target_rot = target_rot / target_rot.norm(dim=1, keepdim=True).clamp(min=1e-7)

        # 预测四元数归一化
        pred_rot_norm = pred_rot / pred_rot.norm(dim=1, keepdim=True).clamp(min=1e-7)

        # 单位四元数 L2 距离
        loss_rot = (pred_rot_norm - target_rot).pow(2).sum(dim=1).mean()

        # 3. 自由度 MSE（标准化空间）
        loss_dof = self.mse(pred[:, self.idx_dof], target[:, self.idx_dof])

        return self.pos_weight * loss_pos + self.rot_weight * loss_rot + self.dof_weight * loss_dof
# ====================================================================


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def train_model(model, train_loader, valid_loader, criterion, optimizer,
                scheduler, device, num_epochs=10):
    model.to(device)
    model_save_path = f"./model_ckpt/pose_{model.name()}.pth"
    best_loss = float("inf")

    for epoch in range(num_epochs):
        # ---- train ----
        model.train()
        train_loss = 0.0
        train_samples = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [train]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_samples = x.size(0)
            train_loss += loss.item() * batch_samples
            train_samples += batch_samples

            wandb.log({"train_loss(step)": loss.item()})

        avg_train_loss = train_loss / train_samples

        # ---- validation ----
        model.eval()
        valid_loss = 0.0
        valid_samples = 0
        with torch.no_grad():
            for x, y in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} [valid]"):
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)

                valid_loss += loss.item() * x.size(0)
                valid_samples += x.size(0)

        avg_valid_loss = valid_loss / valid_samples

        if scheduler is not None:
            scheduler.step()

        wandb.log({
            "train_loss(epoch)": avg_train_loss,
            "valid_loss(epoch)": avg_valid_loss,
            "lr": optimizer.param_groups[0]['lr'],
        })

        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"--> Best Model Saved with Val Loss: {best_loss:.6f}")

        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_valid_loss:.6f}")

    print(f"Training Complete! Best Val Loss: {best_loss:.6f}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', default="SimpleMLP", type=str,
                        choices=["SimpleMLP", "SimpleLSTM","ResMLP","RegLSTM","TransformerRegressor"],)
    args = parser.parse_args()

    # ---- reproducibility ----
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- wandb ----
    wandb.init(project="pose-mapping", name=f"{args.model}")

    # ---- 根据模型选择数据和模型配置 ----
    if args.model == "SimpleMLP":
        model_hps = {
            "hidden_dim": 256,
            "num_layers": 3,
            "dropout_rate": 0.1,
        }
        batch_size = 128
        model = SimpleMLP(**model_hps)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=75)
        # 使用原来的逐帧数据集
        train_dataset, valid_dataset, _ = prepare_pose_datasets(data_dir="data/vector_pairs")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        num_epochs = 75

    elif args.model == "SimpleLSTM":
        model_hps = {
            "input_dim": 69,
            "hidden_dim": 256,
            "num_layers": 2,
            "output_dim": 36,
            "dropout": 0.1,
            "bidirectional": False,
        }
        batch_size = 64   # 序列模型降低 batch 以避免 OOM
        num_epochs = 75
        model = SimpleLSTM(**model_hps)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        # 使用序列数据集（滑动窗口）
        train_loader, valid_loader, _, _ = create_dataloaders(
            root_dir="data/vector_pairs",
            batch_size=batch_size,
            num_workers=0,
        )
    elif args.model == "ResMLP":
        model_hps = {
            "hidden_dim": 512,
            "num_blocks": 4,
            "dropout": 0.2,
        }
        batch_size = 128
        model = ResMLP(**model_hps)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=75)
        train_dataset, valid_dataset, _ = prepare_pose_datasets(data_dir="data/vector_pairs")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        num_epochs = 75
    elif args.model == "RegLSTM":
        model_hps = {
            "input_dim": 69,
            "hidden_dim": 128,
            "num_layers": 1,
            "output_dim": 36,
            "dropout": 0.4,
        }
        batch_size = 64
        num_epochs = 75
        model = RegLSTM(**model_hps)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        # 使用序列数据集（滑动窗口）
        train_loader, valid_loader, _, _ = create_dataloaders(
            root_dir="data/vector_pairs",
            batch_size=batch_size,
            num_workers=0,
        )
    elif args.model == "TransformerRegressor":
        model_hps = {
            "input_dim": 69,
            "output_dim": 36,
            "d_model": 256,
            "nhead": 8,
            "num_layers": 4,
            "dropout": 0.2,
        }
        batch_size = 128
        num_epochs = 75
        model = TransformerRegressor(**model_hps)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        train_dataset, valid_dataset, _ = prepare_pose_datasets(data_dir="data/vector_pairs")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    else:
        raise ValueError(f"Model {args.model} is not implemented!")

    total_params = count_parameters(model)
    print(f'Total number of parameters: {total_params:,}')

    # ---- 保存超参数 ----
    ckpt_path = f"./model_ckpt/pose_{model.name()}_arg.pickle"
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    with open(ckpt_path, 'wb') as f:
        pickle.dump(model_hps, f)

    # -------------------- 选择损失函数 --------------------
    norm_stats_path = "model_ckpt/norm_stats.pkl"
    if os.path.exists(norm_stats_path):
        with open(norm_stats_path, "rb") as f:
            norm_stats = pickle.load(f)
        mean = torch.tensor(norm_stats["mean"], dtype=torch.float32, device=device)
        std  = torch.tensor(norm_stats["std"],  dtype=torch.float32, device=device)
        criterion = CombinedLoss(mean, std,
                                 pos_weight=1.0, rot_weight=5.0, dof_weight=1.0, rot_augment_deg=5.0)
        print("使用 CombinedLoss（位置MSE + 旋转角度损失 + 自由度MSE）。")
    else:
        criterion = nn.MSELoss()
        print("未找到 norm_stats.pkl，回退为普通 MSELoss。请先保存归一化参数以获得更好效果。")
    # -----------------------------------------------------

    # ---- 开始训练 ----
    import time
    start_time = time.time()
    train_model(model, train_loader, valid_loader, criterion, optimizer,
                scheduler, device, num_epochs=num_epochs)

    print(f"Training took {(time.time() - start_time) / 60:.1f} min")
    wandb.finish()