import sys
sys.path.append(".")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from scripts.prepare_data import prepare_pose_datasets
from models.ResMLP import ResMLP
from models.RegLSTM import RegLSTM
from models.TransformerRegressor import TransformerRegressor  # 新增 Transformer 模型
import pickle
import os

# ----------------- 维度定义 -----------------
ROOT_POS_DIMS = slice(0, 3)      # 3
ROOT_ROT_DIMS = slice(3, 7)      # 4
DOF_POS_DIMS  = slice(7, 36)     # 29


def angular_error_quat(q1, q2):
    """q1, q2: (N,4) 单位四元数，返回角度误差（度）"""
    dot = torch.abs(torch.sum(q1 * q2, dim=1))
    dot = torch.clamp(dot, 0.0, 1.0)
    angle_rad = 2.0 * torch.acos(dot)
    return torch.rad2deg(angle_rad)


def test_model_detailed(model, test_loader, criterion, device, norm_stats=None):
    model.eval()
    total_loss = 0.0
    pos_loss = 0.0
    rot_loss = 0.0
    dof_loss = 0.0
    total_samples = 0

    all_angle_errors = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            # 统一预测：如果是 RegLSTM，需将单帧扩展为长度为1的序列
            if isinstance(model, RegLSTM):
                pred = model(x.unsqueeze(1)).squeeze(1)
            else:
                pred = model(x)

            batch_size = x.size(0)
            total_samples += batch_size

            # 损失累积
            total_loss += criterion(pred, y).item() * batch_size
            pos_loss += criterion(pred[:, ROOT_POS_DIMS], y[:, ROOT_POS_DIMS]).item() * batch_size
            rot_loss += criterion(pred[:, ROOT_ROT_DIMS], y[:, ROOT_ROT_DIMS]).item() * batch_size
            dof_loss += criterion(pred[:, DOF_POS_DIMS], y[:, DOF_POS_DIMS]).item() * batch_size

            # 角度误差计算（需要归一化参数）
            if norm_stats is not None:
                mean = torch.tensor(norm_stats["mean"], dtype=torch.float32, device=device)
                std  = torch.tensor(norm_stats["std"], dtype=torch.float32, device=device)
                pred_rot_raw = pred[:, ROOT_ROT_DIMS] * std[ROOT_ROT_DIMS] + mean[ROOT_ROT_DIMS]
                true_rot_raw = y[:, ROOT_ROT_DIMS] * std[ROOT_ROT_DIMS] + mean[ROOT_ROT_DIMS]

                pred_rot_norm = pred_rot_raw / pred_rot_raw.norm(dim=1, keepdim=True)
                true_rot_norm = true_rot_raw / true_rot_raw.norm(dim=1, keepdim=True)

                angles = angular_error_quat(pred_rot_norm, true_rot_norm)
                all_angle_errors.append(angles.cpu())

    N = total_samples
    if N == 0:
        print("错误：测试样本数为0，请检查数据加载。")
        return float('nan')

    avg_total = total_loss / N
    avg_pos = pos_loss / N
    avg_rot = rot_loss / N
    avg_dof = dof_loss / N

    print(f"\n========== 详细测试结果 ==========")
    print(f"样本数: {N}")
    print(f"整体 MSE (标准化):          {avg_total:.6f}")
    print(f"  ├─ root_pos MSE:          {avg_pos:.6f}")
    print(f"  ├─ root_rot MSE (标准):   {avg_rot:.6f}")
    print(f"  └─ dof_pos  MSE:          {avg_dof:.6f}")

    if norm_stats is not None and len(all_angle_errors) > 0:
        all_angles = torch.cat(all_angle_errors, dim=0)
        print(f"\n根旋转角度误差 (°):")
        print(f"  均值: {all_angles.mean().item():.2f}°")
        print(f"  中位数: {all_angles.median().item():.2f}°")
        print(f"  90分位数: {torch.quantile(all_angles, 0.9).item():.2f}°")
        print(f"  最大值: {all_angles.max().item():.2f}°")

    return avg_total


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', default="ResMLP", type=str)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, _, test_dataset = prepare_pose_datasets(data_dir="data/vector_pairs", save_scalers=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    ckpt_path = f"./model_ckpt/pose_{args.model}_arg.pickle"
    with open(ckpt_path, 'rb') as f:
        model_hps = pickle.load(f)

    # ---- 模型重建（注意使用 elif 防止逻辑错误） ----
    if args.model == "ResMLP":
        model = ResMLP(**model_hps)
    elif args.model == "RegLSTM":
        model = RegLSTM(**model_hps)
    elif args.model == "TransformerRegressor":
        model = TransformerRegressor(**model_hps)
    
    else:
        raise ValueError(f"Model {args.model} is not implemented!")

    model_path = f"./model_ckpt/pose_{model.name()}.pth"
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    criterion = nn.MSELoss()

    norm_stats = None
    norm_path = "model_ckpt/norm_stats.pkl"
    if os.path.exists(norm_path):
        with open(norm_path, "rb") as f:
            norm_stats = pickle.load(f)
        print("检测到 norm_stats.pkl，将计算角度误差。")
    else:
        print("未找到 norm_stats.pkl，仅输出标准化MSE。")

    test_loss = test_model_detailed(model, test_loader, criterion, device, norm_stats)
    print(f"\nTest Loss (MSE) on pose test dataset: {test_loss:.6f}")