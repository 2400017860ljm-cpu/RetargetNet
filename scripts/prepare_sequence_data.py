# prepare_sequence_data.py
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

SEQ_LEN = 16   # 窗口长度
STRIDE  = 8    # 步长
NORM_STATS_PATH = "model_ckpt/norm_stats.json"

def collect_npz_files(root_dir):
    """递归收集所有 .npz 文件路径"""
    npz_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith('.npz'):
                npz_files.append(os.path.join(dirpath, f))
    return sorted(npz_files)

def file_based_split(npz_files, test_size=0.15, val_size=0.15, seed=42):
    """文件级划分训练/验证/测试集（纯 NumPy 实现）"""
    n = len(npz_files)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    test_num = int(n * test_size)
    val_num = int(n * val_size)
    train_num = n - test_num - val_num

    test_idx = indices[:test_num]
    val_idx = indices[test_num:test_num + val_num]
    train_idx = indices[test_num + val_num:]

    train_files = [npz_files[i] for i in train_idx]
    val_files   = [npz_files[i] for i in val_idx]
    test_files  = [npz_files[i] for i in test_idx]

    return train_files, val_files, test_files

def sliding_windows(data_X, data_Y, seq_len, stride):
    """从一个完整动作文件的帧序列中生成重叠的窗口"""
    windows_X, windows_Y = [], []
    n_frames = data_X.shape[0]
    for start in range(0, n_frames - seq_len + 1, stride):
        end = start + seq_len
        windows_X.append(data_X[start:end])
        windows_Y.append(data_Y[start:end])
    if not windows_X:
        return np.array([]), np.array([])
    return np.stack(windows_X), np.stack(windows_Y)

def compute_norm_stats(file_list):
    """在训练集文件上计算归一化统计数据，并保存 JSON"""
    X_all, Y_all = [], []
    for path in file_list:
        data = np.load(path)
        X = np.concatenate([data["input_pose_body"], data["input_root_orient"], data["input_trans"]], axis=1)
        Y = np.concatenate([data["output_root_pos"], data["output_root_rot"], data["output_dof_pos"]], axis=1)
        X_all.append(X)
        Y_all.append(Y)
    X_cat = np.concatenate(X_all, axis=0)
    Y_cat = np.concatenate(Y_all, axis=0)

    x_mean = X_cat.mean(axis=0, keepdims=True)
    x_std  = X_cat.std(axis=0, keepdims=True).clip(min=1e-8)
    y_mean = Y_cat.mean(axis=0, keepdims=True)
    y_std  = Y_cat.std(axis=0, keepdims=True).clip(min=1e-8)

    stats = {
        "x_mean": x_mean.tolist(),
        "x_std":  x_std.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std":  y_std.tolist(),
    }
    os.makedirs(os.path.dirname(NORM_STATS_PATH), exist_ok=True)
    with open(NORM_STATS_PATH, 'w') as f:
        json.dump(stats, f)

    return x_mean, x_std, y_mean, y_std

class SequenceDataset(Dataset):
    def __init__(self, file_list, x_mean, x_std, y_mean, y_std, seq_len, stride):
        self.X_list, self.Y_list = [], []
        for path in file_list:
            data = np.load(path)
            X = np.concatenate([data["input_pose_body"], data["input_root_orient"], data["input_trans"]], axis=1)
            Y = np.concatenate([data["output_root_pos"], data["output_root_rot"], data["output_dof_pos"]], axis=1)
            X = (X - x_mean) / x_std
            Y = (Y - y_mean) / y_std
            wX, wY = sliding_windows(X, Y, seq_len, stride)
            if wX.size > 0:
                self.X_list.append(wX)
                self.Y_list.append(wY)
        if self.X_list:
            self.X = np.concatenate(self.X_list, axis=0)
            self.Y = np.concatenate(self.Y_list, axis=0)
        else:
            self.X = np.empty((0, seq_len, 69), dtype=np.float32)
            self.Y = np.empty((0, seq_len, 36), dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.float32)

def create_dataloaders(root_dir="data/vector_pairs", batch_size=128, num_workers=4):
    """主函数：生成训练、验证、测试的 DataLoader"""
    npz_files = collect_npz_files(root_dir)
    train_files, val_files, test_files = file_based_split(npz_files)
    print(f"文件划分：训练 {len(train_files)}，验证 {len(val_files)}，测试 {len(test_files)}")

    x_mean, x_std, y_mean, y_std = compute_norm_stats(train_files)

    train_dataset = SequenceDataset(train_files, x_mean, x_std, y_mean, y_std, SEQ_LEN, STRIDE)
    val_dataset   = SequenceDataset(val_files,   x_mean, x_std, y_mean, y_std, SEQ_LEN, STRIDE)
    test_dataset  = SequenceDataset(test_files,  x_mean, x_std, y_mean, y_std, SEQ_LEN, STRIDE)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, (x_mean, x_std, y_mean, y_std)

if __name__ == "__main__":
    train_loader, val_loader, test_loader, _ = create_dataloaders()
    print(f"训练批次: {len(train_loader)}，验证批次: {len(val_loader)}，测试批次: {len(test_loader)}")