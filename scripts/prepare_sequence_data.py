import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
from tqdm import tqdm
import joblib

class SequencePoseDataset(Dataset):
    def __init__(self, human_poses, robot_poses, sequence_length=30):
        self.sequence_length = sequence_length
        self.human_poses = human_poses
        self.robot_poses = robot_poses
        
        # 计算有效序列数量
        self.num_sequences = len(human_poses) - sequence_length + 1
        
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        # 获取序列
        human_seq = self.human_poses[idx:idx + self.sequence_length]
        robot_seq = self.robot_poses[idx:idx + self.sequence_length]
        
        return torch.FloatTensor(human_seq), torch.FloatTensor(robot_seq[-1])  # 预测最后一个时间步

def _load_all_npz_files(data_dir):
    pattern = os.path.join(data_dir, "**", "*.npz")
    return sorted(glob.glob(pattern, recursive=True))

def _load_single_npz(filepath):
    data = np.load(filepath)
    X = np.concatenate([
        data["input_pose_body"],
        data["input_root_orient"],
        data["input_trans"],
    ], axis=1).astype(np.float32)
    Y = np.concatenate([
        data["output_root_pos"],
        data["output_root_rot"],
        data["output_dof_pos"],
    ], axis=1).astype(np.float32)
    return X, Y

def create_sequence_dataloaders(root_dir="data/vector_pairs",
                               sequence_length=30,
                               batch_size=64,
                               num_workers=0,
                               save_scalers=False,
                               scaler_dir=None):
    """
    为LSTM模型创建序列数据加载器
    
    Args:
        root_dir: 数据目录
        sequence_length: 序列长度
        batch_size: 批大小
        num_workers: 数据加载工作线程数
        save_scalers: 是否保存标准化参数
        scaler_dir: 标准化参数保存目录
    
    Returns:
        train_loader, valid_loader, test_loader, stats
    """
    all_files = _load_all_npz_files(root_dir)
    assert len(all_files) > 0, f"No .npz files found under {root_dir}"
    
    print(f" Found {len(all_files)} .npz files")
    
    # 文件级分割
    np.random.seed(42)
    indices = np.random.permutation(len(all_files))
    n_train = int(len(all_files) * 0.7)
    n_valid = int(len(all_files) * 0.15)
    
    train_files = [all_files[i] for i in indices[:n_train]]
    valid_files = [all_files[i] for i in indices[n_train:n_train+n_valid]]
    test_files = [all_files[i] for i in indices[n_train+n_valid:]]
    
    print(f" Split: {len(train_files)} train, {len(valid_files)} valid, {len(test_files)} test files")
    
    # 加载数据
    def load_files(file_list):
        all_X, all_Y = [], []
        for fp in tqdm(file_list, desc="Loading files"):
            X, Y = _load_single_npz(fp)
            all_X.append(X)
            all_Y.append(Y)
        return np.concatenate(all_X, axis=0), np.concatenate(all_Y, axis=0)
    
    X_train, Y_train = load_files(train_files)
    X_valid, Y_valid = load_files(valid_files)
    X_test, Y_test = load_files(test_files)
    
    print(f" Data shapes - Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")
    
    # 标准化（仅使用训练集统计量）
    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)
    x_std = np.clip(x_std, 1e-8, None)
    
    y_mean = Y_train.mean(axis=0)
    y_std = Y_train.std(axis=0)
    y_std = np.clip(y_std, 1e-8, None)
    
    # 应用标准化
    X_train = (X_train - x_mean) / x_std
    X_valid = (X_valid - x_mean) / x_std
    X_test = (X_test - x_mean) / x_std
    
    Y_train = (Y_train - y_mean) / y_std
    Y_valid = (Y_valid - y_mean) / y_std
    Y_test = (Y_test - y_mean) / y_std
    
    # 保存标准化参数
    stats = {
        'input_mean': x_mean,
        'input_std': x_std,
        'output_mean': y_mean,
        'output_std': y_std,
        'sequence_length': sequence_length
    }
    
    if save_scalers and scaler_dir:
        os.makedirs(scaler_dir, exist_ok=True)
        joblib.dump(stats, os.path.join(scaler_dir, 'lstm_sequence_scalers.pkl'))
        print(f" LSTM sequence scalers saved to {scaler_dir}")
    
    # 创建数据集
    train_dataset = SequencePoseDataset(X_train, Y_train, sequence_length)
    valid_dataset = SequencePoseDataset(X_valid, Y_valid, sequence_length)
    test_dataset = SequencePoseDataset(X_test, Y_test, sequence_length)
    
    print(f" Sequence dataset sizes - Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, valid_loader, test_loader, stats