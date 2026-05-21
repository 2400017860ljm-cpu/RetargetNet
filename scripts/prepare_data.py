import numpy as np
import torch
from torch.utils.data import TensorDataset
import os
import glob
import joblib  # 用于保存标准化参数
import pickle

def _load_all_npz_files(data_dir):
    """Scan data_dir recursively and return a list of all .npz file paths."""
    pattern = os.path.join(data_dir, "**", "*.npz")
    return sorted(glob.glob(pattern, recursive=True))

def _load_single_npz(filepath):
    """
    Load a single .npz file and return (X, Y) as float32 numpy arrays.
    X: (N, 69)  — human pose input
    Y: (N, 36)  — robot pose output
    """
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

def prepare_pose_datasets(data_dir="data/vector_pairs",
                          train_ratio=0.70, valid_ratio=0.15,
                          save_scalers=True,  # 新增：是否保存标准化参数
                          scaler_dir="model_ckpt"):  # 新增：保存目录
    """
    Load all .npz pose pairs, split at the file level, normalize using
    training-set statistics, and return three TensorDatasets.

    Args:
        data_dir: path to the vector_pairs directory
        train_ratio: fraction of files used for training
        valid_ratio: fraction of files used for validation
        save_scalers: whether to save normalization parameters
        scaler_dir: directory to save scalers

    Returns:
        train_dataset, valid_dataset, test_dataset (TensorDataset)
    """
    all_files = _load_all_npz_files(data_dir)
    assert len(all_files) > 0, f"No .npz files found under {data_dir}"

    # --- file-level split to avoid leakage across frames of the same clip ---
    n_total = len(all_files)
    n_train = int(n_total * train_ratio)
    n_valid = int(n_total * valid_ratio)
    n_test = n_total - n_train - n_valid

    np.random.seed(42)  # 固定随机种子
    indices = np.random.permutation(n_total)
    train_files = [all_files[i] for i in indices[:n_train]]
    valid_files = [all_files[i] for i in indices[n_train:n_train + n_valid]]
    test_files  = [all_files[i] for i in indices[n_train + n_valid:]]

    # --- load frames from each split ---
    X_train, Y_train = _load_split(train_files)
    X_valid, Y_valid = _load_split(valid_files)
    X_test,  Y_test  = _load_split(test_files)

    # --- 转换为numpy数组进行标准化 ---
    X_train = X_train.numpy()
    Y_train = Y_train.numpy()
    X_valid = X_valid.numpy()
    Y_valid = Y_valid.numpy()
    X_test = X_test.numpy()
    Y_test = Y_test.numpy()

    # --- normalize: compute stats on training set only ---
    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)
    x_std = np.clip(x_std, a_min=1e-8, a_max=None)  # 防止除零

    y_mean = Y_train.mean(axis=0)
    y_std = Y_train.std(axis=0)
    y_std = np.clip(y_std, a_min=1e-8, a_max=None)

    # 应用标准化
    X_train = (X_train - x_mean) / x_std
    X_valid = (X_valid - x_mean) / x_std
    X_test  = (X_test  - x_mean) / x_std

    Y_train = (Y_train - y_mean) / y_std
    Y_valid = (Y_valid - y_mean) / y_std
    Y_test  = (Y_test  - y_mean) / y_std

    # --- 保存标准化参数 ---
    if save_scalers:
        os.makedirs(scaler_dir, exist_ok=True)
        x_scaler = {
            'mean': x_mean,
            'std': x_std
        }
        y_scaler = {
            'mean': y_mean,
            'std': y_std
        }
        joblib.dump(x_scaler, os.path.join(scaler_dir, 'x_scaler.pkl'))
        joblib.dump(y_scaler, os.path.join(scaler_dir, 'y_scaler.pkl'))
        norm_stats = {
            "mean": y_mean,
            "std": y_std
        }
        with open(os.path.join(scaler_dir, "norm_stats.pkl"), "wb") as f:
            pickle.dump(norm_stats, f)
        print(f"Saved norm_stats.pkl in {scaler_dir}")       
        print(f"Saved standard paras in {scaler_dir}")

    # --- 转换回PyTorch张量 ---
    X_train = torch.from_numpy(X_train).float()
    Y_train = torch.from_numpy(Y_train).float()
    X_valid = torch.from_numpy(X_valid).float()
    Y_valid = torch.from_numpy(Y_valid).float()
    X_test = torch.from_numpy(X_test).float()
    Y_test = torch.from_numpy(Y_test).float()

    return (TensorDataset(X_train, Y_train),
            TensorDataset(X_valid, Y_valid),
            TensorDataset(X_test,  Y_test))

def _load_split(file_list):
    """Load all frames from a list of .npz files and stack into tensors."""
    xs, ys = [], []
    for fp in file_list:
        x, y = _load_single_npz(fp)
        xs.append(torch.from_numpy(x))
        ys.append(torch.from_numpy(y))
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)