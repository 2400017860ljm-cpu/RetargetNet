import sys; sys.path.append(".")
import torch, numpy as np, pickle, glob, joblib, os

# ========== 配置 ==========
DATA_DIR = "data/vector_pairs/Female1General_c3d"   # 目标目录
OUT_DIR = "exported"                                 # 输出中间文件的目录
MAX_FRAMES = 200                                     # 每个动作最多取多少帧（避免太长）

os.makedirs(OUT_DIR, exist_ok=True)

# 加载模型
device = torch.device("cpu")
with open("model_ckpt/pose_ResMLP_arg.pickle", "rb") as f:
    hps = pickle.load(f)
from models.ResMLP import ResMLP
model = ResMLP(**hps)
model.load_state_dict(torch.load("model_ckpt/pose_ResMLP.pth", map_location=device))
model.eval()

# 标准化参数
x_scaler = joblib.load("model_ckpt/x_scaler.pkl")
y_scaler = joblib.load("model_ckpt/y_scaler.pkl")
x_mean = torch.tensor(x_scaler["mean"], dtype=torch.float32)
x_std  = torch.tensor(x_scaler["std"], dtype=torch.float32)
y_mean = y_scaler["mean"]
y_std  = y_scaler["std"]

# 获取所有 .npz 文件，按文件名排序（保证顺序一致）
all_npz = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
print(f"找到 {len(all_npz)} 个动作文件")

for i, npz_file in enumerate(all_npz):
    print(f"[{i+1}/{len(all_npz)}] 处理: {os.path.basename(npz_file)}")
    data = np.load(npz_file)
    X = np.concatenate([data["input_pose_body"],
                        data["input_root_orient"],
                        data["input_trans"]], axis=1).astype(np.float32)
    frames = min(MAX_FRAMES, len(X))
    X = X[:frames]

    # 标准化 + 推理 + 反标准化
    X_norm = (X - x_mean.numpy()) / x_std.numpy()
    with torch.no_grad():
        pred_norm = model(torch.from_numpy(X_norm)).cpu().numpy()
    pred_raw = pred_norm * y_std + y_mean

    root_pos = pred_raw[:, 0:3]
    root_rot = pred_raw[:, 3:7]          # 假设原始为 wxyz，需转 xyzw
    root_rot = root_rot[:, [1,2,3,0]]    # wxyz -> xyzw  (如果你的数据本身就是 xyzw 则注释掉)
    dof_pos = pred_raw[:, 7:36]

    out_path = os.path.join(OUT_DIR, f"anim_{i:03d}.npz")
    np.savez_compressed(out_path,
                        root_pos=root_pos,
                        root_rot=root_rot,
                        dof_pos=dof_pos)
print("全部导出完成！")