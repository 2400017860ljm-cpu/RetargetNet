import sys; sys.path.append(".")
import torch
import numpy as np
import pickle, glob, joblib

# ================== 加载模型 ==================
device = torch.device("cpu")
with open("model_ckpt/pose_ResMLP_arg.pickle", "rb") as f:
    hps = pickle.load(f)
from models.ResMLP import ResMLP
model = ResMLP(**hps)
model.load_state_dict(torch.load("model_ckpt/pose_ResMLP.pth", map_location=device))
model.eval()

# ================== 标准化参数 ==================
x_scaler = joblib.load("model_ckpt/x_scaler.pkl")
y_scaler = joblib.load("model_ckpt/y_scaler.pkl")
x_mean = torch.tensor(x_scaler["mean"], dtype=torch.float32)
x_std  = torch.tensor(x_scaler["std"], dtype=torch.float32)
y_mean = y_scaler["mean"]
y_std  = y_scaler["std"]

# ================== 选择一个动作文件 ==================
all_npz = sorted(glob.glob("data/vector_pairs/**/*.npz", recursive=True))
npz_file = all_npz[0]   # 可以手动改成你想展示的文件路径
print(f"使用文件: {npz_file}")

data = np.load(npz_file)
X = np.concatenate([data["input_pose_body"],
                    data["input_root_orient"],
                    data["input_trans"]], axis=1).astype(np.float32)

frames_to_export = min(300, len(X))   # 最多导出300帧
X = X[:frames_to_export]

# ================== 模型推理 ==================
X_norm = (X - x_mean.numpy()) / x_std.numpy()
with torch.no_grad():
    pred_norm = model(torch.from_numpy(X_norm)).cpu().numpy()
pred_raw = pred_norm * y_std + y_mean   # (N, 36)

# ================== 提取并转换数据 ==================
root_pos = pred_raw[:, 0:3]            # (N, 3)  世界位置
root_rot_raw = pred_raw[:, 3:7]        # 四元数 (可能是 wxyz)
dof_pos = pred_raw[:, 7:36]            # 29个关节角度

# 检查四元数顺序：假设原始数据是 wxyz，需转为 xyzw
# 如果你的数据本身就是 xyzw，请注释掉下面这行
root_rot = root_rot_raw[:, [1,2,3,0]]  # wxyz -> xyzw

# ================== 打印数据范围，辅助诊断 ==================
print(f"root_pos 范围: x={root_pos[:,0].min():.2f}~{root_pos[:,0].max():.2f}, "
      f"y={root_pos[:,1].min():.2f}~{root_pos[:,1].max():.2f}, "
      f"z={root_pos[:,2].min():.2f}~{root_pos[:,2].max():.2f}")
print(f"root_rot 前3帧 (xyzw): {root_rot[:3]}")
print(f"dof_pos 前3帧 (rad): {dof_pos[:3, :5]}...")   # 只打印前5个关节

# ================== 保存为 npz ==================
np.savez_compressed("exported_animation.npz",
                    root_pos=root_pos,
                    root_rot=root_rot,   # 现在为 xyzw
                    dof_pos=dof_pos)
print("已导出 exported_animation.npz，准备在 Blender 中使用。")