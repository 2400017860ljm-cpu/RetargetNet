import numpy as np
import matplotlib.pyplot as plt
import torch, pickle, glob, joblib, sys
sys.path.append(".")
from models.ResMLP import ResMLP

# 加载模型
with open("model_ckpt/pose_ResMLP_arg.pickle", "rb") as f: hps = pickle.load(f)
model = ResMLP(**hps)
model.load_state_dict(torch.load("model_ckpt/pose_ResMLP.pth", map_location='cpu'))
model.eval()

# 加载标准化参数
x_scaler = joblib.load("model_ckpt/x_scaler.pkl")
y_scaler = joblib.load("model_ckpt/y_scaler.pkl")
x_mean = torch.tensor(x_scaler["mean"], dtype=torch.float32)
x_std  = torch.tensor(x_scaler["std"], dtype=torch.float32)
y_mean = y_scaler["mean"]
y_std  = y_scaler["std"]

# 加载一个动作文件
npz_file = sorted(glob.glob("data/vector_pairs/**/*.npz", recursive=True))[0]
data = np.load(npz_file)
X = np.concatenate([data["input_pose_body"], data["input_root_orient"], data["input_trans"]], axis=1).astype(np.float32)
Y = np.concatenate([data["output_root_pos"], data["output_root_rot"], data["output_dof_pos"]], axis=1).astype(np.float32)

# 取前200帧
X = X[:200]; Y = Y[:200]
X_norm = (X - x_mean.numpy()) / x_std.numpy()
with torch.no_grad():
    pred_norm = model(torch.from_numpy(X_norm)).cpu().numpy()
pred = pred_norm * y_std + y_mean   # 反标准化

# 提取 dof (索引7:36)
true_dof = Y[:, 7:36]
pred_dof = pred[:, 7:36]

# 画前5个关节的曲线
fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
for i in range(5):
    axes[i].plot(true_dof[:, i], label='True', alpha=0.7)
    axes[i].plot(pred_dof[:, i], label='Pred', alpha=0.7)
    axes[i].set_ylabel(f'Joint {i}')
    axes[i].legend()
axes[-1].set_xlabel('Frame')
plt.suptitle('Predicted vs True Joint Angles (first 5 DOFs)')
plt.tight_layout()
plt.show()