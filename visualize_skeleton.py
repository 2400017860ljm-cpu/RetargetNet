import sys; sys.path.append(".")
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle
import glob
import joblib

# ================== 正向运动学 ==================
def build_skeleton(dof_angles):
    parents = [-1, 0, 1, 2, 2, 4, 5, 6, 7, 8, 9,
               2, 11, 12, 13, 14, 15, 16,
               0, 18, 19, 20, 21, 22,
               0, 24, 25, 26, 27, 28]

    offsets = np.array([
        [0, 0, 0], [0, 0, 0.2], [0, 0, 0.2], [0, 0, 0.25],
        [0.12, 0, 0.25], [0, 0, 0.15], [0, 0, 0], [0, 0, -0.25],
        [0, 0, -0.125], [0, 0, 0], [0, 0, 0],
        [-0.12, 0, 0.25], [0, 0, 0.15], [0, 0, 0], [0, 0, -0.25],
        [0, 0, -0.125], [0, 0, 0], [0, 0, 0],
        [0.07, -0.1, -0.1], [0, 0, 0], [0, 0, -0.2], [0, 0, -0.35],
        [0, 0, -0.175], [0, 0, 0],
        [-0.07, -0.1, -0.1], [0, 0, 0], [0, 0, -0.2], [0, 0, -0.35],
        [0, 0, -0.175], [0, 0, 0]
    ])

    axes = np.array([
        [0,0,1], [0,1,0], [1,0,0], [0,0,1],
        [1,0,0], [0,1,0], [0,0,1], [1,0,0],
        [0,1,0], [1,0,0], [0,0,1],
        [1,0,0], [0,1,0], [0,0,1], [1,0,0],
        [0,1,0], [1,0,0], [0,0,1],
        [1,0,0], [0,1,0], [0,0,1], [1,0,0],
        [1,0,0], [0,1,0],
        [1,0,0], [0,1,0], [0,0,1], [1,0,0],
        [1,0,0], [0,1,0]
    ])

    n_joints = len(parents)
    world_pos = np.zeros((n_joints, 3))
    world_rot = [np.eye(3) for _ in range(n_joints)]

    for i in range(1, n_joints):
        parent = parents[i]
        axis = axes[i]
        angle = dof_angles[i-1]
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        world_rot[i] = world_rot[parent] @ R
        world_pos[i] = world_pos[parent] + world_rot[parent] @ offsets[i]
    return world_pos

# ================== 四元数转旋转矩阵 ==================
def quat_to_matrix(q):
    """四元数 (w, x, y, z) -> 3x3 旋转矩阵"""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
        [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])

# ================== 加载模型与参数 ==================
device = torch.device("cpu")
with open("model_ckpt/pose_ResMLP_arg.pickle", "rb") as f:
    hps = pickle.load(f)
from models.ResMLP import ResMLP
model = ResMLP(**hps)
model.load_state_dict(torch.load("model_ckpt/pose_ResMLP.pth", map_location=device))
model.eval()

x_scaler = joblib.load("model_ckpt/x_scaler.pkl")
y_scaler = joblib.load("model_ckpt/y_scaler.pkl")
x_mean = torch.tensor(x_scaler["mean"], dtype=torch.float32)
x_std  = torch.tensor(x_scaler["std"], dtype=torch.float32)
y_mean = y_scaler["mean"]
y_std  = y_scaler["std"]

# ================== 加载连续动作序列 ==================
all_npz = sorted(glob.glob("data/vector_pairs/**/*.npz", recursive=True))
npz_file = all_npz[0]
print(f"使用文件: {npz_file}")

data = np.load(npz_file)
X = np.concatenate([data["input_pose_body"],
                    data["input_root_orient"],
                    data["input_trans"]], axis=1).astype(np.float32)
Y = np.concatenate([data["output_root_pos"],
                    data["output_root_rot"],
                    data["output_dof_pos"]], axis=1).astype(np.float32)

frames_to_show = min(300, len(X))
X = X[:frames_to_show]

X_norm = (X - x_mean.numpy()) / x_std.numpy()
X_tensor = torch.from_numpy(X_norm)
with torch.no_grad():
    pred_norm = model(X_tensor).cpu().numpy()
pred_raw = pred_norm * y_std + y_mean   # (N, 36)

# ================== 诊断：打印前几帧的角度范围 ==================
print("前10帧关节角度统计:")
for f in range(min(10, frames_to_show)):
    dof = pred_raw[f, 7:36]
    print(f" 帧{f}: min={dof.min():.2f}, max={dof.max():.2f}, mean={dof.mean():.2f}")

# ================== 动画准备 ==================
bone_pairs = [
    (0,1), (1,2), (2,3),
    (2,4), (4,5), (5,6), (6,7), (7,8), (8,9), (9,10),
    (2,11), (11,12), (12,13), (13,14), (14,15), (15,16), (16,17),
    (0,18), (18,19), (19,20), (20,21), (21,22), (22,23),
    (0,24), (24,25), (25,26), (26,27), (27,28), (28,29)
]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_title('Retargeted Robot Motion (with root rotation)')
lines = [ax.plot([], [], [], 'o-', lw=2, markersize=4)[0] for _ in bone_pairs]

# 预先计算所有帧的全局位置范围
all_global_points = []
for frame in range(frames_to_show):
    dof = pred_raw[frame, 7:36]
    root_pos = pred_raw[frame, 0:3]
    # 根旋转四元数（注意顺序：数据可能是 [w,x,y,z]，我们的转换函数需要 w 在前）
    root_rot = pred_raw[frame, 3:7]   # 假设顺序为 [w, x, y, z] ？需确认
    # 如果你的四元数顺序是 [x,y,z,w]，需调整：
    # root_rot = np.array([root_rot[3], root_rot[0], root_rot[1], root_rot[2]])
    R_root = quat_to_matrix(root_rot)
    pos_local = build_skeleton(dof)
    # 应用根旋转 + 平移
    pos_global = (R_root @ pos_local.T).T + root_pos
    all_global_points.append(pos_global)

all_global_points = np.concatenate(all_global_points, axis=0)
margin = 0.5
ax.set_xlim(all_global_points[:,0].min()-margin, all_global_points[:,0].max()+margin)
ax.set_ylim(all_global_points[:,1].min()-margin, all_global_points[:,1].max()+margin)
ax.set_zlim(all_global_points[:,2].min()-margin, all_global_points[:,2].max()+margin)

def update(frame):
    pos = all_global_points[frame*30:(frame+1)*30]   # 每帧30个关节点
    for i, (p1, p2) in enumerate(bone_pairs):
        lines[i].set_data([pos[p1,0], pos[p2,0]], [pos[p1,1], pos[p2,1]])
        lines[i].set_3d_properties([pos[p1,2], pos[p2,2]])
    return lines

ani = FuncAnimation(fig, update, frames=frames_to_show, interval=33, blit=False)
plt.show()