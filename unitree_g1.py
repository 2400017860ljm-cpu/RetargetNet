import sys; sys.path.append(".")
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle, glob, joblib

# ================== 修正后的 Unitree G1 运动学模型 ==================
parents_g1 = [
    0,    # 0  LeftHipPitch       -> 骨盆 (0)
    0,    # 1  LeftHipRoll        -> 骨盆 (0)
    1,    # 2  LeftHipYaw         -> 1
    2,    # 3  LeftKnee           -> 2
    3,    # 4  LeftAnklePitch     -> 3
    4,    # 5  LeftAnkleRoll      -> 4
    0,    # 6  RightHipPitch      -> 骨盆 (0)
    6,    # 7  RightHipRoll       -> 6
    7,    # 8  RightHipYaw        -> 7
    8,    # 9  RightKnee          -> 8
    9,    # 10 RightAnklePitch    -> 9
    10,   # 11 RightAnkleRoll     -> 10
    0,    # 12 WaistYaw           -> 骨盆 (0)
    12,   # 13 WaistRoll          -> 12
    13,   # 14 WaistPitch         -> 13
    14,   # 15 LeftShoulderPitch  -> 14 (躯干)
    15,   # 16 LeftShoulderRoll   -> 15
    16,   # 17 LeftShoulderYaw    -> 16
    17,   # 18 LeftElbow          -> 17
    18,   # 19 LeftWristRoll      -> 18
    19,   # 20 LeftWristPitch     -> 19
    20,   # 21 LeftWristYaw       -> 20
    14,   # 22 RightShoulderPitch -> 14 (躯干)
    22,   # 23 RightShoulderRoll  -> 22
    23,   # 24 RightShoulderYaw   -> 23
    24,   # 25 RightElbow         -> 24
    25,   # 26 RightWristRoll     -> 25
    26,   # 27 RightWristPitch    -> 26
    27,   # 28 RightWristYaw      -> 27
]

# 关节相对偏移量 (单位米)
offsets_g1 = np.array([
    [0.10, 0.0, 0.0],    # 0 左髋起点
    [0.0,  0.0, 0.0],    # 1
    [0.0,  0.0, 0.0],    # 2
    [0.0,  0.0, -0.35],  # 3 大腿
    [0.0,  0.0, -0.35],  # 4 小腿
    [0.0,  0.0, -0.10],  # 5 脚
    [-0.10,0.0, 0.0],    # 6 右髋起点
    [0.0,  0.0, 0.0],    # 7
    [0.0,  0.0, 0.0],    # 8
    [0.0,  0.0, -0.35],  # 9
    [0.0,  0.0, -0.35],  # 10
    [0.0,  0.0, -0.10],  # 11
    [0.0,  0.0, 0.20],   # 12 腰
    [0.0,  0.0, 0.0],    # 13
    [0.0,  0.0, 0.30],   # 14 躯干
    [0.20, 0.0, 0.0],    # 15 左肩
    [0.0,  0.0, 0.0],    # 16
    [0.0,  0.0, 0.0],    # 17
    [0.0,  0.0, -0.30],  # 18 上臂
    [0.0,  0.0, -0.25],  # 19 前臂
    [0.0,  0.0, 0.0],    # 20
    [0.0,  0.0, -0.10],  # 21 手
    [-0.20,0.0, 0.0],    # 22 右肩
    [0.0,  0.0, 0.0],    # 23
    [0.0,  0.0, 0.0],    # 24
    [0.0,  0.0, -0.30],  # 25
    [0.0,  0.0, -0.25],  # 26
    [0.0,  0.0, 0.0],    # 27
    [0.0,  0.0, -0.10],  # 28
])

# 旋转轴 (局部坐标系)
axes_g1 = np.array([
    [0,1,0],  # 0  HipPitch
    [1,0,0],  # 1  HipRoll
    [0,0,1],  # 2  HipYaw
    [0,1,0],  # 3  Knee
    [0,1,0],  # 4  AnklePitch
    [1,0,0],  # 5  AnkleRoll
    [0,1,0],  # 6  HipPitch
    [1,0,0],  # 7  HipRoll
    [0,0,1],  # 8  HipYaw
    [0,1,0],  # 9  Knee
    [0,1,0],  # 10 AnklePitch
    [1,0,0],  # 11 AnkleRoll
    [0,0,1],  # 12 WaistYaw
    [1,0,0],  # 13 WaistRoll
    [0,1,0],  # 14 WaistPitch
    [0,1,0],  # 15 ShoulderPitch
    [1,0,0],  # 16 ShoulderRoll
    [0,0,1],  # 17 ShoulderYaw
    [0,1,0],  # 18 Elbow
    [1,0,0],  # 19 WristRoll
    [0,1,0],  # 20 WristPitch
    [0,0,1],  # 21 WristYaw
    [0,1,0],  # 22 ShoulderPitch
    [1,0,0],  # 23 ShoulderRoll
    [0,0,1],  # 24 ShoulderYaw
    [0,1,0],  # 25 Elbow
    [1,0,0],  # 26 WristRoll
    [0,1,0],  # 27 WristPitch
    [0,0,1],  # 28 WristYaw
])

def build_skeleton_g1(dof_angles):
    """根据29个关节角度计算局部坐标系下的关节点位置（未加根旋转）"""
    n_joints = len(parents_g1) + 1   # 29个关节 + 根(0)
    world_pos = np.zeros((n_joints, 3))
    world_rot = [np.eye(3) for _ in range(n_joints)]

    for i in range(1, n_joints):
        parent = parents_g1[i-1]
        axis = axes_g1[i-1]
        angle = dof_angles[i-1]
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        world_rot[i] = world_rot[parent] @ R
        world_pos[i] = world_pos[parent] + world_rot[parent] @ offsets_g1[i-1]
    return world_pos

def quat_to_matrix(q):
    """四元数(w,x,y,z)转旋转矩阵"""
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
        [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]
    ])

# ================== 加载模型和标准化参数 ==================
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

# ================== 加载连续动作 ==================
all_npz = sorted(glob.glob("data/vector_pairs/**/*.npz", recursive=True))
npz_file = all_npz[0]
print(f"使用文件: {npz_file}")

data = np.load(npz_file)
X = np.concatenate([data["input_pose_body"], data["input_root_orient"], data["input_trans"]], axis=1).astype(np.float32)
frames_to_show = min(300, len(X))
X = X[:frames_to_show]

X_norm = (X - x_mean.numpy()) / x_std.numpy()
with torch.no_grad():
    pred_norm = model(torch.from_numpy(X_norm)).cpu().numpy()
pred_raw = pred_norm * y_std + y_mean   # (N, 36)

# ================== 预先计算所有帧的全局位置 ==================
all_global_positions = []
for f in range(frames_to_show):
    dof = pred_raw[f, 7:36]
    root_pos = pred_raw[f, 0:3]
    root_rot = pred_raw[f, 3:7]
    R_root = quat_to_matrix(root_rot)
    pos_local = build_skeleton_g1(dof)
    pos_global = (R_root @ pos_local.T).T + root_pos
    all_global_positions.append(pos_global)

all_global_positions = np.concatenate(all_global_positions, axis=0)

# ================== 骨骼连线定义 ==================
bone_pairs = []
for i in range(1, 30):
    bone_pairs.append((parents_g1[i-1], i))
# 添加额外身体连线
bone_pairs += [(6, 0), (15, 22)]   # 左右髋, 左右肩

# ================== 动画 ==================
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_title('Unitree G1 Retargeted Motion (ResMLP+)')
lines = [ax.plot([], [], [], 'o-', lw=2, markersize=3)[0] for _ in bone_pairs]

margin = 0.5
ax.set_xlim(all_global_positions[:,0].min()-margin, all_global_positions[:,0].max()+margin)
ax.set_ylim(all_global_positions[:,1].min()-margin, all_global_positions[:,1].max()+margin)
ax.set_zlim(all_global_positions[:,2].min()-margin, all_global_positions[:,2].max()+margin)

def update(frame):
    start_idx = frame * 30
    pos = all_global_positions[start_idx:start_idx+30]
    for idx, (p1, p2) in enumerate(bone_pairs):
        lines[idx].set_data([pos[p1,0], pos[p2,0]], [pos[p1,1], pos[p2,1]])
        lines[idx].set_3d_properties([pos[p1,2], pos[p2,2]])
    return lines

ani = FuncAnimation(fig, update, frames=frames_to_show, interval=33, blit=False)
plt.show()