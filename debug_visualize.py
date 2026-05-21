import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import glob

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

def quat_to_matrix(q):
    """四元数 (w,x,y,z) -> 3x3旋转矩阵"""
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
        [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]
    ])

# ================== 加载真实标签 ==================
all_npz = sorted(glob.glob("data/vector_pairs/**/*.npz", recursive=True))
npz_file = all_npz[0]
print(f"使用文件: {npz_file}")

data = np.load(npz_file)
Y = np.concatenate([data["output_root_pos"],
                    data["output_root_rot"],
                    data["output_dof_pos"]], axis=1).astype(np.float32)

frames = min(300, len(Y))
Y = Y[:frames]

# 打印前10帧根旋转四元数（用于确认顺序）
print("\n前10帧根旋转 (索引3:7) :")
for f in range(min(10, frames)):
    q = Y[f, 3:7]
    print(f"  帧{f}: {np.round(q, 3)}")

# ================== 动画 ==================
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
ax.set_title('Ground Truth Motion (without root rotation applied)')
lines = [ax.plot([], [], [], 'o-', lw=2, markersize=4)[0] for _ in bone_pairs]

# 计算所有全局点
all_global_points = []
for f in range(frames):
    dof = Y[f, 7:36]
    root_pos = Y[f, 0:3]
    root_rot = Y[f, 3:7]   # 假设顺序为 [w,x,y,z]
    R = quat_to_matrix(root_rot)
    pos_local = build_skeleton(dof)
    pos_global = (R @ pos_local.T).T + root_pos
    all_global_points.append(pos_global)

all_global_points = np.concatenate(all_global_points, axis=0)
margin = 0.5
ax.set_xlim(all_global_points[:,0].min()-margin, all_global_points[:,0].max()+margin)
ax.set_ylim(all_global_points[:,1].min()-margin, all_global_points[:,1].max()+margin)
ax.set_zlim(all_global_points[:,2].min()-margin, all_global_points[:,2].max()+margin)

def update(frame):
    pos = all_global_points[frame*30:(frame+1)*30]
    for i, (p1, p2) in enumerate(bone_pairs):
        lines[i].set_data([pos[p1,0], pos[p2,0]], [pos[p1,1], pos[p2,1]])
        lines[i].set_3d_properties([pos[p1,2], pos[p2,2]])
    return lines

ani = FuncAnimation(fig, update, frames=frames, interval=33, blit=False)
plt.show()