import numpy as np
import matplotlib.pyplot as plt
import glob

# 找一个包含明显步态的动作文件（优先选择名字含"Walk"的）
all_npz = sorted(glob.glob("data/vector_pairs/**/*.npz", recursive=True))
walk_files = [f for f in all_npz if 'walk' in f.lower() or 'Walk' in f]
npz_file = walk_files[0] if walk_files else all_npz[0]
print(f"使用文件: {npz_file}")

data = np.load(npz_file)
Y = np.concatenate([data["output_root_pos"],
                    data["output_root_rot"],
                    data["output_dof_pos"]], axis=1).astype(np.float32)

# 只取前300帧（步态循环通常几十帧）
Y = Y[:300]
dof = Y[:, 7:36]   # (N, 29)

# 打印每个维度的活动范围（最大-最小），辅助识别
ranges = dof.max(axis=0) - dof.min(axis=0)
print("各关节活动范围（度）:")
for i, r in enumerate(ranges):
    print(f"  索引 {i}: {np.rad2deg(r):.1f}°")

# 绘制所有29个关节的曲线（分6个子图，便于观察）
fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
for idx in range(29):
    ax = axes[idx // 6] if idx < 24 else axes[4]
    ax.plot(dof[:, idx], label=f'Idx {idx}')
    ax.set_ylabel('Angle (rad)')
axes[0].set_title('All 29 DOF angles (walking motion)')
axes[-1].set_xlabel('Frame')
plt.tight_layout()
plt.show()