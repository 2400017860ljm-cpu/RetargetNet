import numpy as np

data = np.load("data/vector_pairs/Male1General_c3d/General_A1_-_Stand_stageii_pairs.npz")

# 输入向量 (186帧, 69维) = body pose(63) + root orient(3) + trans(3)
X = np.concatenate([data["input_pose_body"],
                    data["input_root_orient"],
                    data["input_trans"]], axis=1)

# 输出向量 (186帧, 36维) = root pos(3) + root rot(4) + joint angles(29)
Y = np.concatenate([data["output_root_pos"],
                    data["output_root_rot"],
                    data["output_dof_pos"]], axis=1)

print(f"输入向量 X: {X.shape}")   # (186, 69)
print(f"输出向量 Y: {Y.shape}")   # (186, 36)

# 可以输出任意的帧
print(f"\n第5帧:")
print(f"  X[5] (69维人体参数):\n    {X[5]}")
print(f"  Y[5] (36维机器人目标):\n    {Y[5]}")