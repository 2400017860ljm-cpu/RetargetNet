import numpy as np, os, glob

OUT_DIR = "exported"
TRANSITION_FRAMES = 30      # 过渡帧数
ADD_STAND_STILL = True      # 是否在动作间插入站立过渡

def lerp_quats(q1, q2, t):
    """线性插值两个四元数（归一化），q1, q2 均为形状 (4,) 的一维数组"""
    dot = np.dot(q1, q2)
    if dot < 0:
        q2 = -q2
    res = (1 - t) * q1 + t * q2
    return res / np.linalg.norm(res)

def linear_interp(v1, v2, t):
    return (1 - t) * v1 + t * v2

anim_files = sorted(glob.glob(os.path.join(OUT_DIR, "anim_*.npz")))
print(f"准备合并 {len(anim_files)} 个动作片段")

all_rp, all_rr, all_dp = [], [], []

stand_rp, stand_rr, stand_dp = None, None, None

for idx, fname in enumerate(anim_files):
    data = np.load(fname)
    rp = data["root_pos"]   # (N,3)
    rr = data["root_rot"]   # (N,4) xyzw
    dp = data["dof_pos"]    # (N,29)

    if idx == 0:
        # 中立站姿：取第一个动作的第一帧（一维数组）
        stand_rp = rp[0].copy()
        stand_rr = rr[0].copy()
        stand_dp = dp[0].copy()

    # 在开始新动作之前，如果需要站立过渡
    if ADD_STAND_STILL and stand_rp is not None:
        if idx > 0:
            # 上一段的最后一帧（从列表最后一个元素中取索引 -1）
            last_rp = all_rp[-1][-1] if all_rp[-1].ndim == 2 else all_rp[-1]
            last_rr = all_rr[-1][-1] if all_rr[-1].ndim == 2 else all_rr[-1]
            last_dp = all_dp[-1][-1] if all_dp[-1].ndim == 2 else all_dp[-1]
            for f in range(1, TRANSITION_FRAMES + 1):
                t = f / (TRANSITION_FRAMES + 1)
                all_rp.append(linear_interp(last_rp, stand_rp, t))
                all_rr.append(lerp_quats(last_rr, stand_rr, t))
                all_dp.append(linear_interp(last_dp, stand_dp, t))
        # 站定 10 帧（直接添加中立姿态的拷贝）
        for _ in range(10):
            all_rp.append(stand_rp.copy())
            all_rr.append(stand_rr.copy())
            all_dp.append(stand_dp.copy())

    # 过渡到当前动作的第一帧
    if len(all_rp) > 0:
        last_rp = all_rp[-1][-1] if all_rp[-1].ndim == 2 else all_rp[-1]
        last_rr = all_rr[-1][-1] if all_rr[-1].ndim == 2 else all_rr[-1]
        last_dp = all_dp[-1][-1] if all_dp[-1].ndim == 2 else all_dp[-1]
        for f in range(1, TRANSITION_FRAMES + 1):
            t = f / (TRANSITION_FRAMES + 1)
            all_rp.append(linear_interp(last_rp, rp[0], t))
            all_rr.append(lerp_quats(last_rr, rr[0], t))
            all_dp.append(linear_interp(last_dp, dp[0], t))

    # 添加当前动作的所有帧
    all_rp.append(rp)
    all_rr.append(rr)
    all_dp.append(dp)

# 拼接所有帧：列表中的元素可能是单个帧 (1D) 或片段 (2D)，统一为 2D 再拼接
def stack(arr_list):
    arrays = [x if x.ndim == 2 else x.reshape(1, -1) for x in arr_list]
    return np.concatenate(arrays, axis=0)

merged_rp = stack(all_rp)
merged_rr = stack(all_rr)
merged_dp = stack(all_dp)

np.savez_compressed("merged_female1_general.npz",
                    root_pos=merged_rp,
                    root_rot=merged_rr,
                    dof_pos=merged_dp)
print(f"合并完成，总帧数: {len(merged_rp)}")