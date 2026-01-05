import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.colors as mcolors
import cv2
from scipy.ndimage import gaussian_filter

output_dir = 'B0'
os.makedirs(output_dir, exist_ok=True)
canvas_size = (800, 800)
colors = ["#210057", "#42FF70", "#FFF070", "#FF5170"]
cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
obs_df = pd.read_csv('observation.csv')
obs_df['step'] = obs_df['step'].astype(int)
head_dfs = []
for i in range(4):
    head_df = pd.read_csv(f'block0_head{i}.csv')
    head_dfs.append(head_df)
all_steps = head_dfs[0]['step'].unique()
for head_idx, head_df in enumerate(head_dfs):
    print(f"Processing head {head_idx}")
    img_array = []
    head_output_dir = os.path.join(output_dir, f'head{head_idx}')
    os.makedirs(head_output_dir, exist_ok=True)

    for step in tqdm(all_steps, desc=f"Head {head_idx}"):
        obs_grid = np.zeros(canvas_size, dtype=np.float32)
        target_grid = np.zeros(canvas_size, dtype=np.float32)
        obst_grid = np.zeros(canvas_size, dtype=np.float32)

        memory_start = max(0, step - 127)
        available_steps = step - memory_start + 1

        current_attention_df = head_df[head_df['step'] == step]
        if current_attention_df.empty:
            continue
        current_attention = current_attention_df.iloc[0]

        for i in range(available_steps):
            memory_step = step - i
            key_idx = available_steps - 1 - i
            key_col = f'Key {key_idx}'
            weight = float(current_attention[key_col])  # <- 不做平均!!!
            obs_row_df = obs_df[obs_df['step'] == memory_step]
            if obs_row_df.empty:
                continue
            obs_row = obs_row_df.iloc[0]
            # agent
            pos_x = int(float(obs_row['obs_x']))
            pos_y = int(float(obs_row['obs_y']))
            pos_w = int(float(obs_row['obs_w']))
            pos_h = int(float(obs_row['obs_h']))
            pos_r = max(pos_w, pos_h) / 2.0
            # target
            target_x = int(float(obs_row['target_x']))
            target_y = int(float(obs_row['target_y']))
            obs_rx = pos_r
            obs_ry = pos_r
            target_rx = 10
            target_ry = 10
            # agent区域
            obs_x_min = max(0, int(pos_x - obs_rx))
            obs_x_max = min(canvas_size[0] - 1, int(pos_x + obs_rx))
            obs_y_min = max(0, int(pos_y - obs_ry))
            obs_y_max = min(canvas_size[1] - 1, int(pos_y + obs_ry))
            obs_x_range = np.arange(obs_x_min, obs_x_max + 1)
            obs_y_range = np.arange(obs_y_min, obs_y_max + 1)
            X_obs, Y_obs = np.meshgrid(obs_x_range, obs_y_range)
            dx_obs = X_obs - pos_x
            dy_obs = Y_obs - pos_y
            dist_obs = np.sqrt(dx_obs ** 2 + dy_obs ** 2)
            mask_obs = ((dx_obs / obs_rx) ** 2 + (dy_obs / obs_ry) ** 2) <= 1
            obs_contrib = np.exp(-0.5 * (dist_obs / obs_rx) ** 2) * weight
            obs_grid[obs_y_min:obs_y_max + 1, obs_x_min:obs_x_max + 1] += obs_contrib * mask_obs

            # target区域
            target_x_min = max(0, int(target_x - target_rx))
            target_x_max = min(canvas_size[0] - 1, int(target_x + target_rx))
            target_y_min = max(0, int(target_y - target_ry))
            target_y_max = min(canvas_size[1] - 1, int(target_y + target_ry))
            target_x_range = np.arange(target_x_min, target_x_max + 1)
            target_y_range = np.arange(target_y_min, target_y_max + 1)
            X_target, Y_target = np.meshgrid(target_x_range, target_y_range)
            dx_target = X_target - target_x
            dy_target = Y_target - target_y
            dist_target = np.sqrt(dx_target ** 2 + dy_target ** 2)
            mask_target = dist_target <= target_rx
            target_contrib = np.exp(-0.5 * (dist_target / target_rx) ** 2) * weight
            target_grid[target_y_min:target_y_max + 1, target_x_min:target_x_max + 1] += target_contrib * mask_target

            # 遍历障碍物
            for obst_idx in range(1, 6):
                obst_x = float(obs_row.get(f'obst{obst_idx}_x', 0))
                obst_y = float(obs_row.get(f'obst{obst_idx}_y', 0))
                if obst_x <= 1e-1 and obst_y <= 1e-1:
                    continue
                obst_w = float(obs_row.get(f'obst{obst_idx}_w', 0))
                obst_h = float(obs_row.get(f'obst{obst_idx}_h', 0))
                obst_shape = float(obs_row.get(f'obst{obst_idx}_shape', 0))
                rx = obst_w / 2.0
                ry = obst_h / 2.0
                x_min = max(0, int(obst_x - rx))
                x_max = min(canvas_size[0] - 1, int(obst_x + rx))
                y_min = max(0, int(obst_y - ry))
                y_max = min(canvas_size[1] - 1, int(obst_y + ry))
                if obst_shape <= 0.5:  # 矩形
                    obst_grid[y_min:y_max + 1, x_min:x_max + 1] += weight
                else:  # 椭圆
                    x_range = np.arange(x_min, x_max + 1)
                    y_range = np.arange(y_min, y_max + 1)
                    X_obst, Y_obst = np.meshgrid(x_range, y_range)
                    dx_obst = X_obst - obst_x
                    dy_obst = Y_obst - obst_y
                    mask_obst = ((dx_obst / rx) ** 2 + (dy_obst / ry) ** 2) <= 1
                    obst_grid[y_min:y_max + 1, x_min:x_max + 1] += weight * mask_obst

                    # 平滑和合成
        smooth_obs_grid = gaussian_filter(obs_grid, sigma=5)
        smooth_target_grid = gaussian_filter(target_grid, sigma=5)
        integrate_grid = smooth_obs_grid + obst_grid + smooth_target_grid
        integrate_grid[integrate_grid > 1] = 1

        # 可视化
        fig, ax = plt.subplots(figsize=(16, 16))
        ax.set_xlim(0, canvas_size[0])
        ax.set_ylim(0, canvas_size[1])
        ax.imshow(integrate_grid, cmap=cmap, extent=(0, canvas_size[0], 0, canvas_size[1]),
                  origin='upper', alpha=1, vmin=0, vmax=1)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        # 输出文件名区分head
        save_path = os.path.join(head_output_dir, f'step{step:04d}_head{head_idx}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        img = cv2.imread(save_path)
        if img is not None:
            img_array.append(img)
        else:
            print(f"Warning: Could not read image {save_path}")

            # 为每个head输出mp4
    video_path = os.path.join(output_dir, f'B0_head{head_idx}.mp4')
    if img_array:
        height, width, _ = img_array[0].shape
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
        for img in img_array:
            out.write(img)
        out.release()
        print(f"Video for head-{head_idx} created successfully:", video_path)
    else:
        print(f"No frames to create video for head-{head_idx}")