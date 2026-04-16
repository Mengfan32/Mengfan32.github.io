#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


FLOAT_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
TOTAL_REWARD_PATTERN = re.compile(r"Total reward:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def load_linear_policy(parameters_txt: Path, dim_states: int, dim_actions: int):
    values = [
        float(x)
        for x in FLOAT_PATTERN.findall(
            parameters_txt.read_text(encoding="utf-8", errors="ignore")
        )
    ]
    expected = (dim_states + 1) * dim_actions
    if len(values) < expected:
        raise RuntimeError(
            f"Not enough parameters in {parameters_txt}: got {len(values)}, expected {expected}"
        )
    params = np.array(values[:expected], dtype=np.float32).reshape(dim_states + 1, dim_actions)
    return params[:-1], params[-1]


def mean_reward(episode_dir: Path):
    rollout = episode_dir / "training_rollout.txt"
    if not rollout.exists():
        return float("-inf")
    rewards = [
        float(x)
        for x in TOTAL_REWARD_PATTERN.findall(
            rollout.read_text(encoding="utf-8", errors="ignore")
        )
    ]
    return float(np.mean(rewards)) if rewards else float("-inf")


def list_episodes(log_root: Path):
    episodes = []
    for episode_dir in log_root.glob("episode_*"):
        try:
            episode = int(episode_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if (episode_dir / "parameters.txt").exists():
            episodes.append((episode, mean_reward(episode_dir)))
    return sorted(episodes)


def select_representative_episodes(log_root: Path, count: int):
    episodes = list_episodes(log_root)
    if not episodes:
        raise RuntimeError(f"No episodes found in {log_root}")
    if count >= len(episodes):
        return [episode for episode, _ in episodes]
    indices = np.linspace(0, len(episodes) - 1, count, dtype=int)
    selected = {episodes[i][0] for i in indices}
    selected.add(max(episodes, key=lambda x: x[1])[0])
    return sorted(selected)


def overlay_text(frame: np.ndarray, title: str, reward: float, x_pos: float):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        title_font = ImageFont.truetype("Arial.ttf", 36)
        sub_font = ImageFont.truetype("Arial.ttf", 28)
    except OSError:
        title_font = ImageFont.load_default()
        sub_font = ImageFont.load_default()
    draw.rectangle([(0, 0), (img.width, 92)], fill=(245, 245, 245, 230))
    draw.text((14, 10), title, fill=(35, 35, 35), font=title_font)
    draw.text((14, 52), f"reward={reward:.2f}  x={x_pos:.2f}", fill=(35, 35, 35), font=sub_font)
    return np.array(img)


def render_episode(env_name, weight, bias, episode, max_steps, frame_stride):
    env = gym.make(env_name, render_mode="rgb_array")
    state, _ = env.reset(seed=episode)
    frames = []
    x_positions = []
    total_reward = 0.0

    for step in range(max_steps):
        x_pos = float(env.unwrapped.data.qpos[0])
        x_positions.append(x_pos)
        if step % frame_stride == 0:
            frame = env.render()
            if frame is not None:
                frames.append(overlay_text(frame, f"Swimmer episode {episode}", total_reward, x_pos))
        action = np.matmul(np.asarray(state, dtype=np.float32), weight) + bias
        state, reward, terminated, truncated, _ = env.step(action.astype(np.float32))
        total_reward += float(reward)
        if terminated or truncated:
            break

    frame = env.render()
    if frame is not None:
        frames.append(
            overlay_text(frame, f"Swimmer episode {episode}", total_reward, float(env.unwrapped.data.qpos[0]))
        )
    env.close()
    return frames, total_reward, x_positions


def save_trajectory_plot(out_path: Path, episode_metrics):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for episode, x_positions, reward in episode_metrics:
        ax.plot(x_positions, label=f"ep {episode}, reward={reward:.1f}", linewidth=1.6)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Swimmer x-position trajectories")
    ax.set_xlabel("Simulation step")
    ax.set_ylabel("x position")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, default=Path("logs/swimmer_propsp_gemini"))
    parser.add_argument("--out-dir", type=Path, default=Path("logs/swimmer_gemini_visuals"))
    parser.add_argument("--env-name", type=str, default="Swimmer-v5")
    parser.add_argument("--dim-states", type=int, default=8)
    parser.add_argument("--dim-actions", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=420)
    parser.add_argument("--frame-stride", type=int, default=14)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--best-episode", type=int, default=-1)
    parser.add_argument("--learning-count", type=int, default=80)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    episodes = list_episodes(args.log_root)
    if not episodes:
        raise RuntimeError(f"No episodes found in {args.log_root}")
    best_episode = args.best_episode if args.best_episode >= 0 else max(episodes, key=lambda x: x[1])[0]

    summary_rows = []
    trajectory_metrics = []

    best_params = args.log_root / f"episode_{best_episode}" / "parameters.txt"
    weight, bias = load_linear_policy(best_params, args.dim_states, args.dim_actions)
    frames, reward, x_positions = render_episode(
        args.env_name, weight, bias, best_episode, args.max_steps, args.frame_stride
    )
    best_gif = args.out_dir / f"swimmer_propsp_gemini_episode_{best_episode}_success.gif"
    best_png = args.out_dir / f"swimmer_propsp_gemini_episode_{best_episode}_success_preview.png"
    imageio.mimsave(best_gif, frames, fps=args.fps, loop=0)
    imageio.imwrite(best_png, frames[-1])
    summary_rows.append(
        {
            "episode": best_episode,
            "simulation_reward": f"{reward:.6f}",
            "forward_displacement": f"{(x_positions[-1] - x_positions[0]):.6f}",
            "gif": str(best_gif),
            "preview": str(best_png),
        }
    )
    trajectory_metrics.append((best_episode, x_positions, reward))

    learning_frames = []
    learning_episodes = select_representative_episodes(args.log_root, args.learning_count)
    for episode in learning_episodes:
        params = args.log_root / f"episode_{episode}" / "parameters.txt"
        weight, bias = load_linear_policy(params, args.dim_states, args.dim_actions)
        frames, reward, x_positions = render_episode(
            args.env_name, weight, bias, episode, args.max_steps, args.frame_stride
        )
        learning_frames.extend(frames)
        if episode in {learning_episodes[0], best_episode, learning_episodes[-1]}:
            trajectory_metrics.append((episode, x_positions, reward))

    learning_gif = args.out_dir / "swimmer_propsp_gemini_0_79_learning.gif"
    imageio.mimsave(learning_gif, learning_frames, fps=args.fps, loop=0)

    trajectory_plot = args.out_dir / "swimmer_propsp_gemini_key_episode_x_trajectories.png"
    save_trajectory_plot(trajectory_plot, trajectory_metrics)

    summary_path = args.out_dir / "swimmer_propsp_gemini_visualization_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["episode", "simulation_reward", "forward_displacement", "gif", "preview"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Success GIF: {best_gif}")
    print(f"Success preview: {best_png}")
    print(f"Learning GIF: {learning_gif}")
    print(f"Trajectory plot: {trajectory_plot}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
