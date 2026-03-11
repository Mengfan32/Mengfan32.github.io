#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


FLOAT_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def load_policy_params(parameters_txt: Path, dim_states: int, dim_actions: int, bias: bool):
    text = parameters_txt.read_text(encoding="utf-8", errors="ignore")
    values = [float(x) for x in FLOAT_PATTERN.findall(text)]
    expected = (dim_states + (1 if bias else 0)) * dim_actions
    if len(values) < expected:
        raise RuntimeError(
            f"Not enough parameters in {parameters_txt}: got {len(values)}, expected {expected}"
        )
    arr = np.array(values[:expected], dtype=np.float32)
    if bias:
        arr = arr.reshape(dim_states + 1, dim_actions)
        weight = arr[:-1, :]
        bias_vec = arr[-1, :]
    else:
        weight = arr.reshape(dim_states, dim_actions)
        bias_vec = np.zeros((dim_actions,), dtype=np.float32)
    return weight, bias_vec


def overlay_text(frame: np.ndarray, episode: int, reward: float):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        title_font = ImageFont.truetype("Arial.ttf", 54)
        sub_font = ImageFont.truetype("Arial.ttf", 48)
    except OSError:
        title_font = ImageFont.load_default()
        sub_font = ImageFont.load_default()

    banner_h = 140
    draw.rectangle([(0, 0), (img.width, banner_h)], fill=(245, 245, 245, 245))
    draw.text((18, 14), f"Episode {episode}", fill=(58, 58, 58), font=title_font)
    draw.text((18, 76), f"Reward: {reward:.2f}", fill=(58, 58, 58), font=sub_font)
    return np.array(img)


def render_episode_frames(
    env_name: str,
    weight: np.ndarray,
    bias_vec: np.ndarray,
    episode: int,
    max_steps: int,
):
    env = gym.make(env_name, render_mode="rgb_array")
    state, _ = env.reset(seed=episode)

    frames = []
    total_reward = 0.0
    step = 0
    done = False
    while not done and step < max_steps:
        frame = env.render()
        if frame is not None:
            frames.append(overlay_text(frame, episode, total_reward))

        action = np.matmul(state, weight) + bias_vec
        state, reward, terminated, truncated, _ = env.step(action.astype(np.float32))
        total_reward += float(reward)
        done = bool(terminated or truncated)
        step += 1

    frame = env.render()
    if frame is not None:
        frames.append(overlay_text(frame, episode, total_reward))
    env.close()
    return frames, total_reward


def write_video_or_gif(frames, out_base: Path, fps: int, gif_only: bool = False):
    out_mp4 = out_base.with_suffix(".mp4")
    out_gif = out_base.with_suffix(".gif")

    if gif_only:
        imageio.mimsave(out_gif, frames, fps=fps, loop=0)
        return out_gif, "gif"

    try:
        with imageio.get_writer(out_mp4, fps=fps, codec="libx264", quality=8) as writer:
            for frame in frames:
                writer.append_data(frame)
        return out_mp4, "mp4"
    except Exception:
        imageio.mimsave(out_gif, frames, fps=fps, loop=0)
        return out_gif, "gif"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, default=Path("logs/walker2d_props"))
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--end-episode", type=int, default=15)
    parser.add_argument("--env-name", type=str, default="Walker2d-v5")
    parser.add_argument("--dim-states", type=int, default=17)
    parser.add_argument("--dim-actions", type=int, default=6)
    parser.add_argument("--bias", action="store_true", default=True)
    parser.add_argument("--max-steps-per-episode", type=int, default=220)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--gif-only", action="store_true", help="Force GIF output only")
    parser.add_argument(
        "--out-base",
        type=Path,
        default=Path("logs/walker2d_plots/walker2d_props_first16_learning"),
    )
    args = parser.parse_args()

    all_frames = []
    summary = []

    for episode in range(args.start_episode, args.end_episode + 1):
        params_file = args.log_root / f"episode_{episode}" / "parameters.txt"
        if not params_file.exists():
            print(f"Skip episode {episode}: missing {params_file}")
            continue
        weight, bias_vec = load_policy_params(
            params_file, args.dim_states, args.dim_actions, bias=args.bias
        )
        frames, total_reward = render_episode_frames(
            args.env_name,
            weight,
            bias_vec,
            episode,
            args.max_steps_per_episode,
        )
        if not frames:
            print(f"Skip episode {episode}: no rendered frames")
            continue
        all_frames.extend(frames)
        # brief pause between episodes
        all_frames.extend([frames[-1]] * 7)
        summary.append((episode, total_reward, len(frames)))
        print(
            f"Episode {episode}: reward={total_reward:.2f}, frames={len(frames)}"
        )

    if not all_frames:
        raise RuntimeError("No frames produced.")

    args.out_base.parent.mkdir(parents=True, exist_ok=True)
    out_path, out_kind = write_video_or_gif(
        all_frames, args.out_base, args.fps, gif_only=args.gif_only
    )
    print(f"Saved {out_kind.upper()}: {out_path}")
    print(f"Total frames: {len(all_frames)}")
    print("Episode summary:")
    for ep, rew, n in summary:
        print(f"  episode={ep}, reward={rew:.2f}, frames={n}")


if __name__ == "__main__":
    main()
