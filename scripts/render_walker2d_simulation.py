#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


TOTAL_REWARD_PATTERN = re.compile(r"Total reward:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
FLOAT_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
PHASE_PERIOD = 32


def parse_rewards(training_rollout_path: Path):
    text = training_rollout_path.read_text(encoding="utf-8", errors="ignore")
    return [float(x) for x in TOTAL_REWARD_PATTERN.findall(text)]


def find_best_episode(log_root: Path):
    best_ep = None
    best_mean = -np.inf
    for ep_dir in sorted(log_root.glob("episode_*")):
        if not ep_dir.is_dir():
            continue
        try:
            ep = int(ep_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        rollout = ep_dir / "training_rollout.txt"
        if not rollout.exists():
            continue
        rewards = parse_rewards(rollout)
        if not rewards:
            continue
        mean_r = float(np.mean(rewards))
        if mean_r > best_mean:
            best_mean = mean_r
            best_ep = ep
    if best_ep is None:
        raise RuntimeError(f"No valid episodes found in {log_root}")
    return best_ep


def load_policy_params(
    parameters_txt: Path,
    dim_states: int,
    dim_actions: int,
    bias: bool,
    policy_type: str = "linear",
    mlp_hidden_dim: int = 8,
):
    text = parameters_txt.read_text(encoding="utf-8", errors="ignore")
    values = [float(x) for x in FLOAT_PATTERN.findall(text)]
    if policy_type == "mlp":
        expected = (
            dim_states * mlp_hidden_dim
            + mlp_hidden_dim
            + mlp_hidden_dim * dim_actions
            + dim_actions
        )
    else:
        expected = (dim_states + (1 if bias else 0)) * dim_actions
    if len(values) < expected:
        raise RuntimeError(
            f"Not enough parameters in {parameters_txt}: got {len(values)}, expected {expected}"
        )
    arr = np.array(values[:expected], dtype=np.float32)
    if policy_type == "mlp":
        cursor = 0
        w1_size = dim_states * mlp_hidden_dim
        b1_size = mlp_hidden_dim
        w2_size = mlp_hidden_dim * dim_actions
        w1 = arr[cursor : cursor + w1_size].reshape(dim_states, mlp_hidden_dim)
        cursor += w1_size
        b1 = arr[cursor : cursor + b1_size]
        cursor += b1_size
        w2 = arr[cursor : cursor + w2_size].reshape(mlp_hidden_dim, dim_actions)
        cursor += w2_size
        b2 = arr[cursor : cursor + dim_actions]
        return {
            "policy_type": "mlp",
            "w1": w1,
            "b1": b1,
            "w2": w2,
            "b2": b2,
            "policy_dim_states": dim_states,
        }
    if bias:
        arr = arr.reshape(dim_states + 1, dim_actions)
        weight = arr[:-1, :]
        bias_vec = arr[-1, :]
    else:
        weight = arr.reshape(dim_states, dim_actions)
        bias_vec = np.zeros((dim_actions,), dtype=np.float32)
    return {
        "policy_type": "linear",
        "weight": weight,
        "bias_vec": bias_vec,
        "policy_dim_states": dim_states,
    }


def overlay_text(frame: np.ndarray, episode: int, reward: float):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        title_font = ImageFont.truetype("Arial.ttf", 54)
        sub_font = ImageFont.truetype("Arial.ttf", 48)
    except OSError:
        title_font = ImageFont.load_default()
        sub_font = ImageFont.load_default()

    box_h = 140
    draw.rectangle([(0, 0), (img.width, box_h)], fill=(245, 245, 245, 245))
    draw.text((18, 14), f"Episode {episode}", fill=(58, 58, 58), font=title_font)
    draw.text((18, 76), f"Reward: {reward:.2f}", fill=(58, 58, 58), font=sub_font)
    return np.array(img)


def augment_state_for_policy(state: np.ndarray, step: int, policy_dim_states: int):
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape[0] == policy_dim_states:
        return state
    if state.shape[0] + 2 == policy_dim_states:
        phase = 2.0 * np.pi * (step % PHASE_PERIOD) / PHASE_PERIOD
        return np.concatenate(
            (state, np.array([np.sin(phase), np.cos(phase)], dtype=np.float32))
        )
    raise RuntimeError(
        f"State dim mismatch for rendering: state={state.shape[0]}, policy={policy_dim_states}"
    )


def render_episode(
    env_name: str,
    policy_params: dict,
    episode: int,
    max_steps: int,
    fps: int,
    out_gif: Path,
    out_png: Path,
):
    env = gym.make(env_name, render_mode="rgb_array")
    state, _ = env.reset(seed=episode)
    done = False
    step = 0
    total_reward = 0.0
    frames = []

    while not done and step < max_steps:
        frame = env.render()
        if frame is not None:
            frames.append(overlay_text(frame, episode, total_reward))

        policy_state = augment_state_for_policy(
            state, step, policy_params["policy_dim_states"]
        )
        if policy_params["policy_type"] == "mlp":
            hidden = np.tanh(
                np.matmul(policy_state, policy_params["w1"]) + policy_params["b1"]
            )
            action = np.tanh(
                np.matmul(hidden, policy_params["w2"]) + policy_params["b2"]
            )
        else:
            action = (
                np.matmul(policy_state, policy_params["weight"])
                + policy_params["bias_vec"]
            )
        state, reward, terminated, truncated, _ = env.step(action.astype(np.float32))
        total_reward += float(reward)
        done = bool(terminated or truncated)
        step += 1

    frame = env.render()
    if frame is not None:
        frames.append(overlay_text(frame, episode, total_reward))
    env.close()

    if not frames:
        raise RuntimeError("No frames rendered. Check environment/render setup.")

    out_gif.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_gif, frames, fps=fps, loop=0)
    imageio.imwrite(out_png, frames[-1])
    return total_reward, len(frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, default=Path("logs/walker2d_props"))
    parser.add_argument("--episode", type=int, default=-1, help="-1 means best by mean reward")
    parser.add_argument("--env-name", type=str, default="Walker2d-v5")
    parser.add_argument("--dim-states", type=int, default=19)
    parser.add_argument("--dim-actions", type=int, default=6)
    parser.add_argument("--bias", action="store_true", default=True)
    parser.add_argument("--policy-type", type=str, default="linear", choices=["linear", "mlp"])
    parser.add_argument("--mlp-hidden-dim", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--out-dir", type=Path, default=Path("logs/walker2d_plots"))
    args = parser.parse_args()

    episode = args.episode if args.episode >= 0 else find_best_episode(args.log_root)
    params_file = args.log_root / f"episode_{episode}" / "parameters.txt"
    if not params_file.exists():
        raise RuntimeError(f"Missing file: {params_file}")

    policy_params = load_policy_params(
        params_file,
        args.dim_states,
        args.dim_actions,
        bias=args.bias,
        policy_type=args.policy_type,
        mlp_hidden_dim=args.mlp_hidden_dim,
    )

    out_gif = args.out_dir / f"walker2d_props_episode_{episode}_simulation.gif"
    out_png = args.out_dir / f"walker2d_props_episode_{episode}_preview.png"
    total_reward, n_frames = render_episode(
        args.env_name,
        policy_params,
        episode,
        args.max_steps,
        args.fps,
        out_gif,
        out_png,
    )

    print(f"Episode used: {episode}")
    print(f"Frames: {n_frames}")
    print(f"Final reward: {total_reward:.2f}")
    print(f"Saved GIF: {out_gif}")
    print(f"Saved preview: {out_png}")


if __name__ == "__main__":
    main()
