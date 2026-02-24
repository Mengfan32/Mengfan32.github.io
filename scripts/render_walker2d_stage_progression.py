#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np

from render_walker2d_simulation import load_policy_params, parse_rewards, render_episode


def parse_stage_ranges(spec: str):
    ranges = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        start_s, end_s = chunk.split("-")
        start = int(start_s)
        end = int(end_s)
        if end < start:
            raise ValueError(f"Invalid stage range: {chunk}")
        ranges.append((start, end))
    if not ranges:
        raise ValueError("No valid stage ranges provided.")
    return ranges


def mean_reward_of_episode(log_root: Path, episode: int):
    rollout = log_root / f"episode_{episode}" / "training_rollout.txt"
    if not rollout.exists():
        return None
    rewards = parse_rewards(rollout)
    if not rewards:
        return None
    return float(np.mean(rewards))


def pick_episode_for_stage(log_root: Path, start: int, end: int):
    candidates = []
    for ep in range(start, end + 1):
        m = mean_reward_of_episode(log_root, ep)
        if m is None:
            continue
        candidates.append((ep, m))
    if not candidates:
        return None, None
    # Representative episode for the stage: best mean reward.
    ep, m = max(candidates, key=lambda x: x[1])
    return ep, m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, default=Path("logs/walker2d_props"))
    parser.add_argument("--env-name", type=str, default="Walker2d-v5")
    parser.add_argument("--dim-states", type=int, default=17)
    parser.add_argument("--dim-actions", type=int, default=6)
    parser.add_argument("--bias", action="store_true", default=True)
    parser.add_argument("--max-steps", type=int, default=450)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument(
        "--stages",
        type=str,
        default="0-15,16-25,26-77,78-117",
        help="Comma-separated inclusive ranges",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("logs/walker2d_plots"))
    args = parser.parse_args()

    stage_ranges = parse_stage_ranges(args.stages)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for idx, (start, end) in enumerate(stage_ranges, start=1):
        ep, mean_r = pick_episode_for_stage(args.log_root, start, end)
        if ep is None:
            print(f"Stage {idx} ({start}-{end}): skipped (no valid episode data)")
            continue

        params_file = args.log_root / f"episode_{ep}" / "parameters.txt"
        weight, bias_vec = load_policy_params(
            params_file, args.dim_states, args.dim_actions, bias=args.bias
        )

        out_gif = args.out_dir / f"walker2d_stage_{idx}_{start}_{end}_episode_{ep}.gif"
        out_png = args.out_dir / f"walker2d_stage_{idx}_{start}_{end}_episode_{ep}_preview.png"
        final_reward, frames = render_episode(
            args.env_name,
            weight,
            bias_vec,
            ep,
            args.max_steps,
            args.fps,
            out_gif,
            out_png,
        )

        print(
            f"Stage {idx} ({start}-{end}) -> episode {ep}, "
            f"stage_mean={mean_r:.2f}, sim_final_reward={final_reward:.2f}, frames={frames}"
        )
        summary_rows.append(
            {
                "stage": idx,
                "range_start": start,
                "range_end": end,
                "selected_episode": ep,
                "selected_episode_mean_reward": f"{mean_r:.6f}",
                "simulation_final_reward": f"{final_reward:.6f}",
                "frames": frames,
                "gif": str(out_gif),
                "preview": str(out_png),
            }
        )

    if summary_rows:
        summary_csv = args.out_dir / "walker2d_stage_progression_summary.csv"
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "stage",
                    "range_start",
                    "range_end",
                    "selected_episode",
                    "selected_episode_mean_reward",
                    "simulation_final_reward",
                    "frames",
                    "gif",
                    "preview",
                ],
            )
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Saved summary: {summary_csv}")


if __name__ == "__main__":
    main()
