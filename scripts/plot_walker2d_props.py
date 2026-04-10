#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path
from statistics import mean, median


TOTAL_REWARD_PATTERN = re.compile(r"Total reward:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def collect_episode_stats(log_root: Path, n_episodes: int):
    rows = []
    for episode in range(n_episodes):
        rollout_file = log_root / f"episode_{episode}" / "training_rollout.txt"
        if not rollout_file.exists():
            continue
        text = rollout_file.read_text(encoding="utf-8", errors="ignore")
        rewards = [float(x) for x in TOTAL_REWARD_PATTERN.findall(text)]
        if not rewards:
            continue
        rows.append(
            {
                "episode": episode,
                "rollout_count": len(rewards),
                "max_reward": max(rewards),
                "mean_reward": mean(rewards),
                "median_reward": median(rewards),
            }
        )
    return rows


def scale(val, src_min, src_max, dst_min, dst_max):
    if src_max == src_min:
        return (dst_min + dst_max) / 2
    return dst_min + (val - src_min) * (dst_max - dst_min) / (src_max - src_min)


def polyline(xs, ys):
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, default=Path("logs/walker2d_props"))
    parser.add_argument("--n-episodes", type=int, default=117)
    parser.add_argument("--tag", type=str, default="new_117")
    parser.add_argument("--output-prefix", type=str, default="walker2d_props")
    parser.add_argument("--title", type=str, default="Walker2d ProPS: New Run")
    args = parser.parse_args()

    log_root = args.log_root
    n_episodes = args.n_episodes
    out_csv = Path(
        f"logs/walker2d_plots/{args.output_prefix}_{args.tag}_episode_rewards.csv"
    )
    out_svg = Path(
        f"logs/walker2d_plots/{args.output_prefix}_{args.tag}_reward_curve.svg"
    )

    rows = collect_episode_stats(log_root, n_episodes)
    if not rows:
        raise SystemExit(f"No episode rewards found in {log_root}.")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "rollout_count",
                "max_reward",
                "mean_reward",
                "median_reward",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    episodes = [r["episode"] for r in rows]
    max_rewards = [r["max_reward"] for r in rows]
    mean_rewards = [r["mean_reward"] for r in rows]
    median_rewards = [r["median_reward"] for r in rows]
    all_rewards = max_rewards + mean_rewards + median_rewards

    width = 1400
    height = 760
    margin_left = 90
    margin_right = 40
    margin_top = 70
    margin_bottom = 80

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    x_min, x_max = min(episodes), max(episodes)
    y_min, y_max = min(all_rewards), max(all_rewards)
    y_pad = (y_max - y_min) * 0.08 if y_max > y_min else 1.0
    y_min -= y_pad
    y_max += y_pad

    x_coords = [
        scale(ep, x_min, x_max, margin_left, margin_left + plot_w) for ep in episodes
    ]
    y_max_coords = [
        scale(v, y_min, y_max, margin_top + plot_h, margin_top) for v in max_rewards
    ]
    y_mean_coords = [
        scale(v, y_min, y_max, margin_top + plot_h, margin_top) for v in mean_rewards
    ]
    y_median_coords = [
        scale(v, y_min, y_max, margin_top + plot_h, margin_top) for v in median_rewards
    ]
    y_zero = scale(0.0, y_min, y_max, margin_top + plot_h, margin_top)

    ticks = 6
    y_tick_vals = [y_min + i * (y_max - y_min) / ticks for i in range(ticks + 1)]

    lines = []
    lines.append(
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#333" stroke-width="1"/>'
    )
    lines.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#333" stroke-width="1"/>'
    )
    lines.append(
        f'<line x1="{margin_left}" y1="{y_zero:.2f}" x2="{margin_left + plot_w}" y2="{y_zero:.2f}" stroke="#777" stroke-width="1" stroke-dasharray="6,5"/>'
    )

    for yv in y_tick_vals:
        yy = scale(yv, y_min, y_max, margin_top + plot_h, margin_top)
        lines.append(
            f'<line x1="{margin_left}" y1="{yy:.2f}" x2="{margin_left + plot_w}" y2="{yy:.2f}" stroke="#e6e6e6" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{margin_left - 10}" y="{yy + 5:.2f}" font-size="14" text-anchor="end" fill="#444">{yv:.1f}</text>'
        )

    x_tick_vals = [int(round(x_min + i * (x_max - x_min) / ticks)) for i in range(ticks + 1)]
    for xv in x_tick_vals:
        xx = scale(xv, x_min, x_max, margin_left, margin_left + plot_w)
        lines.append(
            f'<line x1="{xx:.2f}" y1="{margin_top}" x2="{xx:.2f}" y2="{margin_top + plot_h}" stroke="#f0f0f0" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{xx:.2f}" y="{margin_top + plot_h + 25}" font-size="14" text-anchor="middle" fill="#444">{xv}</text>'
        )

    lines.append(
        f'<polyline fill="none" stroke="#0b84a5" stroke-width="2.2" points="{polyline(x_coords, y_max_coords)}"/>'
    )
    lines.append(
        f'<polyline fill="none" stroke="#f6c85f" stroke-width="2.0" points="{polyline(x_coords, y_mean_coords)}"/>'
    )
    lines.append(
        f'<polyline fill="none" stroke="#6f4e7c" stroke-width="2.0" points="{polyline(x_coords, y_median_coords)}"/>'
    )

    legend_top = 6

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2:.1f}" y="36" text-anchor="middle" font-size="28" font-family="Arial, sans-serif" fill="#222">{args.title} ({len(rows)} Episodes)</text>
<text x="{width/2:.1f}" y="{height-18}" text-anchor="middle" font-size="18" font-family="Arial, sans-serif" fill="#333">Episode</text>
<text x="24" y="{height/2:.1f}" transform="rotate(-90 24 {height/2:.1f})" text-anchor="middle" font-size="18" font-family="Arial, sans-serif" fill="#333">Reward</text>
{''.join(lines)}
<g font-family="Arial, sans-serif" font-size="14">
  <rect x="{width-320}" y="{legend_top}" width="280" height="58" fill="white" stroke="#ddd"/>
  <line x1="{width-300}" y1="{legend_top+18}" x2="{width-260}" y2="{legend_top+18}" stroke="#0b84a5" stroke-width="3"/>
  <text x="{width-245}" y="{legend_top+23}" fill="#222">Max reward</text>
  <line x1="{width-300}" y1="{legend_top+36}" x2="{width-260}" y2="{legend_top+36}" stroke="#f6c85f" stroke-width="3"/>
  <text x="{width-245}" y="{legend_top+41}" fill="#222">Mean reward</text>
  <line x1="{width-300}" y1="{legend_top+54}" x2="{width-260}" y2="{legend_top+54}" stroke="#6f4e7c" stroke-width="3"/>
  <text x="{width-245}" y="{legend_top+59}" fill="#222">Median reward</text>
</g>
</svg>
"""

    out_svg.write_text(svg, encoding="utf-8")
    print(f"Saved CSV: {out_csv}")
    print(f"Saved SVG: {out_svg}")


if __name__ == "__main__":
    main()
