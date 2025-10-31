from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt


def parse_log(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    blocks = re.split(r"(?m)^\[Learner\] Update\s*$", text)
    if blocks and blocks[0].strip() == "":
        blocks = blocks[1:]

    rows: List[Dict] = []
    for i, blk in enumerate(blocks, start=1):
        row: Dict = {"update": i}

        # Rewards
        m = re.search(r"avg/step=\s*([\-0-9.eE]+)", blk)
        if m:
            row["avg_step"] = float(m.group(1))
        m = re.search(r"avg/traj=\s*([\-0-9.eE]+)", blk)
        if m:
            row["avg_traj"] = float(m.group(1))
        for q in ("p10", "p50", "p90"):
            m = re.search(rf"{q}=\s*([\-0-9.eE]+)", blk)
            if m:
                row[q] = float(m.group(1))

        # Entropy
        m = re.search(r"Entropy\s*:\s*H=\s*([\-0-9.eE]+)", blk)
        if m:
            row["entropy"] = float(m.group(1))

        # Values
        m = re.search(r"Values\s*:\s*mean=\s*([\-0-9.eE]+)\s*std=\s*([\-0-9.eE]+)", blk)
        if m:
            row["value_mean"] = float(m.group(1))
            row["value_std"] = float(m.group(2))

        # Actions
        actions = re.findall(r"([A-Za-z0-9_+-]+)=\s*([0-9.]+)%", blk)
        for name, val in actions:
            if name not in ("p10", "p50", "p90"):
                col = f"action_{name}"
                row[col] = float(val)

        # Gradients, lr, time, steps/s
        m = re.search(r"\|\|g\|\|_2=\s*([\-0-9.eE]+)", blk)
        if m:
            row["grad_norm"] = float(m.group(1))
        m = re.search(r"lr=\s*([\-0-9.eE]+)", blk)
        if m:
            row["lr"] = float(m.group(1))
        m = re.search(r"time=\s*([\-0-9.eE]+)s", blk)
        if m:
            row["time_s"] = float(m.group(1))
        m = re.search(r"steps/s=\s*([\-0-9.eE]+)", blk)
        if m:
            row["steps_per_s"] = float(m.group(1))

        # PPO diag: KL, value_loss, explained_var
        m = re.search(r"KL=\s*([\-0-9.eE]+)", blk)
        if m:
            row["KL"] = float(m.group(1))
        m = re.search(r"value_loss=\s*([\-0-9.eE]+)", blk)
        if m:
            row["value_loss"] = float(m.group(1))
        m = re.search(r"explained_var=\s*([\-0-9.eE]+)", blk)
        if m:
            row["explained_var"] = float(m.group(1))

        # Batch temp or other batch params
        m = re.search(r"temp=\s*([\-0-9.eE]+)", blk)
        if m:
            row["temp"] = float(m.group(1))

        if len(row) > 1:
            rows.append(row)

    if not rows:
        raise RuntimeError(f"No learner update blocks parsed from {path}")

    df = pd.DataFrame(rows).set_index("update")
    df = df.sort_index()
    return df


def plot_df(df: pd.DataFrame, out_prefix: str = "learner_plot", interactive: bool = False) -> str:
    nplots = 6
    fig, axes = plt.subplots(nplots, 1, figsize=(10, 3 * nplots), tight_layout=True)

    # 1) Reward (avg_traj)
    if "avg_traj" in df.columns:
        axes[0].plot(df.index, df["avg_traj"], label="avg_traj")
        axes[0].set_ylabel("avg_traj")
        axes[0].legend()

    # 2) Entropy
    if "entropy" in df.columns:
        axes[1].plot(df.index, df["entropy"], color="tab:orange", label="entropy")
        axes[1].set_ylabel("entropy")
        axes[1].legend()

    # 3) Learning rate and grad norm (two y-axes)
    ax = axes[2]
    if "lr" in df.columns:
        ax.plot(df.index, df["lr"], color="tab:green", label="lr")
    if "grad_norm" in df.columns:
        ax2 = ax.twinx()
        ax2.plot(df.index, df["grad_norm"], color="tab:red", label="grad_norm")
        ax2.set_ylabel("grad_norm")
    ax.set_ylabel("lr")
    ax.set_title("Learning rate and grad norm")

    # 4) PPO diagnostics: KL, value_loss, explained_var
    ax4 = axes[3]
    plotted = False
    if "KL" in df.columns:
        ax4.plot(df.index, df["KL"], label="KL")
        plotted = True
    if "value_loss" in df.columns:
        ax4.plot(df.index, df["value_loss"], label="value_loss")
        plotted = True
    if "explained_var" in df.columns:
        ax4.plot(df.index, df["explained_var"], label="explained_var")
        plotted = True
    if plotted:
        ax4.legend()

    # 5) Steps/s
    if "steps_per_s" in df.columns:
        axes[4].plot(df.index, df["steps_per_s"], label="steps/s")
        axes[4].set_ylabel("steps/s")
        axes[4].legend()

    # 6) Actions: plot top action columns if present
    action_cols = [c for c in df.columns if c.startswith("action_")]
    if action_cols:
        axes[5].stackplot(df.index, [df[c].fillna(0) for c in action_cols], labels=action_cols)
        axes[5].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
        axes[5].set_ylabel("action %")

    out_png = f"{out_prefix}.png"
    fig.suptitle("Learner metrics over updates", fontsize=16)
    fig.savefig(out_png, bbox_inches="tight", dpi=150)

    if interactive:
        try:
            import plotly.express as px
            import plotly.io as pio

            html_path = f"{out_prefix}.html"
            num = df.select_dtypes(include=["number"]).reset_index()
            m = num.melt(id_vars=["update"], var_name="metric", value_name="value")
            figp = px.line(m, x="update", y="value", color="metric", title="Learner metrics")
            pio.write_html(figp, file=html_path, auto_open=False)
        except Exception:
            html_path = ""
    else:
        html_path = ""

    return out_png if not html_path else html_path


def main() -> None:
    p = argparse.ArgumentParser(description="Plot learner.log metrics")
    p.add_argument("log", help="Path to learner.log")
    p.add_argument("--out", default="learner_plot", help="Output file prefix (no extension)")
    p.add_argument("--interactive", action="store_true", help="Also write interactive HTML (plotly)")
    args = p.parse_args()

    df = parse_log(args.log)
    out = plot_df(df, out_prefix=args.out, interactive=args.interactive)
    print(f"Wrote plot to: {out}")


if __name__ == "__main__":
    main()
