"""Write results tables and a coefficient plot for one study."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DML_COLOR = "#2a78d6"
BENCHMARK_COLOR = "#52514e"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def _stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


def to_markdown(results: pd.DataFrame) -> str:
    lines = [
        "| Method | Coef. | Std. err. | p-value | 95% CI |",
        "|---|---:|---:|---:|---|",
    ]
    for r in results.itertuples():
        lines.append(
            f"| {r.method} | {r.coef:.4f}{_stars(r.p_value)} | {r.se:.4f} | {r.p_value:.3f} "
            f"| [{r.ci_low:.4f}, {r.ci_high:.4f}] |"
        )
    return "\n".join(lines)


def coef_plot(results: pd.DataFrame, title: str, path: Path, true_effect: float | None = None) -> None:
    df = results.iloc[::-1].reset_index(drop=True)
    is_bench = df["method"].str.startswith("TWFE")
    colors = [BENCHMARK_COLOR if b else DML_COLOR for b in is_bench]

    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(df) + 1.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.axvline(0, color=INK_MUTED, linewidth=1, zorder=1)
    if true_effect is not None:
        ax.axvline(true_effect, color=INK_MUTED, linewidth=1, linestyle="--", zorder=1)
        ax.annotate(
            f"true effect = {true_effect:g}",
            xy=(true_effect, len(df) - 0.5),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=8,
            color=INK_MUTED,
            va="center",
        )

    for i, r in df.iterrows():
        ax.plot([r.ci_low, r.ci_high], [i, i], color=colors[i], linewidth=2, solid_capstyle="round", zorder=2)
        ax.scatter(r.coef, i, s=64, color=colors[i], edgecolor=SURFACE, linewidth=2, zorder=3)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["method"].str.replace("DML - ", "", regex=False), color=INK, fontsize=9)
    ax.set_ylim(-0.6, len(df) - 0.4)
    ax.set_xlabel("Estimated effect of DID (95% CI, firm-clustered)", color=INK_MUTED, fontsize=9)
    fig.suptitle(title, x=0.02, ha="left", color=INK, fontsize=11)
    ax.tick_params(axis="x", colors=INK_MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def save_results(
    results: pd.DataFrame,
    info: dict,
    name: str,
    title: str,
    out_dir: Path,
    true_effect: float | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8")
    coef_plot(results, title, out_dir / f"{name}.png", true_effect)

    sample = ", ".join(f"{k}: {v}" for k, v in info.items())
    md = f"### {title}\n\nSample - {sample}\n\n{to_markdown(results)}\n\n*** p<0.01, ** p<0.05, * p<0.1\n"
    (out_dir / f"{name}.md").write_text(md, encoding="utf-8")
