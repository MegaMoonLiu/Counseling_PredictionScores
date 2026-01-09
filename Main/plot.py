import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
import pandas as pd


METRICS = ["ACC (↑)", "ACCsoft (↑)", "MAE (↓)"]


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def load_and_clean(excel_path: Path, sheet_name: str | int = 0) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    required = {"Dim.", "Model", *METRICS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["Dim."] = df["Dim."].ffill()

    for m in METRICS:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    df = df.dropna(subset=["Dim.", "Model"])
    return df


def infer_dim_order(df: pd.DataFrame) -> list[str]:
    dims = set(df["Dim."].astype(str).unique().tolist())
    order = [f"D{i}" for i in range(1, 21) if f"D{i}" in dims]
    return order if order else df["Dim."].astype(str).drop_duplicates().tolist()


def plot_grouped_bars(pivoted: pd.DataFrame, metric: str, out_path: Path) -> None:
    pivoted = pivoted.copy()

    preferred = [c for c in ["CoT", "Paper"] if c in pivoted.columns]
    rest = [c for c in pivoted.columns if c not in preferred]
    pivoted = pivoted[preferred + rest]

    x = np.arange(len(pivoted.index))
    n = max(len(pivoted.columns), 1)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, col in enumerate(pivoted.columns):
        ax.bar(
            x + (i - (n - 1) / 2) * width,
            pivoted[col].to_numpy(),
            width=width,
            label=str(col),
        )

    ax.set_title(f"CoT vs Paper — {metric}")
    ax.set_xlabel("Dimension")
    ax.set_ylabel(metric)
    ax.set_xticks(x)
    ax.set_xticklabels(pivoted.index.tolist(), rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    # 绘制图表
    excel_path = Path("./gen_results/evaluation.xlsx")
    out_dir = Path("")

    df = load_and_clean(excel_path)
    dim_order = infer_dim_order(df)

    stamp = int(time.time())  # 防止重名覆盖
    for metric in METRICS:
        pv = df.pivot_table(
            index="Dim.", columns="Model", values=metric, aggfunc="mean"
        )
        pv = pv.reindex(dim_order)

        out_file = out_dir / f"CoT_vs_Paper_{_safe_name(metric)}_{stamp}.png"
        plot_grouped_bars(pv, metric, out_file)

    print("Done. Generated:")
    for metric in METRICS:
        print(f"- CoT_vs_Paper_{_safe_name(metric)}_{stamp}.png")

    # 计算最大差值
    pv = df.pivot_table(index="Dim.", columns="Model", values=METRICS, aggfunc="mean")

    for m in METRICS:
        if (m, "CoT") not in pv.columns or (m, "Paper") not in pv.columns:
            raise ValueError(f"Pivot missing CoT/Paper for metric: {m}")

    diff = pd.DataFrame(index=pv.index)
    abs_diff = pd.DataFrame(index=pv.index)

    for m in METRICS:
        diff[m] = pv[(m, "CoT")] - pv[(m, "Paper")]
        abs_diff[m] = diff[m].abs()

    print("=== Max gap dimension per metric (by absolute difference) ===")
    for m in METRICS:
        max_abs = abs_diff[m].max()
        winners = abs_diff.index[abs_diff[m] == max_abs].tolist()
        for d in winners:
            print(
                f"{m}: {d} | |Δ|={max_abs:.6g} | Δ(CoT-Paper)={diff.loc[d, m]:.6g} "
                f"| CoT={pv.loc[d, (m, 'CoT')]:.6g} | Paper={pv.loc[d, (m, 'Paper')]:.6g}"
            )

    combined = abs_diff.sum(axis=1)
    combined_max = combined.max()
    combined_winners = combined.index[combined == combined_max].tolist()

    print("\n=== Max gap dimension overall (sum of abs gaps across 3 metrics) ===")
    for d in combined_winners:
        print(
            f"{d}: sum(|Δ|)={combined_max:.6g} | per-metric |Δ|={abs_diff.loc[d].to_dict()}"
        )

    # Optional: export detailed table
    detail = pd.concat(
        {
            "CoT": pv.xs("CoT", axis=1, level=1),
            "Paper": pv.xs("Paper", axis=1, level=1),
            "Diff(CoT-Paper)": diff,
            "AbsDiff": abs_diff,
            "SumAbsDiff": combined,
        },
        axis=1,
    )
    detail.to_csv(f"{out_dir}/fmetric_gaps_by_dim.csv", encoding="utf-8-sig")
    print("\nSaved: metric_gaps_by_dim.csv")


if __name__ == "__main__":
    main()
