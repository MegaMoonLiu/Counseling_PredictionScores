import json
import numpy as np
import pandas as pd

# JSONファイルの読み込み
with open("", "r", encoding="utf-8") as f:
    data = json.load(f)

# データの抽出とフラット化
records = []
for session in data:
    for i in range(1, 21):
        key = f"evaluation_items_{i}"
        if key in session:
            item = session[key]
            records.append(
                {
                    "label": item["label"],
                    "output_score": item["output_score"],
                    "reference": item["reference"],
                }
            )

df = pd.DataFrame(records)

df["output_score"] = pd.to_numeric(df["output_score"], errors="coerce")
df["reference"] = pd.to_numeric(df["reference"], errors="coerce")
df = df.dropna(subset=["label", "output_score", "reference"]).reset_index(drop=True)


def calculate_metrics(group: pd.DataFrame) -> pd.Series:
    total = int(len(group))
    if total == 0:
        return pd.Series(
            {
                "Accuracy": np.nan,
                "Accuracy Var": np.nan,
                "Soft Accuracy": np.nan,
                "Soft Accuracy Var": np.nan,
                "MAE": np.nan,
                "MAE Var": np.nan,
                "Count": 0,
            }
        )

    correct = (group["output_score"] == group["reference"]).astype(int)
    soft_correct = (np.abs(group["output_score"] - group["reference"]) <= 1).astype(int)
    abs_err = np.abs(group["output_score"] - group["reference"])

    return pd.Series(
        {
            "Accuracy": float(correct.mean()),
            "Soft Accuracy": float(soft_correct.mean()),
            "MAE": float(abs_err.mean()),
            "Count": total,
        }
    )


# ラベルごとの集計
label_metrics = df.groupby("label", sort=False).apply(calculate_metrics).reset_index()

# 全体の集計
overall_stats = calculate_metrics(df)
overall_df = pd.DataFrame(
    [
        {
            "label": "Overall (Total)",
            "Accuracy": overall_stats["Accuracy"],
            "Soft Accuracy": overall_stats["Soft Accuracy"],
            "MAE": overall_stats["MAE"],
            "Count": int(overall_stats["Count"] / 20) if overall_stats["Count"] else 0,
        }
    ]
)

final_report = pd.concat([label_metrics, overall_df], ignore_index=True)

final_report["Count"] = final_report["Count"].round().astype(int)

# 列順を整える
final_report = final_report[
    [
        "label",
        "Count",
        "Accuracy",
        "Soft Accuracy",
        "MAE",
    ]
]

pd.options.display.float_format = "{:.3f}".format
print(final_report.to_string(index=False))
