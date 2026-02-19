import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

JSON_PATH = "./gen_results/predictions-s2.json"
OUT_PUT = "./plot/"


def load_records(json_path: Path) -> list[dict]:
    return json.loads(json_path.read_text(encoding="utf-8"))


def extract_scores(records: list[dict]) -> tuple[list[int], list[int]]:
    output_scores: list[int] = []
    reference_scores: list[int] = []

    for rec in records:
        for k, v in rec.items():
            if not (
                isinstance(k, str)
                and k.startswith("evaluation_items_")
                and isinstance(v, dict)
            ):
                continue
            if "output_score" in v:
                output_scores.append(int(v["output_score"]))
            if "reference" in v:
                reference_scores.append(int(v["reference"]))

    return output_scores, reference_scores


def build_count_table(
    output_scores: list[int], reference_scores: list[int]
) -> pd.DataFrame:
    out_counter = Counter(output_scores)
    ref_counter = Counter(reference_scores)

    all_scores = sorted(set(out_counter) | set(ref_counter))
    df = pd.DataFrame(
        {
            "score": all_scores,
            "output_score_count": [out_counter.get(s, 0) for s in all_scores],
            "reference_count": [ref_counter.get(s, 0) for s in all_scores],
        }
    )
    df["diff(output-reference)"] = df["output_score_count"] - df["reference_count"]
    return df


def plot_side_by_side_hist(df: pd.DataFrame, out_path: Path) -> None:
    scores = df["score"].tolist()
    x = list(range(len(scores)))
    width = 0.4

    plt.figure(figsize=(9, 4.8))
    plt.bar(
        [i - width / 2 for i in x],
        df["output_score_count"].tolist(),
        width=width,
        label="output_score",
    )
    plt.bar(
        [i + width / 2 for i in x],
        df["reference_count"].tolist(),
        width=width,
        label="reference",
    )
    plt.xticks(x, [str(s) for s in scores])
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title("output_score vs reference — Count by score")
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_path, dpi=200)


def main() -> None:
    json_path = Path(JSON_PATH)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path.resolve()}")

    records = load_records(json_path)
    output_scores, reference_scores = extract_scores(records)
    df = build_count_table(output_scores, reference_scores)

    print(df.to_string(index=False))

    plot_side_by_side_hist(df, Path(f"{OUT_PUT}score_hist_compare.png"))
    print("\nSaved chart -> score_hist_compare.png")


if __name__ == "__main__":
    main()
