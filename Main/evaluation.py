import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, List


@dataclass
class RecordResult:
    index: str
    correct: int
    total: int
    soft_correct: int = 0  # why: expose per-record softness for downstream reporting

    @property
    def acc(self) -> float:
        return (self.correct / self.total) if self.total else 0.0

    @property
    def soft_acc(self) -> float:
        return (self.soft_correct / self.total) if self.total else 0.0


def parse_pairs(text: str) -> Dict[str, int]:
    """Parse 'ラベル=3, ラベル（説明）=2' to {label: int}; strip parentheses notes."""
    if not text:
        return {}
    t = str(text).replace("，", ",").replace(" ", "")
    parts = [p for p in t.split(",") if p]
    pairs: Dict[str, int] = {}
    for p in parts:
        if "=" not in p:
            continue  # noisy fragment; skip
        label_raw, val_raw = p.rsplit("=", 1)
        # why: avoid mismatches from auxiliary notes
        label = re.sub("（.*?）", "", label_raw)
        label = re.sub("[(].*?[)]", "", label).strip("、, ")
        m = re.search("-?[0-9]+", val_raw)
        if not m:
            continue
        pairs[label] = int(m.group())
    return pairs


@dataclass
class EvalSummary:
    total_pairs: int
    correct_pairs: int
    overall_acc: float
    overall_mae: float
    overall_soft_acc: float
    per_record: List[RecordResult]
    per_label_acc: Dict[str, float]
    per_label_mae: Dict[str, float]
    per_label_soft_acc: Dict[str, float]
    per_label_support: Dict[str, int]
    confusion_matrices: Dict[str, Dict[int, Dict[int, int]]]
    soft_tolerance: int


def evaluate(
    records: List[dict],
    pred_key: str,
    gold_key: str = "output_original",
    soft_tolerance: int = 1,
) -> EvalSummary:
    """Compute micro-accuracy, micro-MAE, ACC_soft (|pred-gold|<=tol), per-label metrics and confusion matrices."""
    total_pairs = 0
    correct_pairs = 0
    soft_correct_pairs = 0
    abs_err_sum = 0

    per_record: List[RecordResult] = []

    per_label_total: DefaultDict[str, int] = defaultdict(int)
    per_label_correct: DefaultDict[str, int] = defaultdict(int)
    per_label_soft_correct: DefaultDict[str, int] = defaultdict(int)
    per_label_abs_sum: DefaultDict[str, int] = defaultdict(int)

    # cm[label][gold][pred] -> count
    cm: Dict[str, Dict[int, Dict[int, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )

    for i, rec in enumerate(records):
        if pred_key not in rec or gold_key not in rec:
            continue
        pred = parse_pairs(rec[pred_key])
        gold = parse_pairs(rec[gold_key])
        keys = set(pred) & set(gold)
        if not keys:
            continue
        c = 0
        sc = 0
        for k in keys:
            gv = gold[k]
            pv = pred[k]
            per_label_total[k] += 1
            if pv == gv:
                per_label_correct[k] += 1
                c += 1
            diff = abs(pv - gv)
            if diff <= soft_tolerance:
                per_label_soft_correct[k] += 1
                sc += 1
            per_label_abs_sum[k] += diff
            abs_err_sum += diff
            total_pairs += 1
            cm[k][gv][pv] += 1
        correct_pairs += c
        soft_correct_pairs += sc
        per_record.append(
            RecordResult(
                index=str(rec.get("index", i)),
                correct=c,
                total=len(keys),
                soft_correct=sc,
            )
        )

    overall_acc = (correct_pairs / total_pairs) if total_pairs else 0.0
    overall_soft_acc = (soft_correct_pairs / total_pairs) if total_pairs else 0.0
    overall_mae = (abs_err_sum / total_pairs) if total_pairs else 0.0

    per_label_acc = {
        k: (per_label_correct[k] / per_label_total[k]) for k in per_label_total
    }
    per_label_soft_acc = {
        k: (per_label_soft_correct[k] / per_label_total[k]) for k in per_label_total
    }
    per_label_mae = {
        k: (per_label_abs_sum[k] / per_label_total[k]) for k in per_label_total
    }
    per_label_support = dict(per_label_total)

    # stable ordering for display
    per_label_acc = dict(sorted(per_label_acc.items(), key=lambda x: x[0]))
    per_label_soft_acc = dict(sorted(per_label_soft_acc.items(), key=lambda x: x[0]))
    per_label_mae = dict(sorted(per_label_mae.items(), key=lambda x: x[0]))
    per_label_support = dict(sorted(per_label_support.items(), key=lambda x: x[0]))

    return EvalSummary(
        total_pairs=total_pairs,
        correct_pairs=correct_pairs,
        overall_acc=overall_acc,
        overall_mae=overall_mae,
        overall_soft_acc=overall_soft_acc,
        per_record=per_record,
        per_label_acc=per_label_acc,
        per_label_mae=per_label_mae,
        per_label_soft_acc=per_label_soft_acc,
        per_label_support=per_label_support,
        confusion_matrices=cm,
        soft_tolerance=soft_tolerance,
    )


def _fmt_float(x: float) -> str:
    return f"{x:.4f}".rstrip("0").rstrip(".")


def print_per_label_table(summary: EvalSummary) -> None:
    print("Per-label metrics (label, accuracy, soft_acc, MAE, support):")
    print(f"label\taccuracy\tsoft_acc(tol={summary.soft_tolerance})\tMAE\tsupport")
    for label in summary.per_label_acc.keys():
        acc = summary.per_label_acc[label]
        soft_acc = summary.per_label_soft_acc.get(label, 0.0)
        mae = summary.per_label_mae[label]
        sup = summary.per_label_support.get(label, 0)
        print(f"{label}\t{acc:.2%}\t{soft_acc:.2%}\t{_fmt_float(mae)}\t{sup}")


def print_confusion_matrices(summary: EvalSummary) -> None:
    print("Confusion matrices (per label): rows=gold, cols=pred, count")
    for label in sorted(summary.confusion_matrices.keys()):
        cm = summary.confusion_matrices[label]
        gold_vals = set(cm.keys())
        pred_vals = set(v for gv in cm.values() for v in gv.keys())
        values = sorted(gold_vals | pred_vals)
        if not values:
            continue
        print("[Label]", label)
        header = ["gold|pred"] + [str(v) for v in values]
        print("\t".join(header))
        for gv in values:
            row = [str(gv)]
            for pv in values:
                row.append(str(cm.get(gv, {}).get(pv, 0)))
            print("\t".join(row))


def main() -> None:
    path = Path("")  # 评价生成的结果
    data = json.loads(path.read_text(encoding="utf-8"))

    pred_key = "output_model"
    gold_key = "output_original"
    soft_tol = 1  # ACC_soft 容忍度

    summary = evaluate(data, pred_key, gold_key, soft_tol)

    print(f"Total pairs: {summary.total_pairs}")
    print(f"Correct pairs: {summary.correct_pairs}")
    print(f"Overall accuracy: {summary.overall_acc:.4%}")
    print(
        f"Overall soft accuracy (tol={summary.soft_tolerance}): {summary.overall_soft_acc:.4%}"
    )
    print(f"Overall MAE: {_fmt_float(summary.overall_mae)}")

    print_per_label_table(summary)
    print_confusion_matrices(summary)


if __name__ == "__main__":
    main()
