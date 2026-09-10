#!/usr/bin/env python3
"""Export the paired V2 replays into the paper 2x2 result tables."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from protocol_v2 import DEV_DATASETS, EVAL_DATASETS, PROTOCOL_VERSION


ARMS = ("base", "posterior", "responsibility", "full")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def _macro(target: np.ndarray, pred: np.ndarray) -> float:
    labels = np.unique(target)
    return 100.0 * float(np.mean([np.mean(pred[target == label] == label) for label in labels])) if len(labels) else 0.0


def _trace_metric(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        target = z["target"].reshape(-1).astype(np.int64); pred = z["prediction"].reshape(-1).astype(np.int64)
        correct = pred == target
        return {"correct": int(correct.sum()), "num_samples": len(target), "accuracy": 100.0 * float(correct.mean()),
                "macro_accuracy": _macro(target, pred), "late50_accuracy": 100.0 * float(correct[len(correct)//2:].mean()),
                "prediction": pred, "target": target}


def _pair_dir_metrics(dataset_dir: Path) -> dict[str, Any]:
    payload = json.loads((dataset_dir / "pairs.json").read_text(encoding="utf-8"))
    out: dict[str, Any] = {}
    for pair, names in (("base_posterior", ("base", "posterior")), ("responsibility_full", ("responsibility", "full"))):
        block = payload[pair]
        for name in names:
            m = dict(block[name])
            trace = Path(m["trace_path"])
            if not trace.is_absolute(): trace = Path.cwd() / trace
            tm = _trace_metric(trace)
            m.update({k: tm[k] for k in ("macro_accuracy", "late50_accuracy")})
            m["_trace"] = trace; m["_target"] = tm["target"]; m["_prediction"] = tm["prediction"]
            out[name] = m
    return out


def _aggregate(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    if not rows: return {"correct": 0, "num_samples": 0, "micro_accuracy": None, "macro_accuracy": None, "late50_macro": None}
    return {"correct": sum(int(r[arm]["correct"]) for r in rows), "num_samples": sum(int(r[arm]["num_samples"]) for r in rows),
            "micro_accuracy": 100.0 * sum(int(r[arm]["correct"]) for r in rows) / sum(int(r[arm]["num_samples"]) for r in rows),
            "macro_accuracy": float(np.mean([float(r[arm]["macro_accuracy"]) for r in rows])),
            "late50_macro": float(np.mean([float(r[arm]["late50_accuracy"]) for r in rows]))}


def export(pairs_root: Path, output: Path) -> dict[str, Any]:
    rows_by_dataset: list[dict[str, Any]] = []
    correction_rows, responsibility_rows, temporal_rows = [], [], []
    verification = []
    for dataset_dir in sorted(p for p in pairs_root.iterdir() if p.is_dir() and (p / "pairs.json").exists()):
        dataset = dataset_dir.name; metrics = _pair_dir_metrics(dataset_dir)
        row: dict[str, Any] = {"dataset": dataset, "split": "development" if dataset in DEV_DATASETS else "evaluation" if dataset in EVAL_DATASETS else "unknown", "num_samples": metrics["base"]["num_samples"]}
        for arm in ARMS:
            row[arm] = metrics[arm]["accuracy"]; row[arm + "_correct"] = metrics[arm]["correct"]
        row.update({"delta_posterior": row["posterior"] - row["base"], "delta_responsibility": row["responsibility"] - row["base"], "delta_full": row["full"] - row["base"], "interaction": row["full"] - row["posterior"] - row["responsibility"] + row["base"]})
        rows_by_dataset.append({"dataset": dataset, **{arm: metrics[arm] for arm in ARMS}, "row": row})
        p, u, f, b = metrics["posterior"], metrics["responsibility"], metrics["full"], metrics["base"]
        correction_rows.extend([
            {"dataset": dataset, "comparison": "posterior_vs_base", "wrong_to_correct": p["wrong_to_correct"], "correct_to_wrong": p["correct_to_wrong"], "net": p["wrong_to_correct"] - p["correct_to_wrong"], "changed_prediction_rate": p["changed_prediction_rate"]},
            {"dataset": dataset, "comparison": "full_vs_responsibility", "wrong_to_correct": f["wrong_to_correct"], "correct_to_wrong": f["correct_to_wrong"], "net": f["wrong_to_correct"] - f["correct_to_wrong"], "changed_prediction_rate": f["changed_prediction_rate"]},
        ])
        responsibility_rows.append({"dataset": dataset, "base_wrm": b["wrm"], "u_wrm": u["wrm"], "delta_wrm": u["wrm"] - b["wrm"], "base_crm": b["crm"], "u_crm": u["crm"], "delta_crm": u["crm"] - b["crm"], "base_total_mass": b["total_update_mass"], "u_total_mass": u["total_update_mass"], "correct_allocation_gate": u.get("correct_allocation_gate"), "wrong_allocation_gate": u.get("wrong_allocation_gate")})
        for arm in ARMS:
            target, pred = metrics[arm]["_target"], metrics[arm]["_prediction"]
            for index, window in enumerate(np.array_split(np.arange(len(target)), 10), 1):
                if len(window): temporal_rows.append({"dataset": dataset, "arm": arm, "window": index, "start": int(window[0]), "end_exclusive": int(window[-1] + 1), "accuracy": 100.0 * float(np.mean(pred[window] == target[window]))})
        verification.append({"dataset": dataset, "pair_a_state_equal": b["state_sha256"] == p["state_sha256"], "pair_a_trajectory_equal": b["trajectory_sha256"] == p["trajectory_sha256"], "pair_a_compatibility_equal": b["compatibility_sha256"] == p["compatibility_sha256"], "pair_b_state_equal": u["state_sha256"] == f["state_sha256"], "pair_b_trajectory_equal": u["trajectory_sha256"] == f["trajectory_sha256"], "pair_b_compatibility_equal": u["compatibility_sha256"] == f["compatibility_sha256"], "health": all(metrics[a].get("health_status") == "healthy" for a in ARMS)})
    groups = {}
    all_rows = rows_by_dataset
    for name, datasets in (("all", all_rows), ("development", [r for r in all_rows if r["dataset"] in DEV_DATASETS]), ("evaluation", [r for r in all_rows if r["dataset"] in EVAL_DATASETS])):
        groups[name] = {arm: _aggregate(datasets, arm) for arm in ARMS}
    contrasts = {}
    for group, arm_rows in groups.items():
        contrasts[group] = {}
        for metric in ("correct", "micro_accuracy", "macro_accuracy", "late50_macro"):
            b, p, u, f = [arm_rows[a][metric] for a in ARMS]
            if b is None or p is None or u is None or f is None:
                contrasts[group][metric] = {"delta_posterior": None, "delta_responsibility": None, "delta_full": None, "interaction": None}
            else:
                contrasts[group][metric] = {"delta_posterior": p - b, "delta_responsibility": u - b, "delta_full": f - b, "interaction": f - p - u + b}
    per = [r["row"] for r in rows_by_dataset]
    summary = {"status": "complete", "protocol_version": PROTOCOL_VERSION, "groups": groups, "contrasts": contrasts, "per_dataset": per}
    _write_json(output / "paper_2x2_summary.json", summary); _write_csv(output / "per_dataset.csv", per); _write_csv(output / "paper_2x2_summary.csv", [{"group": g, "arm": a, **v} for g, v0 in groups.items() for a, v in v0.items()]); _write_csv(output / "posterior_correction.csv", correction_rows); _write_csv(output / "responsibility_analysis.csv", responsibility_rows); _write_csv(output / "temporal_analysis.csv", temporal_rows); _write_json(output / "verification.json", {"status": "passed" if all(x["health"] and x["pair_a_state_equal"] and x["pair_a_trajectory_equal"] and x["pair_b_state_equal"] and x["pair_b_trajectory_equal"] for x in verification) else "failed", "protocol_version": PROTOCOL_VERSION, "datasets": verification})
    all_group = groups.get("all", {})
    report = [
        "# OCR-DOTA PaperCore V2 ten-dataset report", "",
        f"Protocol: {PROTOCOL_VERSION}",
        "The Base arm is exact Original DOTA (LegacyState, FP32, zero-shot CLIP responsibility).",
        "Posterior and responsibility parameters were selected only on the preregistered Dev-5 split.",
        "All ten streams were historically exposed; Evaluation-5 is therefore held-out in this campaign, not pristine external test.", "",
        "## All-10 paired result",
    ]
    for arm in ARMS:
        value = all_group.get(arm, {})
        report.append(f"- {arm}: {value.get('correct')}/{value.get('num_samples')} ({value.get('micro_accuracy')}% micro; {value.get('macro_accuracy')}% macro)")
    report += ["", "## Interpretation", "- Posterior uses direct/entropy compatibility only and has no effect on Base state trajectory.", "- Responsibility uses one shared compatibility relation and clean-mass gate; U and Full share one updated state trajectory.", "- See posterior_correction.csv, responsibility_analysis.csv and temporal_analysis.csv for mechanism evidence.", ""]
    (output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--pairs-root", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args(); print(json.dumps(export(args.pairs_root.resolve(), args.output.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__": main()
