"""Bounded-memory agreement analysis for the four official ANYA CSV outputs.

The final ``Predicted_Label`` field is treated as a model output, never as
verified ground truth.  This script only measures agreement/disagreement and
does not train a classifier or write into ``datasets/raw``.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from posture_ai.data.anya import iter_anya_csv  # noqa: E402


RAW_DIR = ROOT / "datasets" / "raw" / "anya" / "official-download"
METRICS_PATH = ROOT / "outputs" / "metrics" / "anya_pseudolabel_consensus.json"
REPORT_PATH = ROOT / "reports" / "anya_pseudolabel_consensus.md"
MODEL_FILES = {
    "ANN": RAW_DIR / "ANN.csv",
    "Random_Forest": RAW_DIR / "Random_Forest.csv",
    "lgbm": RAW_DIR / "lgbm.csv",
    "xgb": RAW_DIR / "xgb.csv",
}
USECOLS = ["filename", "clusterid", "normclusterid", "Predicted_Label"]
PAIR_NAMES = [(left, right) for left, right in combinations(MODEL_FILES, 2)]


def _kappa_from_confusion(matrix: np.ndarray) -> float | None:
    total = int(matrix.sum())
    if total == 0:
        return None
    observed = float(np.trace(matrix) / total)
    expected = float(matrix.sum(axis=1).dot(matrix.sum(axis=0)) / (total * total))
    if np.isclose(1.0 - expected, 0.0):
        return 1.0 if np.isclose(observed, 1.0) else 0.0
    return (observed - expected) / (1.0 - expected)


def _pairwise_summary(left: np.ndarray, right: np.ndarray) -> dict:
    left = left.astype(np.int8, copy=False)
    right = right.astype(np.int8, copy=False)
    matrix = np.zeros((2, 2), dtype=np.int64)
    np.add.at(matrix, (left, right), 1)
    disagreements = int(np.count_nonzero(left != right))
    total = int(left.size)
    return {
        "rows": total,
        "agreements": total - disagreements,
        "disagreements": disagreements,
        "agreement_rate": (total - disagreements) / total if total else None,
        "disagreement_rate": disagreements / total if total else None,
        "cohen_kappa": _kappa_from_confusion(matrix),
        "confusion_matrix_rows_left_0_1_cols_right_0_1": matrix.tolist(),
    }


def _row_consensus_summary(predictions: np.ndarray) -> dict:
    ones = predictions.sum(axis=0)
    total = int(predictions.shape[1])
    unanimous_mask = (ones == 0) | (ones == 4)
    strict_three_mask = (ones == 1) | (ones == 3)
    tie_mask = ones == 2
    majority_or_better = unanimous_mask | strict_three_mask
    return {
        "rows": total,
        "unanimous_rows": int(unanimous_mask.sum()),
        "unanimous_rate": float(unanimous_mask.mean()) if total else None,
        "disagreement_rows_not_unanimous": int((~unanimous_mask).sum()),
        "disagreement_rate_not_unanimous": float((~unanimous_mask).mean()) if total else None,
        "strict_3_of_4_rows": int(strict_three_mask.sum()),
        "strict_3_of_4_rate": float(strict_three_mask.mean()) if total else None,
        "two_vs_two_tie_rows": int(tie_mask.sum()),
        "two_vs_two_tie_rate": float(tie_mask.mean()) if total else None,
        "three_of_four_or_unanimous_rows": int(majority_or_better.sum()),
        "three_of_four_or_unanimous_rate": float(majority_or_better.mean()) if total else None,
        "unanimous_label_counts": {
            "0": int((ones[unanimous_mask] == 0).sum()),
            "1": int((ones[unanimous_mask] == 4).sum()),
        },
        "majority_label_counts_excluding_2_vs_2_ties": {
            "0": int((ones <= 1).sum()),
            "1": int((ones >= 3).sum()),
        },
    }


def _group_summary(keys: np.ndarray, predictions: np.ndarray) -> dict:
    output: dict[str, dict] = {}
    ones = predictions.sum(axis=0)
    unanimous = (ones == 0) | (ones == 4)
    strict_three = (ones == 1) | (ones == 3)
    ties = ones == 2
    for key in np.unique(keys):
        key_text = str(key)
        mask = keys == key
        entry = {
            "rows": int(mask.sum()),
            "unanimous_rows": int(unanimous[mask].sum()),
            "unanimous_rate": float(unanimous[mask].mean()),
            "disagreement_rows_not_unanimous": int((~unanimous[mask]).sum()),
            "strict_3_of_4_rows": int(strict_three[mask].sum()),
            "two_vs_two_tie_rows": int(ties[mask].sum()),
            "majority_label_counts_excluding_2_vs_2_ties": {
                "0": int(((ones <= 1) & mask).sum()),
                "1": int(((ones >= 3) & mask).sum()),
            },
            "model_label_counts": {
                str(model): {
                    "0": int(((predictions[index] == 0) & mask).sum()),
                    "1": int(((predictions[index] == 1) & mask).sum()),
                }
                for index, model in enumerate(MODEL_FILES)
            },
            "pairwise_disagreements": {
                f"{left}_vs_{right}": int(((predictions[i] != predictions[j]) & mask).sum())
                for (left, right), (i, j) in zip(PAIR_NAMES, combinations(range(4), 2))
            },
        }
        output[key_text] = entry
    return output


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.3f}%"


def _report(data: dict) -> str:
    pair_rows = []
    for name, values in data["pairwise"].items():
        pair_rows.append(
            f"| {name} | {values['agreements']:,} | {values['disagreements']:,} | "
            f"{_pct(values['agreement_rate'])} | {values['cohen_kappa']:.6f} |"
        )
    consensus = data["row_consensus"]
    lines = [
        "# ANYA pseudo-label consensus analysis",
        "",
        "> This is a diagnostic of four model-output columns. It is **not** a ground-truth validation and no classifier was trained.",
        "",
        "## Source and method",
        "",
        f"- Official local release: `datasets/raw/anya/official-download/`.",
        f"- Files scanned in lockstep: `{', '.join(data['files'])}`.",
        f"- Rows: `{data['rows']:,}`; columns used: `filename`, `clusterid`, `normclusterid`, `Predicted_Label`.",
        "- Bounded-memory pandas chunk reading; the source CSVs were not modified.",
        "- A row is **unanimous** when all four predictions are equal; **strict 3/4** means exactly three models agree and one differs; **2/2 tie** means two models predict each class.",
        "",
        "## Pairwise agreement",
        "",
        "| Pair | Agreements | Disagreements | Agreement rate | Cohen's kappa |",
        "|---|---:|---:|---:|---:|",
        *pair_rows,
        "",
        "## Four-model consensus",
        "",
        f"- Unanimous: **{consensus['unanimous_rows']:,} / {consensus['rows']:,} ({_pct(consensus['unanimous_rate'])})**.",
        f"- Not unanimous (disagreement): **{consensus['disagreement_rows_not_unanimous']:,} ({_pct(consensus['disagreement_rate_not_unanimous'])})**.",
        f"- Exact 3-of-4 split: **{consensus['strict_3_of_4_rows']:,} ({_pct(consensus['strict_3_of_4_rate'])})**.",
        f"- 2-vs-2 ties: **{consensus['two_vs_two_tie_rows']:,} ({_pct(consensus['two_vs_two_tie_rate'])})**.",
        f"- 3-of-4-or-unanimous (binary majority excluding ties): **{consensus['three_of_four_or_unanimous_rows']:,} ({_pct(consensus['three_of_four_or_unanimous_rate'])})**.",
        f"- Unanimous label counts: `0={consensus['unanimous_label_counts']['0']:,}`, `1={consensus['unanimous_label_counts']['1']:,}`.",
        "",
        "## Group breakdown",
        "",
        "The following tables are diagnostic only. `filename`, `clusterid`, and `normclusterid` are not asserted to be participant IDs or expert labels.",
        "",
        "### By filename",
        "",
        "| Filename | Rows | Unanimous | Unanimous rate | Exact 3/4 | 2/2 tie | Majority 0 | Majority 1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, entry in data["by_filename"].items():
        lines.append(
            f"| `{key}` | {entry['rows']:,} | {entry['unanimous_rows']:,} | {_pct(entry['unanimous_rate'])} | "
            f"{entry['strict_3_of_4_rows']:,} | {entry['two_vs_two_tie_rows']:,} | "
            f"{entry['majority_label_counts_excluding_2_vs_2_ties']['0']:,} | "
            f"{entry['majority_label_counts_excluding_2_vs_2_ties']['1']:,} |"
        )
    for title, field in (("clusterid", "by_clusterid"), ("normclusterid", "by_normclusterid")):
        lines.extend([
            "",
            f"### By `{title}`",
            "",
            f"| {title} | Rows | Unanimous | Unanimous rate | Exact 3/4 | 2/2 tie | Majority 0 | Majority 1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for key, entry in data[field].items():
            lines.append(
                f"| `{key}` | {entry['rows']:,} | {entry['unanimous_rows']:,} | {_pct(entry['unanimous_rate'])} | "
                f"{entry['strict_3_of_4_rows']:,} | {entry['two_vs_two_tie_rows']:,} | "
                f"{entry['majority_label_counts_excluding_2_vs_2_ties']['0']:,} | "
                f"{entry['majority_label_counts_excluding_2_vs_2_ties']['1']:,} |"
            )
    lines.extend([
        "",
        "## Engineering decision",
        "",
        "- `Predicted_Label` remains a model output/pseudo-label field, not verified expert ground truth.",
        "- Agreement, including unanimous agreement, cannot establish correctness without the expert-labeled subset or an official cluster-to-TRUE/FALSE mapping.",
        "- These results may support pseudo-label provenance analysis and a future model-output comparison, but not a production ANYA ergonomic classifier in this investigation.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    missing = [str(path) for path in MODEL_FILES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing official ANYA files: " + ", ".join(missing))

    iterators = [iter_anya_csv(path, chunksize=20_000, usecols=USECOLS) for path in MODEL_FILES.values()]
    pair_confusions = {name: np.zeros((2, 2), dtype=np.int64) for name in PAIR_NAMES}
    total_predictions: list[np.ndarray] = []
    filename_parts: list[np.ndarray] = []
    cluster_parts: list[np.ndarray] = []
    normcluster_parts: list[np.ndarray] = []
    row_count = 0

    while True:
        chunks = []
        exhausted = []
        for iterator in iterators:
            try:
                chunks.append(next(iterator))
                exhausted.append(False)
            except StopIteration:
                exhausted.append(True)
        if any(exhausted):
            if not all(exhausted):
                raise ValueError("ANYA files do not have equal row counts")
            break
        if any(len(chunk) != len(chunks[0]) for chunk in chunks):
            raise ValueError("ANYA files are not aligned chunk-by-chunk")

        reference = chunks[0]
        for other in chunks[1:]:
            for field in ("filename", "clusterid", "normclusterid"):
                if not reference[field].equals(other[field]):
                    raise ValueError(f"Shared observation field {field!r} differs across files")

        predictions = np.vstack([
            chunk["Predicted_Label"].to_numpy(dtype=np.int8, copy=False)
            for chunk in chunks
        ])
        if not np.isin(predictions, [0, 1]).all():
            raise ValueError("Predicted_Label contains values outside binary {0, 1}")
        for (left_name, right_name), (left_index, right_index) in zip(PAIR_NAMES, combinations(range(4), 2)):
            np.add.at(pair_confusions[(left_name, right_name)], (predictions[left_index], predictions[right_index]), 1)
        total_predictions.append(predictions)
        filename_parts.append(reference["filename"].to_numpy(copy=True))
        cluster_parts.append(reference["clusterid"].to_numpy(copy=True))
        normcluster_parts.append(reference["normclusterid"].to_numpy(copy=True))
        row_count += len(reference)

    prediction_matrix = np.hstack(total_predictions) if total_predictions else np.empty((4, 0), dtype=np.int8)
    filenames = np.concatenate(filename_parts) if filename_parts else np.array([], dtype=object)
    clusterids = np.concatenate(cluster_parts) if cluster_parts else np.array([], dtype=object)
    normclusterids = np.concatenate(normcluster_parts) if normcluster_parts else np.array([], dtype=object)
    if prediction_matrix.shape[1] != row_count:
        raise AssertionError("Prediction matrix row count mismatch")

    pairwise = {
        f"{left}_vs_{right}": {
            **_pairwise_summary(prediction_matrix[i], prediction_matrix[j]),
        }
        for (left, right), (i, j) in zip(PAIR_NAMES, combinations(range(4), 2))
    }
    data = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": [path.name for path in MODEL_FILES.values()],
        "rows": row_count,
        "columns_used": USECOLS,
        "prediction_column": "Predicted_Label",
        "prediction_values_verified": [0, 1],
        "pairwise": pairwise,
        "row_consensus": _row_consensus_summary(prediction_matrix),
        "by_filename": _group_summary(filenames, prediction_matrix),
        "by_clusterid": _group_summary(clusterids, prediction_matrix),
        "by_normclusterid": _group_summary(normclusterids, prediction_matrix),
        "ground_truth_status": "not_verified; Predicted_Label is treated as pseudo-label/model output",
        "raw_files_modified": False,
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(_report(data), encoding="utf-8")
    print(json.dumps({
        "rows": row_count,
        "metrics": str(METRICS_PATH),
        "report": str(REPORT_PATH),
        "unanimous_rows": data["row_consensus"]["unanimous_rows"],
        "disagreement_rows": data["row_consensus"]["disagreement_rows_not_unanimous"],
        "two_vs_two_tie_rows": data["row_consensus"]["two_vs_two_tie_rows"],
    }, indent=2))


if __name__ == "__main__":
    main()
