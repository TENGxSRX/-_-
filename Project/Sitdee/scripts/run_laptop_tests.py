"""Run the existing tests that do not require Desktop-only raw datasets."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_GROUPS = (
    ("tests/test_laptop_dev.py", None),
    ("tests/test_live_webcam_v1.py", None),
    ("tests/test_temporal_and_serialization.py", None),
    ("tests/test_rgb_native_v4.py", "not v4_does_not_change_multiposture_raw_file"),
    ("tests/test_rgb_native_v5_contract.py", None),
    ("tests/test_grouped_splitting.py", None),
    ("tests/test_mvsp_manual_import.py", None),
    ("tests/test_posture_engine_v1.py", "not multiposture_adapter_exact_126_feature_schema"),
    ("tests/test_overnight_benchmark.py", None),
)


def main() -> int:
    failures: list[str] = []
    for relative_path, expression in TEST_GROUPS:
        path = ROOT / relative_path
        if not path.exists():
            failures.append(f"missing test module: {relative_path}")
            continue
        command = [sys.executable, "-m", "pytest", "-q", str(path)]
        if expression:
            command.extend(["-k", expression])
        print(f"\n[LAPTOP TEST] {relative_path}" + (f"  (-k {expression})" if expression else ""))
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            failures.append(relative_path)
    if failures:
        print("\nLAPTOP_TESTS_NEEDS_FIX")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("\nLAPTOP_TESTS_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
