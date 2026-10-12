"""Fast, non-destructive health check for the lightweight Acer checkout.

This script never downloads data, opens a camera by default, or changes model
and dataset files. Use --probe-camera only when an explicit camera check is
wanted; camera failure is reported separately and never blocks development.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MIN_FREE_BYTES = 15 * 1024**3


def _version(module_name: str, attribute: str = "__version__") -> tuple[bool, str]:
    try:
        module = __import__(module_name)
        return True, str(getattr(module, attribute, "installed"))
    except Exception as exc:  # pragma: no cover - exercised on an unconfigured laptop
        return False, f"{type(exc).__name__}: {exc}"


def _memory_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    if os.name != "nt":
        return None
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_phys", ctypes.c_ulonglong),
            ("avail_phys", ctypes.c_ulonglong),
            ("total_page", ctypes.c_ulonglong),
            ("avail_page", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("avail_virtual", ctypes.c_ulonglong),
            ("avail_extended", ctypes.c_ulonglong),
        ]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    try:
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total_phys)
    except Exception:
        return None
    return None


def _git_status() -> dict[str, Any]:
    git_dir = ROOT / ".git"
    if not git_dir.exists():
        return {"repository": False, "status": "NOT_CONFIGURED", "branch": None, "remotes": []}
    try:
        status = subprocess.run(
            ["git", "status", "--short", "--branch"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        remote = subprocess.run(
            ["git", "remote", "-v"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        return {
            "repository": status.returncode == 0,
            "status": status.stdout.strip() or "CLEAN",
            "branch": status.stdout.splitlines()[0] if status.stdout else None,
            "remotes": remote.stdout.strip().splitlines() if remote.stdout.strip() else [],
        }
    except OSError as exc:
        return {"repository": False, "status": f"GIT_UNAVAILABLE: {exc}", "branch": None, "remotes": []}


def _load_v4() -> dict[str, Any]:
    model_dir = ROOT / "outputs" / "models" / "rgb_native_v4"
    required = ["model.joblib", "feature_config.json", "labels.json", "metadata.json"]
    missing = [name for name in required if not (model_dir / name).exists()]
    if missing:
        return {"ok": False, "status": "MISSING_FILES", "missing": missing}
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from src.posture_ai.inference.rgb_native_v4 import RGBNativeV4Runner

        runner = RGBNativeV4Runner(model_dir)
        return {
            "ok": True,
            "status": "LOADED",
            "model_type": type(runner.model).__name__,
            "family": runner.family,
            "labels": runner.labels,
            "feature_count": len(runner.expected_names),
        }
    except Exception as exc:
        return {"ok": False, "status": f"LOAD_FAILED: {type(exc).__name__}: {exc}"}


def _camera_check(index: int) -> dict[str, Any]:
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from src.posture_ai.inference.webcam_v1 import probe_camera

        result = probe_camera(index)
        result["status"] = "AVAILABLE" if result.get("readable_frame") else "NOT_AVAILABLE"
        return result
    except Exception as exc:
        return {"status": "NOT_AVAILABLE", "error": f"{type(exc).__name__}: {exc}"}


def run(*, probe_camera_index: int | None = None) -> tuple[dict[str, Any], int]:
    disk = shutil.disk_usage(ROOT.anchor or ROOT)
    free_gib = disk.free / 1024**3
    modules = {
        "opencv": _version("cv2"),
        "mediapipe": _version("mediapipe"),
        "scikit_learn": _version("sklearn"),
        "joblib": _version("joblib"),
        "yaml": _version("yaml"),
        "pytest": _version("pytest"),
    }
    required_modules_ok = all(value[0] for value in modules.values())
    files = {
        "posture_engine_config": (ROOT / "configs" / "posture_engine.yaml").exists(),
        "low_spec_config": (ROOT / "configs" / "laptop_low_spec.yaml").exists(),
        "pose_model": (ROOT / "metadata" / "pose_landmarker_heavy.task").exists(),
    }
    python_64bit = sys.maxsize > 2**32
    python_ok = sys.version_info[:2] == (3, 12) and python_64bit
    v4 = _load_v4()
    payload: dict[str, Any] = {
        "status": None,
        "root": str(ROOT),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "hostname": platform.node(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "python_recommendation_match": python_ok,
        "python_64bit": python_64bit,
        "cpu": platform.processor() or platform.machine(),
        "logical_cores": os.cpu_count(),
        "ram_gib": (_memory_bytes() / 1024**3) if _memory_bytes() is not None else None,
        "disk": {
            "free_gib": round(free_gib, 2),
            "total_gib": round(disk.total / 1024**3, 2),
            "warning": free_gib < 15.0,
            "rule": "Warn below 15 GiB; never delete files automatically.",
        },
        "modules": {name: {"ok": ok, "version": version} for name, (ok, version) in modules.items()},
        "files": files,
        "v4": v4,
        "git": _git_status(),
        "camera": "NOT_PROBED",
    }
    if probe_camera_index is not None:
        payload["camera"] = _camera_check(probe_camera_index)
    needs_fix = (
        not python_ok
        or not required_modules_ok
        or not all(files.values())
        or not bool(v4.get("ok"))
        or free_gib < 15.0
    )
    payload["status"] = "LAPTOP_DEV_NEEDS_FIX" if needs_fix else "LAPTOP_DEV_READY"
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if free_gib < 15.0:
        print("WARNING: free disk is below 15 GiB. Do not download datasets or large videos.")
    if probe_camera_index is None:
        print("Camera: NOT_PROBED (this is intentional; use --probe-camera only when explicitly needed).")
    elif payload["camera"].get("status") == "NOT_AVAILABLE":
        print("Camera: NOT_AVAILABLE; this does not block laptop development.")
    print(payload["status"])
    return payload, 0 if payload["status"] == "LAPTOP_DEV_READY" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Non-destructive Acer development health check")
    parser.add_argument("--probe-camera", type=int, metavar="INDEX", help="Explicitly probe one camera index; optional and non-blocking")
    args = parser.parse_args()
    _, exit_code = run(probe_camera_index=args.probe_camera)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
