from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.posture_ai.inference.pipeline import infer_source
from src.posture_ai.inference.ergonomic_pipeline import run_posture_engine


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "outputs" / "models" / "multiposture_v1"
POSE_MODEL = ROOT / "metadata" / "pose_landmarker_heavy.task"


@pytest.mark.integration
def test_prerecorded_video_source_runs_without_webcam(tmp_path):
    """Exercise the video path with a generated blank clip, never camera 0."""
    if not MODEL_DIR.joinpath("model.joblib").exists() or not POSE_MODEL.exists():
        pytest.skip("requires the exported model and MediaPipe task asset")

    video_path = tmp_path / "blank.avi"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (64, 64)
    )
    assert writer.isOpened()
    for _ in range(2):
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()

    predictions = infer_source(
        video_path,
        MODEL_DIR,
        POSE_MODEL,
        min_confidence=0.0,
        max_frames=2,
    )

    assert [item["frame"] for item in predictions] == [0, 1]
    assert all(item["status"] in {"no_pose_detected", "ok", "low_visibility_or_invalid_landmarks", "low_model_confidence"} for item in predictions)


@pytest.mark.integration
def test_posture_engine_writes_annotated_video_and_telemetry(tmp_path):
    """Exercise V1 video output without opening any camera device."""
    if not MODEL_DIR.joinpath("model.joblib").exists() or not POSE_MODEL.exists():
        pytest.skip("requires the exported model and MediaPipe task asset")
    video_path = tmp_path / "engine_blank.avi"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (64, 64))
    assert writer.isOpened()
    for _ in range(2):
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()
    annotated = tmp_path / "engine_analyzed.mp4"
    result = run_posture_engine(
        video_path,
        MODEL_DIR,
        POSE_MODEL,
        config_path=ROOT / "configs" / "posture_engine.yaml",
        max_frames=2,
        annotated_video=annotated,
        event_log=tmp_path / "events.jsonl",
        session_report=tmp_path / "session.json",
        performance_report=tmp_path / "performance.json",
    )
    assert len(result["records"]) == 2
    assert annotated.exists() and annotated.stat().st_size > 0
    assert (tmp_path / "events.jsonl").exists()
    assert (tmp_path / "session.json").exists()
    assert (tmp_path / "performance.json").exists()
    assert all(item["risk_level"] == "POSE_LOST" for item in result["records"])


@pytest.mark.integration
def test_posture_engine_accepts_image_input(tmp_path):
    if not MODEL_DIR.joinpath("model.joblib").exists() or not POSE_MODEL.exists():
        pytest.skip("requires the exported model and MediaPipe task asset")
    image_path = tmp_path / "engine_blank.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((64, 64, 3), dtype=np.uint8))
    result = run_posture_engine(
        image_path,
        MODEL_DIR,
        POSE_MODEL,
        config_path=ROOT / "configs" / "posture_engine.yaml",
        max_frames=1,
        event_log=tmp_path / "image-events.jsonl",
        session_report=tmp_path / "image-session.json",
        performance_report=tmp_path / "image-performance.json",
    )
    assert len(result["records"]) == 1
    assert result["records"][0]["status"] == "no_pose_detected"
