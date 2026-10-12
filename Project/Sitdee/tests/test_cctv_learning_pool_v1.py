from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.posture_ai.features.upper_body_posture_v2 import CORE_NAMES, INDICES
from src.posture_ai.inference.pose_v2_stage import PoseV2Result
from src.posture_ai.monitoring.cctv_learning_pool import LearningPool, PoolCapacityReached, observation, spatial_metadata, validate_record


def evidence():
    world = np.full((33, 3), np.nan)
    world[list(INDICES[:3])] = [[0, -0.9, -0.1], [-0.2, -0.6, 0], [0.2, -0.6, 0]]
    image = np.full((33, 3), np.nan)
    image[list(INDICES[:3])] = [[0.5, 0.2, 0], [0.3, 0.4, 0], [0.7, 0.4, 0]]
    return PoseV2Result(track_id=1, bbox=(10, 10, 110, 210), bbox_confidence=0.9, pose_state="POSE_OK", sampling_reason="SAMPLED", pose_available=True,
                        core_valid=True, pose_quality=1.0, landmark_visibility=dict.fromkeys(CORE_NAMES, 1.0), world_landmarks_xyz=world.tolist(), image_landmarks_xyz=image.tolist())


def record(seconds=0, track=1, primary="OOD"):
    p = {"output": "UNKNOWN" if primary != "VALID_NORMAL" else "CORRECT_SITTING", "subtype": None,
         "reason": "POSSIBLE_OOD_EXPERIMENTAL" if primary == "OOD" else "MODEL_DISAGREEMENT_EXPERIMENTAL" if primary == "UNCERTAIN" else "EXPERIMENTAL_SOURCE_DOMAIN",
         "ood": primary == "OOD", "confidence": None, "range_excess_max": 0.0, "label_source": "MODEL_PREDICTION"}
    return observation(replace(evidence(), track_id=track), session_id="TEST", seconds=seconds, frame_index=1, width=300, height=300,
                       predictions={"v1": deepcopy(p), "v2": deepcopy(p)}, tracking_state="ACTIVE")


def test_unlabeled_prediction_and_nested_gt_are_never_gt():
    r = record()
    validate_record(r)
    assert r["label_status"] == "UNLABELED" and r["ground_truth"] is None
    for key, value in (("label_status", "HUMAN_GROUND_TRUTH"), ("ground_truth", "TUP"), ("label_source", "MODEL_PREDICTION")):
        bad = deepcopy(r)
        bad[key] = value
        with pytest.raises(ValueError, match="GT"):
            validate_record(bad)
    bad = deepcopy(r)
    bad["diagnostics"]["v2"]["ground_truth"] = "TUP"
    with pytest.raises(ValueError, match="GT"):
        validate_record(bad)


def test_spatial_boundaries_and_invalid_bbox():
    assert spatial_metadata([0, 0, 20, 50], 100, 100)["scale_group"] == "NEAR"
    assert spatial_metadata([0, 0, 20, 25], 100, 100)["scale_group"] == "MID"
    assert spatial_metadata([0, 0, 20, 24.9], 100, 100)["scale_group"] == "FAR"
    assert spatial_metadata([0, 0, 20, 50], 100, 100)["zone"] == "0:0"
    with pytest.raises(ValueError):
        spatial_metadata([0, 0, np.inf, 100], 100, 100)


def test_quality_rejects_nonfinite_core_contamination_overlap_but_not_hips():
    r = record()
    assert r["features"] is not None and r["quality_rejection"] is None
    assert not r["landmarks"]["left_hip"]["available"]
    assert "OVERLAP_OR_OCCLUSION" in r["tags"]
    e = evidence()
    e.world_landmarks_xyz[11] = [np.nan] * 3
    bad = observation(e, session_id="TEST", seconds=0, frame_index=1, width=300, height=300, predictions=r["diagnostics"], tracking_state="ACTIVE")
    assert bad["quality_rejection"] == "INVALID_FEATURES_OR_CORE"
    validate_record(bad)
    e = evidence()
    e.image_landmarks_xyz[11][0] = 2
    assert observation(e, session_id="TEST", seconds=0, frame_index=1, width=300, height=300, predictions=r["diagnostics"], tracking_state="ACTIVE")["quality_rejection"] == "CROP_CONTAMINATION_PROXY"
    e = replace(evidence(), pose_state="POSE_AMBIGUOUS", core_valid=False)
    assert observation(e, session_id="TEST", seconds=0, frame_index=1, width=300, height=300, predictions=r["diagnostics"], tracking_state="ACTIVE")["quality_rejection"] == "SEVERE_OVERLAP"
    for value in (np.nan, np.inf):
        bad = record()
        bad["features"][0] = value
        with pytest.raises(ValueError, match="FEATURES"):
            validate_record(bad)


def test_dedup_gap_feature_change_refresh_and_state_change(tmp_path):
    p = LearningPool(tmp_path, "TEST", min_free_bytes=0)
    try:
        assert p.add(record(0))
        assert not p.add(record(1))
        assert not p.add(record(10))
        changed = record(11)
        changed["features"][3] += 1
        assert p.add(changed)
        assert not p.add(record(12))
        assert p.add(record(320))
        assert p.add(record(330, primary="UNCERTAIN"))
        assert p.counts["retained"] == 4 and p.counts["dedup_rejected"] == 3
    finally:
        p.close({}, "TEST_DONE")


def test_cross_track_duplicates_and_balanced_representative_retention(tmp_path):
    p = LearningPool(tmp_path, "TEST", min_free_bytes=0, max_representatives=4)
    try:
        assert p.add(record(0, 1))
        assert not p.add(record(1, 2))
        for i in range(2, 6):
            r = record(i * 10, i)
            r["features"][3] += i
            p.add(r)
        diverse = record(70, 7, "VALID_NORMAL")
        diverse["spatial"]["scale_group"] = "FAR"
        assert p.add(diverse)
        assert len(p.representatives) == 4
        assert any(cell[0][0] == "FAR" and cell[0][2] == "VALID_NORMAL" for cell in p.representatives)
    finally:
        p.close({}, "TEST_DONE")


def test_capacity_file_and_root_bytes_finalize_without_deleting(tmp_path):
    p = LearningPool(tmp_path / "files", "TEST", min_free_bytes=0)
    p.observations.max_bytes, p.observations.max_files = 4096, 1
    try:
        p.add(record(0))
        with pytest.raises(PoolCapacityReached, match="FILE"):
            for i in range(1, 10):
                p.add(record(i))
    finally:
        p.close({}, "CAPACITY")
    assert p.closed and p.observations._handle is None and p.indices._handle is None
    assert json.loads((p.directory / "manifest.json").read_text())["stopped_reason"] == "CAPACITY"
    assert list(p.directory.glob("observations*.jsonl"))
    q = LearningPool(tmp_path / "bytes", "TEST", max_bytes=2 * 1024**2, min_free_bytes=0)
    try:
        large = record()
        large["padding"] = "x" * 600000
        q.add(large)
        with pytest.raises(PoolCapacityReached, match="BYTE"):
            q.add(large)
    finally:
        q.close({}, "CAPACITY")
    assert sum(f.stat().st_size for f in q.directory.iterdir() if f.is_file()) < 2 * 1024**2


def test_manifest_references_no_duplicate_payload_lock_and_clean_shutdown(tmp_path):
    p = LearningPool(tmp_path, "TEST", min_free_bytes=0)
    with pytest.raises(FileExistsError):
        LearningPool(tmp_path, "SECOND", min_free_bytes=0)
    p.event(1, "NEW", 0)
    p.add(record())
    p.close({"processed_frames": 2}, "DURATION_COMPLETED")
    p.close({}, "IGNORED")
    manifest = json.loads((p.directory / "manifest.json").read_text())
    assert manifest["state"] == "FINALIZED" and manifest["session_ids"] == ["TEST"]
    assert manifest["sample_counts"]["retained"] == 1 and manifest["runtime"]["processed_frames"] == 2
    indices = [json.loads(l) for f in p.directory.glob("candidates*.jsonl") for l in f.read_text().splitlines()]
    assert "features" not in indices[0] and indices[0]["ground_truth"] is None
    reps = json.loads((p.directory / "representatives.json").read_text())
    assert reps["sample_ids"] == [indices[0]["sample_id"]]
    assert not p.lock.exists()
    with pytest.raises(ValueError, match="CLOSED"):
        p.add(record())


def test_credential_identity_and_pseudo_label_fields_never_written(tmp_path):
    p = LearningPool(tmp_path, "TEST", min_free_bytes=0)
    try:
        for key, value in (("rtsp_url", "rtsp://user:secret@host"), ("name", "Alice"), ("pseudo_label", "TUP")):
            bad = record()
            bad[key] = value
            with pytest.raises(ValueError):
                p.add(bad)
        bad = record()
        bad["diagnostics"]["v2"]["reason"] = "rtsp://user:secret@host"
        with pytest.raises(ValueError):
            p.add(bad)
        assert p.observations.records_written == 0
    finally:
        p.close({}, "TEST_DONE")
    assert "secret" not in (p.directory / "manifest.json").read_text()


def test_disk_preflight_and_runtime_reserve_cleanup(tmp_path, monkeypatch):
    from src.posture_ai.monitoring import cctv_learning_pool as module
    p = LearningPool(tmp_path / "running", "TEST", min_free_bytes=1)
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    try:
        with pytest.raises(PoolCapacityReached, match="DISK"):
            p.add(record())
    finally:
        p.close({}, "DISK_RESERVE")
    with pytest.raises(PoolCapacityReached, match="PREFLIGHT"):
        LearningPool(tmp_path / "preflight", "TEST", min_free_bytes=1)
    assert not (tmp_path / "preflight/.collector.lock").exists()


def test_representatives_are_reproducible_and_do_not_freeze_first_cells(tmp_path):
    selections = []
    for root in (tmp_path / "first", tmp_path / "second"):
        p = LearningPool(root, "TEST", min_free_bytes=0, max_representatives=3)
        try:
            for i in range(20):
                r = record(i * 10, 1)
                r["features"][3] += i
                p.add(r)
            selections.append(sorted(v[0] for v in p.representatives.values()))
        finally:
            p.close({}, "TEST_DONE")
    assert selections[0] == selections[1]
    assert selections[0] != ["TEST:1", "TEST:2", "TEST:3"]


def test_malformed_features_and_no_lower_body_payload():
    r = record()
    assert set(r["landmarks"]) == {"nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"}
    assert not any(name in json.dumps(r) for name in ("knee", "ankle", "heel", "foot"))
    for features in ([1, 2], ["bad"] * 6, None):
        bad = deepcopy(r)
        bad["features"] = features
        with pytest.raises(ValueError):
            validate_record(bad)


def test_nonfinite_extractor_result_is_counted_rejection_not_a_collector_crash(monkeypatch):
    from src.posture_ai.monitoring import cctv_learning_pool as module
    from scripts.collect_cctv_learning_pool_v1 import diagnostic
    monkeypatch.setattr(module, "build_features", lambda *a: np.full((1, 6), np.inf))
    r = record()
    assert r["features"] is None and r["quality_rejection"] == "INVALID_FEATURES_OR_CORE"
    validate_record(r)
    prediction = {"output": "UNKNOWN", "subtype": None, "reason": "INVALID_FEATURES"}
    assert diagnostic(prediction, {}, np.full((1, 6), np.inf))["range_excess_max"] is None


@pytest.mark.parametrize("error", [KeyboardInterrupt(), RuntimeError("rtsp://user:secret@host")])
def test_actual_collector_loop_shutdown_and_secret_exception_text(tmp_path, monkeypatch, capsys, error):
    from scripts import collect_cctv_learning_pool_v1 as lab
    resources = {}

    class Input:
        def __init__(self, *args, **kwargs):
            self.reads, self.closed = 0, False
            resources["source"] = self

        def open(self):
            return True

        def read(self):
            self.reads += 1
            if self.reads > 2:
                raise error
            return SimpleNamespace(ok=True, frame=np.zeros((300, 300, 3), dtype=np.uint8), frame_index=self.reads)

        def close(self):
            self.closed = True

        def health(self):
            return {"source_status": "SOURCE_CLOSED" if self.closed else "SOURCE_READY", "reconnect_count": 0}

    class Stage:
        def __init__(self, *args, **kwargs):
            self.closed = False
            resources["stage"] = self

        def process(self, frame, selected, seconds, **kwargs):
            return [evidence()] if selected else []

        def close(self):
            self.closed = True

    from src.posture_ai.detection.yolo_optional import PersonBox
    tracker = SimpleNamespace(track=lambda frame: [PersonBox(10, 10, 110, 210, 0.9, track_id=1)])
    monkeypatch.setattr(lab, "model_hashes", lambda: {"v4": "UNCHANGED"})
    monkeypatch.setattr(lab, "VideoSource", Input)
    monkeypatch.setattr(lab, "LatestFrameInput", lambda source: source)
    monkeypatch.setattr(lab, "UltralyticsPersonTracker", lambda *a, **kw: tracker)
    monkeypatch.setattr(lab, "MediaPipeCropPoseBackend", lambda *a, **kw: None)
    monkeypatch.setattr(lab, "PoseV2Stage", Stage)
    monkeypatch.setattr(lab, "UpperBodyPostureRunner", lambda: SimpleNamespace(bundle={"minimum": np.full(8, -10), "maximum": np.full(8, 10)}))
    monkeypatch.setattr(lab, "UpperBodyPostureV2Runner", lambda: SimpleNamespace(bundle={"condition": "NO_HIP", "minimum": np.full(6, -10), "maximum": np.full(6, 10)}))
    pred = {"output": "UNKNOWN", "subtype": None, "reason": "POSSIBLE_OOD_EXPERIMENTAL"}
    monkeypatch.setattr(lab, "paired_predictions", lambda *a: ("both_hidden", pred, pred))
    ticks = iter(i * 0.6 for i in range(100))
    monkeypatch.setattr(lab.time, "perf_counter", lambda: next(ticks))
    args = SimpleNamespace(output_root=tmp_path, max_pool_mib=10, max_files=4, min_free_mib=0, max_representatives=4, duration_minutes=10, max_reconnects=5)
    manifest = lab.run(args)
    assert resources["source"].closed and resources["stage"].closed
    assert manifest["state"] == "FINALIZED" and manifest["runtime"]["source_closed"]
    assert manifest["limits"]["source_max_reconnects"] == 5 and manifest["limits"]["track_id_capacity"] == 65536
    assert manifest["runtime"]["pose_attempts"] == 2 and manifest["runtime"]["valid_upper_body_poses"] == 2
    assert "secret" not in capsys.readouterr().out
    assert all("secret" not in p.read_text() for p in tmp_path.rglob("*.json*"))
