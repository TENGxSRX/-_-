"""Bounded unlabeled CCTV observations and deterministic representative indices."""
from __future__ import annotations

from collections import Counter, OrderedDict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from src.posture_ai.features.upper_body_posture_v2 import CORE_NAMES, INDICES, NO_HIP_NAMES, build_features
from src.posture_ai.inference.upper_body_posture_v2 import evidence_xyz
from src.posture_ai.monitoring.long_session import atomic_write_json
from src.posture_ai.monitoring.shadow import RotatingJsonlWriter

SCHEMA = "CCTV_LEARNING_POOL_V1"
ANCHORS = (*CORE_NAMES, "left_hip", "right_hip")
TAGS = {"VALID_NORMAL", "OOD", "UNCERTAIN", "NOVEL_DIVERSE", "OVERLAP_OR_OCCLUSION", "TRACKING_EVENT"}


class PoolCapacityReached(RuntimeError):
    pass


def spatial_metadata(bbox, width, height):
    b = np.asarray(bbox, dtype=float)
    if b.shape != (4,) or not np.isfinite(b).all() or width <= 0 or height <= 0 or b[2] <= b[0] or b[3] <= b[1]:
        raise ValueError("INVALID_BBOX")
    cx, cy = (b[0] + b[2]) / (2 * width), (b[1] + b[3]) / (2 * height)
    fraction = (b[3] - b[1]) / height
    return {"center_normalized": [float(cx), float(cy)], "height_fraction": float(fraction),
            "area_fraction": float((b[2] - b[0]) * (b[3] - b[1]) / (width * height)),
            "scale_group": "NEAR" if fraction >= 0.5 else "MID" if fraction >= 0.25 else "FAR",
            "zone": f"{min(2, max(0, int(cx * 3)))}:{min(2, max(0, int(cy * 3)))}"}


def finite_point(points, index):
    try:
        a = np.asarray(points[index], dtype=float)
        return a.tolist() if a.shape == (3,) and np.isfinite(a).all() else None
    except (IndexError, TypeError, ValueError):
        return None


def observation(evidence, *, session_id, seconds, frame_index, width, height, predictions, tracking_state):
    """Construct fields explicitly: caller metadata cannot introduce GT or identities."""
    record = {"schema_version": SCHEMA, "record_type": "POSE_OBSERVATION", "session_id": session_id,
              "timestamp_utc": datetime.now(timezone.utc).isoformat(), "timestamp_seconds": seconds,
              "frame_index": frame_index, "anonymous_track_id": evidence.track_id, "bbox": list(evidence.bbox),
              "detector_confidence": evidence.bbox_confidence, "spatial": spatial_metadata(evidence.bbox, width, height),
              "pose_state": evidence.pose_state, "ambiguity_state": evidence.pose_state == "POSE_AMBIGUOUS",
              "pose_quality": evidence.pose_quality, "tracking_state": tracking_state,
              "landmarks": {}, "features": None, "feature_schema": "upper_body_posture_v2/NO_HIP",
              "diagnostics": predictions, "tags": [], "quality_rejection": None,
              "label_status": "UNLABELED", "ground_truth": None, "label_source": None,
              "source_domain": "HIKVISION_CCTV", "source_camera_profile": "HIKVISION_MAIN", "raw_media_recorded": False}
    for name, index in zip(ANCHORS, INDICES):
        world = finite_point(evidence.world_landmarks_xyz, index)
        image = finite_point(evidence.image_landmarks_xyz, index)
        visibility = evidence.landmark_visibility.get(name)
        visibility = float(visibility) if visibility is not None and np.isfinite(visibility) else None
        record["landmarks"][name] = {"world_xyz": world, "crop_image_xyz": image, "visibility": visibility,
                                     "coordinate_present": world is not None, "presence_metric": None,
                                     "available": world is not None and visibility is not None and visibility >= 0.5}
    if evidence.pose_state == "POSE_AMBIGUOUS":
        record["quality_rejection"] = "SEVERE_OVERLAP"
        record["tags"] = ["OVERLAP_OR_OCCLUSION"]
    elif evidence.pose_state != "POSE_OK" or not evidence.core_valid:
        record["quality_rejection"] = "POSE_UNAVAILABLE" if evidence.pose_state == "POSE_UNAVAILABLE" else "INVALID_CORE"
    else:
        try:
            x = build_features(evidence_xyz(evidence), "NO_HIP")[0]
            if not np.isfinite(x).all():
                raise ValueError("INVALID_FEATURES")
            images = [record["landmarks"][n]["crop_image_xyz"] for n in CORE_NAMES]
            if any(p is None or not (-0.05 <= p[0] <= 1.05 and -0.05 <= p[1] <= 1.05) for p in images):
                record["quality_rejection"] = "CROP_CONTAMINATION_PROXY"
            else:
                record["features"] = x.tolist()
        except (TypeError, ValueError):
            record["quality_rejection"] = "INVALID_FEATURES_OR_CORE"
    if record["quality_rejection"] is None:
        v2 = predictions["v2"]
        record["tags"] = ["OOD" if v2["ood"] else "UNCERTAIN" if v2["output"] == "UNKNOWN" else "VALID_NORMAL"]
        if not all(record["landmarks"][n]["available"] for n in ANCHORS[3:]):
            record["tags"].append("OVERLAP_OR_OCCLUSION")  # confidence proxy, never physical occlusion GT
    if tracking_state != "ACTIVE":
        record["tags"].append("TRACKING_EVENT")
    return record


def validate_record(record):
    if record.get("label_status") != "UNLABELED" or record.get("ground_truth") is not None or record.get("label_source") is not None:
        raise ValueError("AUTOMATIC_GT_FORBIDDEN")
    if record.get("source_domain") != "HIKVISION_CCTV" or not set(record.get("tags", [])) <= TAGS:
        raise ValueError("INVALID_POOL_METADATA")
    x = record.get("features")
    if record.get("record_type") == "POSE_OBSERVATION" and record.get("quality_rejection") is None and x is None:
        raise ValueError("INVALID_FEATURES")
    if x is not None and (np.asarray(x).shape != (6,) or not np.isfinite(np.asarray(x, dtype=float)).all()):
        raise ValueError("INVALID_FEATURES")
    def privacy(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if (key == "ground_truth" and item is not None) or (key == "label_status" and item != "UNLABELED"):
                    raise ValueError("AUTOMATIC_GT_FORBIDDEN")
                if key.lower() in {"password", "username", "url", "rtsp_url", "name", "identity", "human_label", "pseudo_label"}:
                    raise ValueError("PRIVATE_OR_LABEL_FIELD_FORBIDDEN")
                privacy(item)
        elif isinstance(value, list):
            for item in value:
                privacy(item)
        elif isinstance(value, str) and ("://" in value or "HUMAN_GROUND_TRUTH" in value):
            raise ValueError("SECRET_OR_GT_VALUE_FORBIDDEN")
    privacy(record)
    json.dumps(record, allow_nan=False)


class LearningPool:
    def __init__(self, root, session_id, *, max_bytes=512 * 1024**2, max_files=64, min_free_bytes=512 * 1024**2, max_representatives=512):
        if max_bytes < 2 * 1024**2 or max_files < 4 or min_free_bytes < 0 or not 1 <= max_representatives <= 4096:
            raise ValueError("INVALID_STORAGE_LIMITS")
        if Path(session_id).name != session_id or not session_id or session_id in {".", ".."}:
            raise ValueError("INVALID_SESSION_ID")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = self.root / ".collector.lock"
        with self.lock.open("x"):
            pass
        try:
            self.existing_bytes = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
            if self.existing_bytes + 1024**2 >= max_bytes or shutil.disk_usage(self.root).free < min_free_bytes + 1024**2:
                raise PoolCapacityReached("STORAGE_PREFLIGHT")
            self.directory = self.root / session_id
            self.directory.mkdir(exist_ok=False)
            self.max_bytes, self.min_free_bytes = max_bytes, min_free_bytes
            self.max_representatives = max_representatives
            self.observations = RotatingJsonlWriter(self.directory, "observations", max_bytes=8 * 1024**2, max_files=max_files * 3 // 4)
            self.indices = RotatingJsonlWriter(self.directory, "candidates", max_bytes=8 * 1024**2, max_files=max_files - max_files * 3 // 4)
            self.counts, self.rejections, self.coverage, self.retained_coverage = Counter(), Counter(), Counter(), Counter()
            self.recent, self.cells, self.representatives = OrderedDict(), OrderedDict(), {}
            self.last_disk_check, self.closed = -1e12, False
            self.manifest = {"schema_version": SCHEMA, "pool_identity": SCHEMA, "feature_schema": "upper_body_posture_v2/NO_HIP",
                             "feature_names": NO_HIP_NAMES, "landmark_names": ANCHORS, "source_camera_profile": "HIKVISION_MAIN",
                             "created_utc": datetime.now(timezone.utc).isoformat(), "session_ids": [session_id], "state": "RUNNING",
                             "label_policy": "UNLABELED_ONLY; ground_truth=null; predictions are diagnostic, never GT; no training",
                             "privacy_policy": "No pixels, clips, faces, names, identities or URLs; transient track IDs only",
                             "limits": {"root_max_bytes": max_bytes, "session_max_jsonl_files": max_files, "min_free_bytes": min_free_bytes, "max_representatives": max_representatives, "finalization_reserve_bytes": 1024**2},
                             "selection_policy": {"min_gap_seconds": 5, "feature_rms_distance": 0.10, "refresh_seconds": 300,
                                                  "cross_track_recent_seconds": 60, "cross_track_rms_distance": 0.05,
                                                  "feature_cell_width": 0.15, "time_stratum_seconds": 1800,
                                                  "scale": "bbox_height/image_height: NEAR>=0.5, MID>=0.25, otherwise FAR", "zone": "3x3 normalized image grid"}}
            self.checkpoint({})
        except BaseException:
            self.lock.unlink(missing_ok=True)
            raise

    def _write(self, writer, record, seconds):
        validate_record(record)
        size = len((json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8"))
        if self.existing_bytes + self.observations.bytes_written + self.indices.bytes_written + size + 1024**2 > self.max_bytes:
            raise PoolCapacityReached("ROOT_BYTE_CAPACITY")
        if seconds - self.last_disk_check >= 5:
            self.last_disk_check = seconds
            if shutil.disk_usage(self.root).free < self.min_free_bytes + 1024**2:
                raise PoolCapacityReached("DISK_RESERVE")
        if not writer.write(record, now_seconds=seconds):
            raise PoolCapacityReached("JSONL_FILE_CAPACITY")

    def add(self, record):
        if self.closed:
            raise ValueError("POOL_CLOSED")
        validate_record(record)
        seconds, track = record["timestamp_seconds"], record["anonymous_track_id"]
        sample_id = f"{self.manifest['session_ids'][0]}:{self.observations.records_written + 1}"
        record = {**record, "sample_id": sample_id}
        rejection = record["quality_rejection"]
        retain, cell, stratum = False, None, None
        if rejection:
            self.rejections[rejection] += 1
        else:
            x = np.asarray(record["features"])
            primary = record["tags"][0]
            stratum = (record["spatial"]["scale_group"], record["spatial"]["zone"], primary, int(seconds // 1800))
            cell = (stratum, tuple(np.floor(x / 0.15).astype(int)))
            self.counts["valid"] += 1
            self.counts[primary] += 1
            self.coverage[record["spatial"]["scale_group"]] += 1
            previous = self.recent.get(track)
            retain = previous is None or (seconds - previous[0] >= 5 and
                       (np.linalg.norm(x - previous[1]) / np.sqrt(6) >= 0.10 or stratum != previous[2] or seconds - previous[0] >= 300))
            if retain and record["tracking_state"] == "ACTIVE":
                retain = not any(seconds - t <= 60 and s == stratum and np.linalg.norm(x - v) / np.sqrt(6) < 0.05
                                 for tid, (t, v, s) in self.recent.items() if tid != track)
            if cell not in self.cells:
                record["tags"] = [*record["tags"], "NOVEL_DIVERSE"]
            self.counts["dedup_rejected"] += int(not retain)
        self._write(self.observations, record, seconds)
        if retain:
            index = {"schema_version": SCHEMA, "sample_id": sample_id, "observation_file": self.observations.files[-1],
                     "observation_ordinal": self.observations.records_written, "tags": record["tags"], "source_domain": "HIKVISION_CCTV",
                     "label_status": "UNLABELED", "ground_truth": None, "label_source": None}
            self._write(self.indices, index, seconds)
            self.recent.pop(track, None)
            self.recent[track] = (seconds, x.copy(), stratum)
            while len(self.recent) > 4096:
                self.recent.popitem(last=False)
            self.counts["retained"] += 1
            self.retained_coverage[record["spatial"]["scale_group"]] += 1
            self.cells[cell] = None
            # ponytail: bounded recent cells; recurring regions may appear novel after eviction.
            while len(self.cells) > 4096:
                self.cells.popitem(last=False)
            self._represent(cell, sample_id, track)
        return retain

    def event(self, track, state, seconds):
        if self.closed or state not in {"NEW", "LOST", "REACQUIRED"}:
            raise ValueError("INVALID_TRACK_EVENT")
        record = {"schema_version": SCHEMA, "record_type": "TRACKING_EVENT", "timestamp_seconds": seconds,
                  "timestamp_utc": datetime.now(timezone.utc).isoformat(), "session_id": self.manifest["session_ids"][0],
                  "anonymous_track_id": track, "tracking_state": state, "tags": ["TRACKING_EVENT"],
                  "source_domain": "HIKVISION_CCTV", "label_status": "UNLABELED", "ground_truth": None, "label_source": None}
        self._write(self.observations, record, seconds)
        self.counts["tracking:" + state] += 1

    def _represent(self, cell, sample_id, track):
        if cell in self.representatives:
            return
        rank = hashlib.sha256(sample_id.encode()).hexdigest()
        if len(self.representatives) >= self.max_representatives:
            strata = Counter(k[0] for k in self.representatives)
            tracks = Counter(v[1] for v in self.representatives.values())
            # ponytail: O(cap) scan, bounded at 512 by default; offline selection for larger pools.
            victim = max(self.representatives, key=lambda k: (strata[k[0]], tracks[self.representatives[k][1]], self.representatives[k][2]))
            old = self.representatives[victim]
            if cell[0] != victim[0] and strata[cell[0]] >= strata[victim[0]]:
                return
            if (cell[0] == victim[0] or strata[cell[0]] == strata[victim[0]] - 1) and tracks[track] >= tracks[old[1]] and rank >= old[2]:
                return
            del self.representatives[victim]
        self.representatives[cell] = (sample_id, track, rank)

    def checkpoint(self, runtime):
        self.manifest.update({"sample_counts": dict(self.counts), "quality_rejections": dict(self.rejections),
                              "spatial_coverage": dict(self.coverage), "retained_spatial_coverage": dict(self.retained_coverage),
                              "representative_samples": len(self.representatives), "runtime": runtime,
                              "storage": {"observations": self.observations.summary(), "candidate_indices": self.indices.summary()}})
        atomic_write_json(self.directory / "manifest.json", self.manifest)

    def close(self, runtime, reason):
        if self.closed:
            return
        try:
            self.observations.close()
            self.indices.close()
            representatives = {"schema_version": SCHEMA, "label_status": "UNLABELED", "ground_truth": None,
                               "selection": "One per feature cell; balance scale/zone/status/time strata and tracks, deterministic SHA256 tie-break",
                               "sample_ids": sorted(v[0] for v in self.representatives.values())}
            atomic_write_json(self.directory / "representatives.json", representatives)
            self.manifest.update(state="FINALIZED", stopped_reason=reason)
            self.checkpoint(runtime)
        finally:
            self.closed = True
            self.lock.unlink(missing_ok=True)
