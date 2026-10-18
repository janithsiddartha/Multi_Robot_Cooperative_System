"""Persistent Target IK and Motion Look-Up Table (LUT) Cache.

Provides cross-stage persistent caching of inverse kinematics solutions
for target coordinates and orientations.

Features:
- Deterministic config hashing (MD5 / SHA-256 fingerprint): Automatically detects
  any changes in robot DH parameters, base transformations, TCP offsets, or
  handover configurations, preventing physical collision from stale data.
- Compact Target-IK caching: Caches all reachable robot solutions for a target
  without bloating the file with thousands of trajectory interpolation points.
- Robot-Selection Flexibility: Preserves all candidate robot solutions so users
  can choose which robot to perform the pick or place operation.
- Cross-stage unified format (Stage 1, Stage 1.5, Stage 2, Stage 2.5).
"""

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from robot_control.handover.stage2_5 import Stage2_5TargetIKResult


def compute_config_fingerprint(
    robots: Dict[str, Any],
    handover_cfg: Optional[dict] = None
) -> str:
    """Compute a deterministic 16-character SHA-256 fingerprint of robot kinematics and setup."""
    tokens = []
    for rid in sorted(robots.keys()):
        cfg = robots[rid]
        tokens.append(f"rid:{rid}")
        tokens.append(f"a:{list(np.round(np.asarray(cfg.a, dtype=float), 4))}")
        alpha_vals = getattr(cfg, 'alpha_deg', getattr(cfg, 'alpha_rad', []))
        tokens.append(f"alpha:{list(np.round(np.asarray(alpha_vals, dtype=float), 4))}")
        tokens.append(f"d:{list(np.round(np.asarray(cfg.d, dtype=float), 4))}")
        tokens.append(f"theta_off:{list(np.round(np.asarray(cfg.theta_offset_deg, dtype=float), 4))}")
        tokens.append(f"limits:{list(np.round(np.asarray(cfg.joint_limits_deg, dtype=float), 2).flatten())}")
        tokens.append(f"base_xyz:{list(np.round(np.asarray(cfg.world_base_xyz, dtype=float), 2))}")
        tokens.append(f"base_R:{list(np.round(np.asarray(cfg.world_base_R, dtype=float), 4).flatten())}")
        tokens.append(f"tcp_off:{list(np.round(np.asarray(cfg.tcp_offset, dtype=float), 2))}")
        tokens.append(f"grp_close:{round(float(getattr(cfg, 'gripper_close', 55.0)), 2)}")
        tokens.append(f"grp_open:{round(float(getattr(cfg, 'gripper_open', -55.0)), 2)}")
        tokens.append(f"grp_axis:{getattr(cfg, 'gripper_finger_axis', 'x')}")
        tokens.append(f"grp_app:{getattr(cfg, 'gripper_approach_axis', 'z')}")

    if handover_cfg:
        ho = handover_cfg.get('handover', {})
        tokens.append(f"safe_dist:{round(float(ho.get('safe_distance', 50.0)), 2)}")
        pos = ho.get('handover_positions', {})
        for pair in sorted(pos.keys()):
            tokens.append(f"pos_{pair}:{pos[pair]}")
        pre_pos = ho.get('pre_handover_positions', {})
        for pair in sorted(pre_pos.keys()):
            tokens.append(f"pre_pos_{pair}:{pre_pos[pair]}")

    full_str = "|".join(tokens)
    return hashlib.sha256(full_str.encode('utf-8')).hexdigest()[:16]


def resolve_cache_filepath(config_dir: Optional[str] = None) -> str:
    """Resolve the JSON cache filepath, preferring workspace source repo config."""
    cur = os.path.abspath(os.getcwd())
    while cur != '/':
        cand = os.path.join(cur, 'src', 'robot_control', 'config')
        if os.path.isdir(cand):
            return os.path.join(cand, 'trajectory_cache.json')
        cur = os.path.dirname(cur)

    if config_dir and os.path.isdir(config_dir):
        return os.path.join(config_dir, 'trajectory_cache.json')

    return os.path.join(os.getcwd(), 'trajectory_cache.json')


def target_cache_key(
    xyz: Union[List[float], Tuple[float, ...], np.ndarray],
    yaw_deg: float = 0.0
) -> str:
    """Generate normalized string key for a target coordinate and yaw."""
    arr = [round(float(v), 2) for v in xyz]
    yaw = round(float(yaw_deg) % 360.0, 1)
    return f"{arr[0]},{arr[1]},{arr[2]},{yaw}"


def target_result_to_dict(t: Stage2_5TargetIKResult) -> dict:
    """Serialize a Stage2_5TargetIKResult to dict."""
    return {
        "robot_id": str(t.robot_id),
        "xyz_world": [round(float(v), 2) for v in t.xyz_world],
        "xyz_base": [round(float(v), 2) for v in t.xyz_base],
        "q_deg": [round(float(v), 2) for v in t.q_deg],
        "error_mm": round(float(t.error_mm), 3),
        "solver_type": str(t.solver_type),
        "is_fallback": bool(t.is_fallback),
        "requested_yaw_deg": round(float(t.requested_yaw_deg), 1) if t.requested_yaw_deg is not None else None,
        "approach_tilt_deg": round(float(t.approach_tilt_deg), 1) if t.approach_tilt_deg is not None else None,
        "alignment_error_deg": round(float(t.alignment_error_deg), 2) if t.alignment_error_deg is not None else None,
        "score": round(float(t.score), 2) if t.score is not None else 0.0,
        "solution_count": int(t.solution_count),
    }


def dict_to_target_result(td: dict) -> Stage2_5TargetIKResult:
    """Deserialize dict to Stage2_5TargetIKResult."""
    q_deg = np.array(td.get("q_deg", []), dtype=float)
    return Stage2_5TargetIKResult(
        robot_id=td.get("robot_id", ""),
        xyz_world=np.array(td.get("xyz_world", []), dtype=float),
        xyz_base=np.array(td.get("xyz_base", []), dtype=float),
        q_deg=q_deg,
        q_rad=np.deg2rad(q_deg),
        error_mm=float(td.get("error_mm", 0.0)),
        solver_type=td.get("solver_type", "object_pick_ik"),
        is_fallback=bool(td.get("is_fallback", False)),
        requested_yaw_deg=float(td["requested_yaw_deg"]) if td.get("requested_yaw_deg") is not None else None,
        approach_tilt_deg=float(td["approach_tilt_deg"]) if td.get("approach_tilt_deg") is not None else None,
        alignment_error_deg=float(td["alignment_error_deg"]) if td.get("alignment_error_deg") is not None else None,
        score=float(td.get("score", 0.0)),
        solution_count=int(td.get("solution_count", 1)),
        object_ik_result=None,
    )


def load_cache_store(cache_path: str) -> dict:
    """Read cache file safely, returning default structure on error or missing file."""
    if not os.path.exists(cache_path):
        return {"version": "2.0", "fingerprints": {}}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                return {"version": "2.0", "fingerprints": {}}
            if "fingerprints" not in data:
                data["fingerprints"] = {}
            return data
    except Exception:
        return {"version": "2.0", "fingerprints": {}}


def save_cache_store(cache_path: str, store: dict) -> bool:
    """Write cache store atomically to disk."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        tmp_path = cache_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(store, f, indent=2)
        os.replace(tmp_path, cache_path)
        return True
    except Exception:
        return False


def lookup_cached_target_ik(
    cache_path: str,
    config_hash: str,
    xyz: Union[List[float], Tuple[float, ...], np.ndarray],
    yaw_deg: float = 0.0,
) -> Optional[Tuple[Dict[str, Stage2_5TargetIKResult], bool]]:
    """Look up cached IK solutions for a target XYZ and yaw.
    
    Returns
    -------
    (reachable_dict, is_fallback) or None if not found.
    """
    store = load_cache_store(cache_path)
    fp_data = store.get("fingerprints", {}).get(config_hash, {})
    targets = fp_data.get("targets", {}) if isinstance(fp_data, dict) else {}
    key = target_cache_key(xyz, yaw_deg)

    if key in targets:
        try:
            entry = targets[key]
            is_fallback = bool(entry.get("is_fallback", False))
            sols = {}
            for rid, sdict in entry.get("solutions", {}).items():
                sols[rid] = dict_to_target_result(sdict)
            if sols:
                return sols, is_fallback
        except Exception:
            return None
    return None


def store_cached_target_ik(
    cache_path: str,
    config_hash: str,
    xyz: Union[List[float], Tuple[float, ...], np.ndarray],
    yaw_deg: float,
    opts: Dict[str, Stage2_5TargetIKResult],
    is_fallback: bool,
) -> bool:
    """Store solved IK options for a target into disk cache."""
    store = load_cache_store(cache_path)
    if "fingerprints" not in store:
        store["fingerprints"] = {}
    if config_hash not in store["fingerprints"] or not isinstance(store["fingerprints"][config_hash], dict):
        store["fingerprints"][config_hash] = {"targets": {}}

    key = target_cache_key(xyz, yaw_deg)
    solutions_dict = {rid: target_result_to_dict(sol) for rid, sol in opts.items()}

    entry = {
        "xyz_world": [round(float(v), 2) for v in xyz],
        "yaw_deg": round(float(yaw_deg) % 360.0, 1),
        "is_fallback": bool(is_fallback),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "solutions": solutions_dict,
    }

    if "targets" not in store["fingerprints"][config_hash]:
        store["fingerprints"][config_hash]["targets"] = {}

    store["fingerprints"][config_hash]["targets"][key] = entry
    return save_cache_store(cache_path, store)


def clear_cache(
    cache_path: str,
    stage: Optional[str] = None,
    config_hash: Optional[str] = None,
) -> int:
    """Clear cached entries, optionally filtered by config_hash. Returns cleared count."""
    store = load_cache_store(cache_path)
    if not store.get("fingerprints"):
        return 0

    count = 0
    if config_hash is None:
        for fp, data in store["fingerprints"].items():
            if isinstance(data, dict):
                count += len(data.get("targets", {}))
        store["fingerprints"] = {}
    elif config_hash in store["fingerprints"]:
        fp_data = store["fingerprints"][config_hash]
        if isinstance(fp_data, dict):
            count = len(fp_data.get("targets", {}))
        store["fingerprints"].pop(config_hash, None)

    save_cache_store(cache_path, store)
    return count


def get_cache_stats(cache_path: str, config_hash: Optional[str] = None) -> dict:
    """Get cache summary statistics."""
    store = load_cache_store(cache_path)
    fps = store.get("fingerprints", {})
    total_targets = 0
    for fp, data in fps.items():
        if isinstance(data, dict):
            total_targets += len(data.get("targets", {}))
    active_targets = 0
    if config_hash and config_hash in fps and isinstance(fps[config_hash], dict):
        active_targets = len(fps[config_hash].get("targets", {}))

    return {
        "file_exists": os.path.exists(cache_path),
        "total_targets": total_targets,
        "active_config_targets": active_targets,
        "fingerprints_count": len(fps),
    }
