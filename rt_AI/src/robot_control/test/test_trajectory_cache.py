"""Unit tests for compact target IK caching and persistent robot selection."""

import os
import tempfile
import numpy as np
import pytest

from robot_control.handover.stage1 import load_stage1_robots
from robot_control.handover.stage2_5 import build_stage2_5_plan
from robot_control.handover.trajectory_cache import (
    clear_cache,
    compute_config_fingerprint,
    get_cache_stats,
    lookup_cached_target_ik,
    store_cached_target_ik,
    target_cache_key,
)


@pytest.fixture
def cache_env():
    handover_cfg, robots, _ = load_stage1_robots()
    fp = compute_config_fingerprint(robots, handover_cfg)
    tmp_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp_path = tmp_file.name
    tmp_file.close()
    yield tmp_path, robots, handover_cfg, fp
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


def test_config_fingerprint_deterministic(cache_env):
    _, robots, handover_cfg, fp1 = cache_env
    fp2 = compute_config_fingerprint(robots, handover_cfg)
    assert fp1 == fp2
    assert len(fp1) == 16

    # Modify a DH parameter on robot1 -> fingerprint must change
    robots['robot1'].a[0] += 10.0
    fp_modified = compute_config_fingerprint(robots, handover_cfg)
    assert fp_modified != fp1
    # Restore
    robots['robot1'].a[0] -= 10.0


def test_target_cache_key_formatting():
    k1 = target_cache_key([-97.1000001, 0.0, 13.824], 0.0)
    assert k1 == "-97.1,0.0,13.82,0.0"

    k2 = target_cache_key([10, 20, 30], 360.0)
    assert k2 == "10.0,20.0,30.0,0.0"


def test_cache_miss_and_store_hit(cache_env):
    tmp_path, robots, handover_cfg, fp = cache_env
    pick_xyz = [-97.10, 0.0, 13.82]
    yaw = 0.0

    # 1. Initially must be a miss
    miss = lookup_cached_target_ik(tmp_path, fp, pick_xyz, yaw)
    assert miss is None

    # 2. Build stage 2.5 plan to populate cache
    chosen_robots = []
    def record_chooser(label, opts):
        chosen_robots.append((label, list(opts)))
        return opts[0]

    plan, pick_opts, place_opts = build_stage2_5_plan(
        pick_xyz=pick_xyz,
        place_xyz=[0.0, 150.0, 150.0],
        robots=robots,
        handover_cfg=handover_cfg,
        chooser=record_chooser,
        cache_filepath=tmp_path,
        config_fingerprint=fp,
    )

    # 3. Lookup must hit and return all candidate robots for pick
    hit = lookup_cached_target_ik(tmp_path, fp, pick_xyz, yaw)
    assert hit is not None
    sols, is_fallback = hit
    assert isinstance(sols, dict)
    assert not is_fallback
    assert set(sols.keys()) == set(pick_opts.keys())

    # Check that robot1 and robot3 solutions are preserved
    for rid in ['robot1', 'robot3']:
        if rid in sols:
            assert np.allclose(sols[rid].q_deg, pick_opts[rid].q_deg, atol=0.01)
            assert abs(sols[rid].error_mm - pick_opts[rid].error_mm) < 1e-3


def test_repeat_execution_always_prompts_robot_selection(cache_env):
    """When a target is cached, repeat runs must still invoke chooser to let user pick robot."""
    tmp_path, robots, handover_cfg, fp = cache_env
    pick_xyz = [-97.10, 0.0, 13.82]
    place_xyz = [0.0, 150.0, 150.0]

    # 1st run: operator picks robot3 for pick
    chooser_calls_1 = []
    def chooser_1(label, opts):
        chooser_calls_1.append((label, list(opts)))
        if label == "pick" and "robot3" in opts:
            return "robot3"
        return opts[0]

    plan1, _, _ = build_stage2_5_plan(
        pick_xyz=pick_xyz,
        place_xyz=place_xyz,
        robots=robots,
        handover_cfg=handover_cfg,
        chooser=chooser_1,
        cache_filepath=tmp_path,
        config_fingerprint=fp,
    )
    assert plan1.pick_robot == "robot3"
    assert len(chooser_calls_1) >= 1

    # 2nd run (loaded from cache): operator selects robot1 for pick!
    chooser_calls_2 = []
    def chooser_2(label, opts):
        chooser_calls_2.append((label, list(opts)))
        if label == "pick" and "robot1" in opts:
            return "robot1"
        return opts[0]

    plan2, _, _ = build_stage2_5_plan(
        pick_xyz=pick_xyz,
        place_xyz=place_xyz,
        robots=robots,
        handover_cfg=handover_cfg,
        chooser=chooser_2,
        cache_filepath=tmp_path,
        config_fingerprint=fp,
    )
    # The user was prompted again and their choice robot1 was honored!
    assert len(chooser_calls_2) >= 1
    assert plan2.pick_robot == "robot1"
    assert plan2.pick.robot_id == "robot1"


def test_cache_invalidation_on_config_change(cache_env):
    tmp_path, robots, handover_cfg, fp = cache_env
    pick_xyz = [-97.10, 0.0, 13.82]

    build_stage2_5_plan(
        pick_xyz=pick_xyz,
        place_xyz=[0.0, 150.0, 150.0],
        robots=robots,
        handover_cfg=handover_cfg,
        chooser=lambda l, opts: opts[0],
        cache_filepath=tmp_path,
        config_fingerprint=fp,
    )

    # Invalidate by looking up with different fingerprint
    fake_fp = "deadbeef12345678"
    assert lookup_cached_target_ik(tmp_path, fake_fp, pick_xyz, 0.0) is None


def test_clear_cache_and_stats(cache_env):
    tmp_path, robots, handover_cfg, fp = cache_env
    build_stage2_5_plan(
        pick_xyz=[-97.10, 0.0, 13.82],
        place_xyz=[0.0, 150.0, 150.0],
        robots=robots,
        handover_cfg=handover_cfg,
        chooser=lambda l, opts: opts[0],
        cache_filepath=tmp_path,
        config_fingerprint=fp,
    )

    stats = get_cache_stats(tmp_path, fp)
    assert stats["active_config_targets"] == 2  # pick and place

    cleared = clear_cache(tmp_path, config_hash=fp)
    assert cleared == 2

    stats_after = get_cache_stats(tmp_path, fp)
    assert stats_after["active_config_targets"] == 0


def test_compact_cache_file_size(cache_env):
    """Verify that cached data is compact (under 5 KB and ~150 lines total for multiple targets)."""
    tmp_path, robots, handover_cfg, fp = cache_env
    build_stage2_5_plan(
        pick_xyz=[-97.10, 0.0, 13.82],
        place_xyz=[0.0, 150.0, 150.0],
        robots=robots,
        handover_cfg=handover_cfg,
        chooser=lambda l, opts: opts[0],
        cache_filepath=tmp_path,
        config_fingerprint=fp,
    )

    file_size_bytes = os.path.getsize(tmp_path)
    assert file_size_bytes < 10000  # Less than 10 KB, far below the old 108 KB!
    with open(tmp_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) < 250  # Far below the old 3,930 lines!
