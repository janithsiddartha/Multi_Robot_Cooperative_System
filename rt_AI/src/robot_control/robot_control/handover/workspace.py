"""Position-workspace computation and common-workspace caching.

This module contains two workspace implementations:

1. ``WorkspaceSampler`` is the legacy Cartesian sampler retained for
   compatibility with the existing handover planner/static-map code.

2. ``PositionWorkspaceComputer`` computes the actual position workspace
   from each robot's DH/FK model and joint limits. No manually specified
   Cartesian workspace boundary is used for this calculation.

The computed workspace is position-only. No TCP orientation or handover
constraint is applied here.
"""

import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np

from robot_control.kinematics.fk_math import fk
from .geometry import euler_to_matrix


CACHE_SCHEMA_VERSION = 2
DEFAULT_CACHE_DIR = os.path.expanduser(
    "~/.ros/robot_control/workspace_cache"
)


@dataclass
class PoseSample:
    """A verified feasible pose for a specific robot."""

    position: np.ndarray
    rotation: np.ndarray
    robot_id: int
    joint_solution: np.ndarray
    position_error_mm: float = 0.0
    orientation_error_deg: float = 0.0
    manipulability: float = 0.0
    joint_limit_margin_deg: float = 0.0


class WorkspaceSampler:
    """Legacy Cartesian position/orientation sampler.

    This class is retained for compatibility with the current planner.
    It does NOT represent actual robot reachability.
    """

    def __init__(self, bounds, position_step, orientation_step_deg):
        self.x_min = bounds.get("x_min", -500.0)
        self.x_max = bounds.get("x_max", 500.0)
        self.y_min = bounds.get("y_min", -500.0)
        self.y_max = bounds.get("y_max", 500.0)
        self.z_min = bounds.get("z_min", 0.0)
        self.z_max = bounds.get("z_max", 500.0)
        self.position_step = float(position_step)
        self.orientation_step = np.deg2rad(orientation_step_deg)

    def get_position_candidates(self):
        """Yield configured world XYZ grid points."""
        for x in np.arange(
            self.x_min,
            self.x_max + self.position_step,
            self.position_step,
        ):
            for y in np.arange(
                self.y_min,
                self.y_max + self.position_step,
                self.position_step,
            ):
                for z in np.arange(
                    self.z_min,
                    self.z_max + self.position_step,
                    self.position_step,
                ):
                    yield np.array([x, y, z], dtype=float)

    def get_orientation_candidates(self):
        """Yield legacy phi angles around TCP Z."""
        for phi in np.arange(
            0.0,
            2.0 * np.pi,
            self.orientation_step,
        ):
            yield phi

    def get_positions(self):
        """Legacy alias."""
        return self.get_position_candidates()

    def get_orientations(self):
        """Legacy alias."""
        return self.get_orientation_candidates()


class PositionWorkspaceComputer:
    """Compute and cache actual robot position workspaces.

    The workspace is obtained directly from:

    - robot DH/FK model,
    - robot joint limits,
    - robot world-base transform.

    No manually specified XYZ workspace bounds are used.

    Each robot is sampled in joint space using a deterministic Halton
    sequence. The resulting TCP positions are transformed into the common
    world frame.

    All robots use one automatically derived world-frame voxel origin so
    that voxel intersections are physically meaningful.
    """

    def __init__(
        self,
        handover_config,
        robots,
        handover_yaml_path=None,
        robot_yaml_paths=None,
        cache_dir=None,
    ):
        self.config = handover_config or {}
        self.robots = robots
        self.handover_yaml_path = handover_yaml_path
        self.robot_yaml_paths = robot_yaml_paths or {}

        self.cache_dir = os.path.expanduser(
            cache_dir or DEFAULT_CACHE_DIR
        )
        os.makedirs(self.cache_dir, exist_ok=True)

        ws_cfg = self.config.get("workspace", {})
        sampling = ws_cfg.get("sampling", {})

        self.voxel_size_mm = float(
            sampling.get("voxel_size_mm", 50.0)
        )
        self.num_samples = int(
            sampling.get("workspace_samples", 50000)
        )

        if self.voxel_size_mm <= 0.0:
            raise ValueError(
                "workspace voxel_size_mm must be > 0"
            )

        if self.num_samples <= 0:
            raise ValueError(
                "workspace_samples must be > 0"
            )

        # Automatically derived after FK sampling.
        self.world_origin = None

        # Actual occupied point bounds.
        self.robot_point_bounds = {}
        self.common_bounds = {}

        self.robot_voxels = {}
        self.common_voxels = {}

        self.loaded_from_cache = False

        self.current_hash = self._make_cache_hash()

        self.cache_path = os.path.join(
            self.cache_dir,
            f"workspace_{self.current_hash}.npz",
        )

    # ------------------------------------------------------------
    # Configuration / cache identity
    # ------------------------------------------------------------

    @staticmethod
    def _jsonable(value):
        """Convert config objects/NumPy values into deterministic JSON."""
        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, (np.integer, np.floating)):
            return value.item()

        if isinstance(value, dict):
            return {
                str(k): PositionWorkspaceComputer._jsonable(v)
                for k, v in sorted(
                    value.items(),
                    key=lambda item: str(item[0]),
                )
            }

        if isinstance(value, (list, tuple)):
            return [
                PositionWorkspaceComputer._jsonable(v)
                for v in value
            ]

        if hasattr(value, "__dict__"):
            return PositionWorkspaceComputer._jsonable(
                value.__dict__
            )

        return value

    @staticmethod
    def _file_sha256(path):
        """Return SHA256 of a file."""
        if not path or not os.path.isfile(path):
            return None

        digest = hashlib.sha256()

        with open(path, "rb") as handle:
            for chunk in iter(
                lambda: handle.read(1024 * 1024),
                b"",
            ):
                digest.update(chunk)

        return digest.hexdigest()

    def _make_cache_hash(self):
        """Hash all configuration affecting the workspace."""
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,

            # Only configuration that is relevant to workspace
            # generation is retained here.
            "workspace_sampling": {
                "voxel_size_mm": self.voxel_size_mm,
                "workspace_samples": self.num_samples,
            },

            "robots": {
                str(rid): self._jsonable(cfg)
                for rid, cfg in sorted(
                    self.robots.items()
                )
            },

            "handover_config": self._jsonable(
                self.config.get("robots", {})
            ),

            "handover_yaml_sha256": self._file_sha256(
                self.handover_yaml_path
            ),

            "robot_yaml_sha256": {
                str(rid): self._file_sha256(
                    self.robot_yaml_paths.get(rid)
                )
                for rid in sorted(
                    self.robot_yaml_paths
                )
            },
        }

        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        return hashlib.sha256(encoded).hexdigest()

    # ------------------------------------------------------------
    # World/base transforms
    # ------------------------------------------------------------

    def get_world_to_base(self, robot_id):
        """Return the base-to-world homogeneous transform.

        The historical function name is retained for compatibility.
        The returned matrix maps base-frame points into world frame.
        """
        info = self.config.get("robots", {}).get(
            f"robot{robot_id}",
            {},
        )

        xyz = np.asarray(
            info.get(
                "world_base_xyz",
                [0.0, 0.0, 0.0],
            ),
            dtype=float,
        )

        rpy = np.deg2rad(
            info.get(
                "world_base_rpy_deg",
                [0.0, 0.0, 0.0],
            )
        )

        transform = np.eye(4)

        transform[:3, :3] = euler_to_matrix(rpy)
        transform[:3, 3] = xyz

        return transform

    # ------------------------------------------------------------
    # Deterministic joint-space sampling
    # ------------------------------------------------------------

    @staticmethod
    def _first_primes(count):
        """Return the first ``count`` prime numbers."""
        primes = []
        candidate = 2

        while len(primes) < count:
            is_prime = True

            for prime in primes:
                if prime * prime > candidate:
                    break

                if candidate % prime == 0:
                    is_prime = False
                    break

            if is_prime:
                primes.append(candidate)

            candidate += 1

        return primes

    @staticmethod
    def _halton(index, base):
        """Return the Van der Corput value for a 1-based index."""
        result = 0.0
        fraction = 1.0 / base
        value = index

        while value:
            value, remainder = divmod(
                value,
                base,
            )

            result += remainder * fraction
            fraction /= base

        return result

    def _joint_samples(self, cfg, robot_id):
        """Generate deterministic joint samples within limits."""
        del robot_id

        n_joints = cfg.num_joints

        bases = self._first_primes(n_joints)

        q_min = np.asarray(
            cfg.joint_limits_rad[:, 0],
            dtype=float,
        )

        q_max = np.asarray(
            cfg.joint_limits_rad[:, 1],
            dtype=float,
        )

        qs = np.empty(
            (
                self.num_samples,
                n_joints,
            ),
            dtype=float,
        )

        for k in range(self.num_samples):
            index = k + 1

            for joint, base in enumerate(bases):
                u = self._halton(
                    index,
                    base,
                )

                qs[k, joint] = (
                    q_min[joint]
                    + (
                        q_max[joint]
                        - q_min[joint]
                    )
                    * u
                )

        return qs

    # ------------------------------------------------------------
    # Robot workspace point generation
    # ------------------------------------------------------------

    def _get_position_points(self, robot_id):
        """Generate world-frame TCP positions for one robot."""
        cfg = self.robots[robot_id]

        transform_world_base = self.get_world_to_base(
            robot_id
        )

        qs = self._joint_samples(
            cfg,
            robot_id,
        )

        points_world = np.empty(
            (
                self.num_samples,
                3,
            ),
            dtype=float,
        )

        for index, q in enumerate(qs):
            point_base, _ = fk(
                q,
                cfg,
            )

            point_h = np.append(
                point_base,
                1.0,
            )

            points_world[index] = (
                transform_world_base @ point_h
            )[:3]

        return points_world

    # ------------------------------------------------------------
    # Automatic voxel origin
    # ------------------------------------------------------------

    def _derive_world_origin(self, all_points):
        """Derive one common voxel origin from all robot points."""
        minimum = np.min(
            all_points,
            axis=0,
        )

        return (
            np.floor(
                minimum / self.voxel_size_mm
            )
            * self.voxel_size_mm
        )

    def _world_to_voxel(self, points_world):
        """Convert world XYZ points into common voxel indices."""
        if self.world_origin is None:
            raise RuntimeError(
                "world_origin has not been initialized"
            )

        return np.floor(
            (
                points_world
                - self.world_origin
            )
            / self.voxel_size_mm
        ).astype(np.int32)

    def _voxel_bounds(self, voxels):
        """Return world XYZ bounds of occupied voxel cubes."""
        if len(voxels) == 0:
            return None

        low_index = np.min(
            voxels,
            axis=0,
        )

        high_index = np.max(
            voxels,
            axis=0,
        )

        low = (
            self.world_origin
            + low_index * self.voxel_size_mm
        )

        high = (
            self.world_origin
            + (high_index + 1)
            * self.voxel_size_mm
        )

        return {
            "x": [
                float(low[0]),
                float(high[0]),
            ],
            "y": [
                float(low[1]),
                float(high[1]),
            ],
            "z": [
                float(low[2]),
                float(high[2]),
            ],
        }

    # ------------------------------------------------------------
    # Full calculation
    # ------------------------------------------------------------

    def compute(self):
        """Compute actual robot workspaces and intersections."""
        self.robot_voxels = {}
        self.common_voxels = {}
        self.robot_point_bounds = {}
        self.common_bounds = {}

        all_points = {}
        combined_points = []

        # --------------------------------------------------------
        # 1. Generate actual FK workspaces
        # --------------------------------------------------------

        for robot_id in sorted(self.robots):
            points = self._get_position_points(
                robot_id
            )

            all_points[robot_id] = points
            combined_points.append(points)

            self.robot_point_bounds[robot_id] = {
                "x": [
                    float(np.min(points[:, 0])),
                    float(np.max(points[:, 0])),
                ],
                "y": [
                    float(np.min(points[:, 1])),
                    float(np.max(points[:, 1])),
                ],
                "z": [
                    float(np.min(points[:, 2])),
                    float(np.max(points[:, 2])),
                ],
            }

        # --------------------------------------------------------
        # 2. Derive one common world voxel origin
        # --------------------------------------------------------

        combined = np.vstack(
            combined_points
        )

        self.world_origin = self._derive_world_origin(
            combined
        )

        # --------------------------------------------------------
        # 3. Voxelize each robot
        # --------------------------------------------------------

        for robot_id, points in all_points.items():
            self.robot_voxels[robot_id] = np.unique(
                self._world_to_voxel(points),
                axis=0,
            )

        # --------------------------------------------------------
        # 4. Compute intersections
        # --------------------------------------------------------

        voxel_sets = {
            robot_id: set(
                map(
                    tuple,
                    voxels,
                )
            )
            for robot_id, voxels
            in self.robot_voxels.items()
        }

        pairs = [
            (1, 2),
            (2, 3),
            (3, 1),
        ]

        for robot_a, robot_b in pairs:
            common = (
                voxel_sets[robot_a]
                & voxel_sets[robot_b]
            )

            array = (
                np.asarray(
                    sorted(common),
                    dtype=np.int32,
                )
                if common
                else np.empty(
                    (0, 3),
                    dtype=np.int32,
                )
            )

            self.common_voxels[
                (robot_a, robot_b)
            ] = array

            self.common_bounds[
                (robot_a, robot_b)
            ] = self._voxel_bounds(
                array
            )

        triple = (
            voxel_sets[1]
            & voxel_sets[2]
            & voxel_sets[3]
        )

        triple_array = (
            np.asarray(
                sorted(triple),
                dtype=np.int32,
            )
            if triple
            else np.empty(
                (0, 3),
                dtype=np.int32,
            )
        )

        self.common_voxels[
            (1, 2, 3)
        ] = triple_array

        self.common_bounds[
            (1, 2, 3)
        ] = self._voxel_bounds(
            triple_array
        )

        self.loaded_from_cache = False

        return self

    # ------------------------------------------------------------
    # Voxel access
    # ------------------------------------------------------------

    def voxel_centers(self, voxels):
        """Convert voxel indices to world XYZ centers."""
        voxels = np.asarray(
            voxels,
            dtype=np.int32,
        )

        if voxels.size == 0:
            return np.empty(
                (0, 3),
                dtype=float,
            )

        if self.world_origin is None:
            raise RuntimeError(
                "world_origin has not been initialized"
            )

        return (
            self.world_origin
            + (
                voxels.astype(float)
                + 0.5
            )
            * self.voxel_size_mm
        )

    def get_robot_voxel_centers(self, robot_id):
        """Return occupied voxel centers for one robot."""
        return self.voxel_centers(
            self.robot_voxels[robot_id]
        )

    def get_common_voxel_centers(self, *robot_ids):
        """Return voxel centers for a common workspace."""
        key = tuple(robot_ids)

        if key not in self.common_voxels:
            raise KeyError(
                f"No common workspace stored for {key}"
            )

        return self.voxel_centers(
            self.common_voxels[key]
        )

    # ------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------

    def save(self):
        """Save workspace, intersections, and report metadata."""
        if not self.robot_voxels:
            raise RuntimeError(
                "compute() must run before save()"
            )

        np.savez_compressed(
            self.cache_path,
            world_origin=self.world_origin,

            robot1=self.robot_voxels[1],
            robot2=self.robot_voxels[2],
            robot3=self.robot_voxels[3],

            common12=self.common_voxels[(1, 2)],
            common23=self.common_voxels[(2, 3)],
            common31=self.common_voxels[(3, 1)],
            common123=self.common_voxels[(1, 2, 3)],

            # Preserve the actual FK point bounds so that a
            # cache-loaded report is identical to a freshly
            # computed report.
            robot1_point_bounds=np.asarray(
                [
                    self.robot_point_bounds[1]["x"],
                    self.robot_point_bounds[1]["y"],
                    self.robot_point_bounds[1]["z"],
                ],
                dtype=float,
            ),
            robot2_point_bounds=np.asarray(
                [
                    self.robot_point_bounds[2]["x"],
                    self.robot_point_bounds[2]["y"],
                    self.robot_point_bounds[2]["z"],
                ],
                dtype=float,
            ),
            robot3_point_bounds=np.asarray(
                [
                    self.robot_point_bounds[3]["x"],
                    self.robot_point_bounds[3]["y"],
                    self.robot_point_bounds[3]["z"],
                ],
                dtype=float,
            ),
        )

    def load(self):
        """Load the workspace cache."""
        if not os.path.isfile(self.cache_path):
            return False

        with np.load(
            self.cache_path,
            allow_pickle=False,
        ) as data:
            self.world_origin = np.asarray(
                data["world_origin"],
                dtype=float,
            )

            self.robot_voxels = {
                1: np.asarray(
                    data["robot1"],
                    dtype=np.int32,
                ),
                2: np.asarray(
                    data["robot2"],
                    dtype=np.int32,
                ),
                3: np.asarray(
                    data["robot3"],
                    dtype=np.int32,
                ),
            }

            self.common_voxels = {
                (1, 2): np.asarray(
                    data["common12"],
                    dtype=np.int32,
                ),
                (2, 3): np.asarray(
                    data["common23"],
                    dtype=np.int32,
                ),
                (3, 1): np.asarray(
                    data["common31"],
                    dtype=np.int32,
                ),
                (1, 2, 3): np.asarray(
                    data["common123"],
                    dtype=np.int32,
                ),
            }

            # Restore actual FK point bounds.
            self.robot_point_bounds = {}

            for robot_id in (1, 2, 3):
                bounds = np.asarray(
                    data[
                        f"robot{robot_id}_point_bounds"
                    ],
                    dtype=float,
                )

                self.robot_point_bounds[robot_id] = {
                    "x": [
                        float(bounds[0, 0]),
                        float(bounds[0, 1]),
                    ],
                    "y": [
                        float(bounds[1, 0]),
                        float(bounds[1, 1]),
                    ],
                    "z": [
                        float(bounds[2, 0]),
                        float(bounds[2, 1]),
                    ],
                }

        self.common_bounds = {
            key: self._voxel_bounds(value)
            for key, value in self.common_voxels.items()
        }

        self.loaded_from_cache = True

        return True

    def load_or_compute(
        self,
        force_rebuild=False,
    ):
        """Load matching cache or compute a new workspace."""
        if (
            not force_rebuild
            and self.load()
        ):
            return self

        self.compute()
        self.save()

        return self

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------

    @staticmethod
    def _format_bounds(bounds):
        """Format XYZ bounds for the terminal."""
        if bounds is None:
            return "none"

        return (
            f"X: {bounds['x'][0]:.1f} .. "
            f"{bounds['x'][1]:.1f} mm, "
            f"Y: {bounds['y'][0]:.1f} .. "
            f"{bounds['y'][1]:.1f} mm, "
            f"Z: {bounds['z'][0]:.1f} .. "
            f"{bounds['z'][1]:.1f} mm"
        )

    def report(self):
        """Return workspace/intersection summary."""
        lines = [
            "POSITION WORKSPACE",
            "------------------",
            (
                f"R1: "
                f"{len(self.robot_voxels.get(1, []))} "
                f"occupied voxels"
            ),
            (
                f"R2: "
                f"{len(self.robot_voxels.get(2, []))} "
                f"occupied voxels"
            ),
            (
                f"R3: "
                f"{len(self.robot_voxels.get(3, []))} "
                f"occupied voxels"
            ),
            "",
            "ROBOT WORKSPACE BOUNDS",
            "-----------------------",
        ]

        for robot_id in (1, 2, 3):
            bounds_text = self._format_bounds(
                self.robot_point_bounds.get(robot_id)
            )

            lines.append(
                f"R{robot_id}: {bounds_text}"
            )

        lines.extend(
            [
                "",
                "COMMON WORKSPACE",
                "----------------",
            ]
        )

        for key, label in [
            ((1, 2), "R1 ∩ R2"),
            ((2, 3), "R2 ∩ R3"),
            ((3, 1), "R3 ∩ R1"),
            ((1, 2, 3), "R1 ∩ R2 ∩ R3"),
        ]:
            voxels = self.common_voxels.get(
                key,
                np.empty(
                    (0, 3),
                    dtype=np.int32,
                ),
            )

            lines.append(
                f"{label}: {len(voxels)} voxels"
            )

            lines.append(
                "    XYZ bounds: "
                + self._format_bounds(
                    self.common_bounds.get(key)
                )
            )

        lines.extend(
            [
                "",
                f"Voxel size: "
                f"{self.voxel_size_mm:.1f} mm",
                f"Joint samples / robot: "
                f"{self.num_samples}",
                (
                    "Automatically derived world origin: "
                    f"[{self.world_origin[0]:.1f}, "
                    f"{self.world_origin[1]:.1f}, "
                    f"{self.world_origin[2]:.1f}] mm"
                    if self.world_origin is not None
                    else "Automatically derived world origin: none"
                ),
                f"Cache: {self.cache_path}",
                f"Cache hash: {self.current_hash}",
                f"Loaded from cache: "
                f"{self.loaded_from_cache}",
            ]
        )

        return "\n".join(lines)