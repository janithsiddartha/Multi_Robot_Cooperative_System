import numpy as np


class CollisionChecker:
    """Check for collisions between two robot trajectories.

    Collision types distinguished:
    ─────────────────────────────
    1. **Robot-vs-robot links**: checked by pairwise distance
       between all non-TCP frame positions.
    2. **Robot-vs-TCP/gripper**: NOT checked — grippers are
       expected to meet at the handover point.
    3. **Handover contact**: The final frame (TCP) of each robot
       is explicitly excluded from collision checks because
       the grippers are designed to meet during handover.

    The check is performed over the COMPLETE synchronized motion,
    not only the endpoint.  If robots move at different durations,
    the caller must synchronize the trajectory time before passing
    frame data here.
    """

    def __init__(self, safe_distance):
        self.safe_distance = safe_distance

    def check_trajectory_collision(self, trajectory_1, trajectory_2):
        """Check for collisions along two synchronized trajectories.

        Parameters
        ----------
        trajectory_1, trajectory_2 :
            Lists of frame-position arrays, one per time step.
            Each element is shape (num_frames+1, 3).
            The last point in each is the TCP — excluded from checks.

        Returns
        -------
        bool : True if safe (no collision), False if collision detected.
        """
        n_steps = min(len(trajectory_1), len(trajectory_2))
        for t_idx in range(n_steps):
            # Exclude TCP (last point) — grippers are meant to meet
            points_1 = trajectory_1[t_idx][:-1]
            points_2 = trajectory_2[t_idx][:-1]

            for p1 in points_1:
                for p2 in points_2:
                    dist = np.linalg.norm(p1 - p2)
                    if dist < self.safe_distance:
                        return False
        return True
