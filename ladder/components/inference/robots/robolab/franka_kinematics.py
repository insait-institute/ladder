from pathlib import Path

import numpy as np
import pytorch_kinematics as pk
import torch
from pytorch_kinematics.ik import delta_pose
from pytorch_kinematics.transforms import rotation_conversions

from ladder.core.common.platform import REPO_ROOT

# panda_link8 (flange) is ladder's canonical DROID EE frame; pytorch_kinematics keeps this torch/numpy-2 native.
_PANDA_URDF = Path(REPO_ROOT) / "ladder/components/urdf/robots/panda/robot.urdf"
_EEF_LINK = "panda_link8"

# Franka Panda joint position limits (rad).
_JOINT_LIMITS_LOWER = [-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159]
_JOINT_LIMITS_UPPER = [2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159]


class FrankaKinematics:
    def __init__(self) -> None:
        with open(_PANDA_URDF) as urdf_file:
            self._chain = pk.build_serial_chain_from_urdf(urdf_file.read(), _EEF_LINK).to(dtype=torch.float32)
        self._joint_limits = torch.tensor([_JOINT_LIMITS_LOWER, _JOINT_LIMITS_UPPER], dtype=torch.float32).T
        self._solver = pk.PseudoInverseIK(
            self._chain,
            retry_configs=torch.zeros(1, 7),
            joint_limits=self._joint_limits,
            max_iterations=100,
            pos_tolerance=1e-4,
            rot_tolerance=1e-3,
            early_stopping_any_converged=True,
        )

    def forward(self, joint_position: np.ndarray) -> np.ndarray:
        joints = torch.as_tensor(np.array(joint_position, dtype=np.float32)).reshape(1, -1)
        pose = self._chain.forward_kinematics(joints, end_only=True).get_matrix()[0].numpy()
        # A diverged sim step can report non-finite joints; a neutral pose keeps the rotation orthonormal
        # so one diverged env doesn't crash the whole batched collate.
        return pose if np.isfinite(pose).all() else np.eye(4, dtype=np.float32)

    def inverse(self, target_poses: np.ndarray, seed_joint_positions: np.ndarray) -> np.ndarray:
        q = torch.as_tensor(np.array(seed_joint_positions, dtype=np.float32))
        target = torch.as_tensor(np.array(target_poses, dtype=np.float32)).reshape(-1, 4, 4)
        target_pos = target[:, :3, 3]
        target_wxyz = rotation_conversions.matrix_to_quaternion(target[:, :3, :3])
        remaining = torch.ones(len(q), dtype=torch.bool)
        err_min, no_improve = None, None
        with torch.no_grad():
            for _ in range(self._solver.max_iterations):
                if not remaining.any():
                    break
                jacobian, pose = self._chain.jacobian(q, ret_eef_pose=True)
                dx, pos_diff, rot_diff = delta_pose(pose.view(-1, 1, 4, 4), target_pos, target_wxyz)
                dq = self._solver.compute_dq(jacobian, dx).squeeze(2)
                q = torch.where(remaining.unsqueeze(1), q + self._solver.lr * dq, q)
                converged = (pos_diff.squeeze(2).norm(dim=-1) < self._solver.pos_tolerance) & (
                    rot_diff.squeeze(2).norm(dim=-1) < self._solver.rot_tolerance
                )
                remaining &= ~converged
                err = dx.squeeze(2).norm(dim=-1)
                if no_improve is None:
                    no_improve = torch.zeros_like(err)
                elif err_min is None:
                    err_min = err.clone()
                else:
                    improved = err < err_min
                    err_min[improved] = err[improved]
                    no_improve[improved] = 0
                    no_improve[~improved] += 1
                    remaining &= no_improve <= self._solver.early_stopping_no_improvement_patience
        return q.numpy()
