import math

import numpy as np
import numpy.typing as npt
import scipy.spatial.transform


def rotmat_to_euler(rot_mat: npt.NDArray[np.float32], degrees=False) -> npt.NDArray[np.float32]:
    return scipy.spatial.transform.Rotation.from_matrix(rot_mat).as_euler("xyz", degrees=degrees)


def euler_to_rotmat(euler: npt.NDArray[np.float32], degrees=False) -> npt.NDArray[np.float32]:
    return scipy.spatial.transform.Rotation.from_euler("xyz", euler, degrees=degrees).as_matrix()


def quat_to_euler(quat: np.ndarray, degrees: bool = False) -> np.ndarray:
    euler = scipy.spatial.transform.Rotation.from_quat(quat).as_euler("xyz", degrees=degrees)
    return euler


def euler_to_quat(euler: np.ndarray, degrees: bool = False) -> np.ndarray:
    return scipy.spatial.transform.Rotation.from_euler("xyz", euler, degrees=degrees).as_quat()


def make_homogenous_matrix(
    rot_mat: npt.NDArray[np.float32], translation: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    return np.block(
        [
            [rot_mat, translation.reshape(3, 1)],
            [np.zeros((1, 3)), 1],
        ]
    )


def quat_wxyz_to_rotmat(wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(wxyz, dtype=np.float64) / np.linalg.norm(wxyz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotmat_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    # Shepperd's branch, not scipy: it agrees on the rotation to 1e-16 but flips the quaternion
    # SIGN on 7% of matrices, and callers hand this value on as a pose.
    m = np.asarray(matrix, dtype=np.float64)
    trace = np.trace(m)
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    return q / np.linalg.norm(q)
