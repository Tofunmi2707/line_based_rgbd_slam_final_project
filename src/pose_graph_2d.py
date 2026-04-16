from __future__ import annotations

"""
Lightweight 2D pose-graph optimisation utilities for the loop-closure experiment.

This module implements a simple planar pose-graph backend used to evaluate the
effect of loop constraints on odometry consistency. Poses are represented in a
2D SE(2)-style form with translation and yaw, and graph optimisation is carried
out by iterative weighted least squares.

Inspiration:
- The formulation follows standard graph-SLAM practice, in which odometry and
  loop constraints are represented as relative pose edges and optimised by
  minimising edge residuals.
- This implementation is a project-specific lightweight backend written for
  experimental evaluation and residual analysis, rather than a full external
  optimisation framework such as g2o.

Notes:
- The optimiser operates on a planar approximation.
- Jacobians are computed numerically.
- The first pose is strongly fixed to remove gauge freedom.
"""

import math
import numpy as np


def wrap(a: float | np.ndarray) -> float | np.ndarray:
    """
    Wrap an angle or array of angles to the interval [-pi, pi).

    Args:
        a: Input angle in radians.

    Returns:
        Wrapped angle in radians.
    """
    return (a + np.pi) % (2 * np.pi) - np.pi


def v2t(p: np.ndarray) -> np.ndarray:
    """
    Convert a planar pose vector to a homogeneous transform.

    Args:
        p: Pose vector [x, y, theta].

    Returns:
        3 x 3 homogeneous transform matrix.
    """
    x, y, th = p
    c, s = np.cos(th), np.sin(th)
    T = np.array(
        [
            [c, -s, x],
            [s,  c, y],
            [0,  0, 1],
        ],
        dtype=float,
    )
    return T


def t2v(T: np.ndarray) -> np.ndarray:
    """
    Convert a homogeneous planar transform to a pose vector.

    Args:
        T: 3 x 3 homogeneous transform matrix.

    Returns:
        Pose vector [x, y, theta].
    """
    x, y = T[0, 2], T[1, 2]
    th = math.atan2(T[1, 0], T[0, 0])
    return np.array([x, y, th], dtype=float)


def invT(T: np.ndarray) -> np.ndarray:
    """
    Invert a planar homogeneous transform.

    Args:
        T: 3 x 3 homogeneous transform matrix.

    Returns:
        Inverse transform.
    """
    R = T[:2, :2]
    t = T[:2, 2]
    Ti = np.eye(3)
    Ti[:2, :2] = R.T
    Ti[:2, 2] = -R.T @ t
    return Ti


def between(xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
    """
    Compute the relative planar pose from xi to xj.

    Args:
        xi: Pose vector of node i.
        xj: Pose vector of node j.

    Returns:
        Relative pose vector from i to j.
    """
    Ti = invT(v2t(xi))
    Tij = Ti @ v2t(xj)
    return t2v(Tij)


def edge_error(xi: np.ndarray, xj: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Compute the residual of a relative pose constraint.

    Args:
        xi: Pose vector of node i.
        xj: Pose vector of node j.
        z: Measured relative pose constraint.

    Returns:
        Residual vector in planar pose coordinates.
    """
    Z = v2t(z)
    Tij = v2t(between(xi, xj))
    E = invT(Z) @ Tij
    e = t2v(E)
    e[2] = wrap(e[2])
    return e


def edge_residual_norm(xi: np.ndarray, xj: np.ndarray, z: np.ndarray) -> float:
    """
    Compute the Euclidean norm of an edge residual.

    Args:
        xi: Pose vector of node i.
        xj: Pose vector of node j.
        z: Measured relative pose constraint.

    Returns:
        Residual norm.
    """
    e = edge_error(xi, xj, z)
    return float(np.linalg.norm(e))


def num_jacobian(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Compute a numerical Jacobian using forward finite differences.

    Args:
        f: Function mapping a pose vector to a 3-vector residual.
        x: Evaluation point.
        eps: Finite-difference step size.

    Returns:
        Numerical Jacobian matrix.
    """
    J = np.zeros((3, len(x)))
    fx = f(x)
    for k in range(len(x)):
        xp = x.copy()
        xp[k] += eps
        fp = f(xp)
        J[:, k] = (fp - fx) / eps
    return J


def compute_edge_residuals(poses: np.ndarray, edges: list[dict]) -> dict:
    """
    Compute residual norms for odometry and loop edges separately.

    Args:
        poses: Array of planar poses.
        edges: List of edge dictionaries with keys including i, j, z, and type.

    Returns:
        Dictionary containing odometry residuals, loop residuals, and edge counts.
    """
    odom_residuals = []
    loop_residuals = []

    for e in edges:
        i, j, z = e["i"], e["j"], e["z"]
        r = edge_residual_norm(poses[i], poses[j], z)

        if e["type"] == "loop":
            loop_residuals.append(r)
        else:
            odom_residuals.append(r)

    return {
        "odom_residuals": np.asarray(odom_residuals, dtype=float),
        "loop_residuals": np.asarray(loop_residuals, dtype=float),
        "num_odom_edges": int(sum(1 for e in edges if e["type"] != "loop")),
        "num_loop_edges": int(sum(1 for e in edges if e["type"] == "loop")),
    }


def optimise_pose_graph(
    poses: np.ndarray,
    edges: list[dict],
    iters: int = 10,
    w_odo: float = 1.0,
    w_loop: float = 3.0,
) -> np.ndarray:
    """
    Optimise a planar pose graph using iterative weighted least squares.

    The first pose is strongly anchored to remove gauge freedom.

    Args:
        poses: Initial planar poses.
        edges: Relative pose constraints.
        iters: Maximum number of optimisation iterations.
        w_odo: Weight applied to odometry edges.
        w_loop: Weight applied to loop edges.

    Returns:
        Optimised planar poses.
    """
    x = poses.copy()
    N = len(x)

    for _ in range(iters):
        H = np.zeros((3 * N, 3 * N))
        b = np.zeros(3 * N)

        for e in edges:
            i, j, z = e["i"], e["j"], e["z"]
            w = w_loop if e["type"] == "loop" else w_odo

            xi = x[i].copy()
            xj = x[j].copy()

            err = edge_error(xi, xj, z)
            Ji = num_jacobian(lambda p: edge_error(p, xj, z), xi)
            Jj = num_jacobian(lambda p: edge_error(xi, p, z), xj)

            Ji *= np.sqrt(w)
            Jj *= np.sqrt(w)
            err *= np.sqrt(w)

            ii = slice(3 * i, 3 * i + 3)
            jj = slice(3 * j, 3 * j + 3)

            H[ii, ii] += Ji.T @ Ji
            H[ii, jj] += Ji.T @ Jj
            H[jj, ii] += Jj.T @ Ji
            H[jj, jj] += Jj.T @ Jj

            b[ii] += Ji.T @ err
            b[jj] += Jj.T @ err

        H[0:3, 0:3] += 1e6 * np.eye(3)
        H += 1e-6 * np.eye(3 * N)

        try:
            dx = np.linalg.solve(H, -b).reshape(N, 3)
        except np.linalg.LinAlgError:
            dx = np.linalg.lstsq(H, -b, rcond=None)[0].reshape(N, 3)

        x += dx
        x[:, 2] = np.vectorize(wrap)(x[:, 2])

        if np.max(np.abs(dx)) < 1e-5:
            break

    return x


def optimise_pose_graph_with_metrics(
    poses: np.ndarray,
    edges: list[dict],
    iters: int = 10,
    w_odo: float = 1.0,
    w_loop: float = 3.0,
) -> dict:
    """
    Optimise a pose graph and return before/after residual metrics.

    Args:
        poses: Initial planar poses.
        edges: Relative pose constraints.
        iters: Maximum number of optimisation iterations.
        w_odo: Weight applied to odometry edges.
        w_loop: Weight applied to loop edges.

    Returns:
        Dictionary containing poses before optimisation, poses after optimisation,
        residual arrays before and after optimisation, and edge counts.
    """
    poses_before = poses.copy()

    before = compute_edge_residuals(poses_before, edges)
    poses_after = optimise_pose_graph(
        poses_before.copy(),
        edges,
        iters=iters,
        w_odo=w_odo,
        w_loop=w_loop,
    )
    after = compute_edge_residuals(poses_after, edges)

    return {
        "poses_before": poses_before,
        "poses_after": poses_after,
        "odom_residuals_before": before["odom_residuals"],
        "odom_residuals_after": after["odom_residuals"],
        "loop_residuals_before": before["loop_residuals"],
        "loop_residuals_after": after["loop_residuals"],
        "num_odom_edges": before["num_odom_edges"],
        "num_loop_edges": before["num_loop_edges"],
    }
