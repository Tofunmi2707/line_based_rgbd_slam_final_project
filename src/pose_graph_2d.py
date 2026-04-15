import numpy as np
import math


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def v2t(p):
    x, y, th = p
    c, s = np.cos(th), np.sin(th)
    T = np.array([
        [c, -s, x],
        [s,  c, y],
        [0,  0, 1]
    ], dtype=float)
    return T


def t2v(T):
    x, y = T[0, 2], T[1, 2]
    th = math.atan2(T[1, 0], T[0, 0])
    return np.array([x, y, th], dtype=float)


def invT(T):
    R = T[:2, :2]
    t = T[:2, 2]
    Ti = np.eye(3)
    Ti[:2, :2] = R.T
    Ti[:2, 2] = -R.T @ t
    return Ti


def between(xi, xj):
    Ti = invT(v2t(xi))
    Tij = Ti @ v2t(xj)
    return t2v(Tij)


def edge_error(xi, xj, z):
    Z = v2t(z)
    Tij = v2t(between(xi, xj))
    E = invT(Z) @ Tij
    e = t2v(E)
    e[2] = wrap(e[2])
    return e


def edge_residual_norm(xi, xj, z):
    e = edge_error(xi, xj, z)
    return float(np.linalg.norm(e))


def num_jacobian(f, x, eps=1e-6):
    J = np.zeros((3, len(x)))
    fx = f(x)
    for k in range(len(x)):
        xp = x.copy()
        xp[k] += eps
        fp = f(xp)
        J[:, k] = (fp - fx) / eps
    return J


def compute_edge_residuals(poses, edges):
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


def optimise_pose_graph(poses, edges, iters=10, w_odo=1.0, w_loop=3.0):
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


def optimise_pose_graph_with_metrics(poses,
                                     edges,
                                     iters=10,
                                     w_odo=1.0,
                                     w_loop=3.0):
    poses_before = poses.copy()

    before = compute_edge_residuals(poses_before, edges)
    poses_after = optimise_pose_graph(
        poses_before.copy(),
        edges,
        iters=iters,
        w_odo=w_odo,
        w_loop=w_loop
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
