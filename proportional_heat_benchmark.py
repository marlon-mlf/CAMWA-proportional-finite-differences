"""Reproducibility script for the proportional finite-difference manuscript.

The script is deterministic and uses no random sampling. By default it computes all
numerical values reported in the manuscript, prints compact summaries, writes CSV
files, writes LaTeX table fragments, and regenerates the convergence figure.

Usage from the article root directory:
    python3 Supplementary/proportional_heat_benchmark.py --outdir Supplementary/output
    python3 Supplementary/proportional_heat_benchmark.py --check
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional for headless checks
    plt = None

NS = [20, 40, 80, 160]


def add_orders(values: Sequence[float]) -> list[float | None]:
    orders: list[float | None] = [None]
    for prev, curr in zip(values[:-1], values[1:]):
        orders.append(math.log(prev / curr, 2.0))
    return orders


def solve_tridiagonal(lower: np.ndarray, diagonal: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve a tridiagonal linear system by the Thomas algorithm."""
    n = len(diagonal)
    if n == 0:
        return np.array([], dtype=float)
    a = lower.astype(float).copy()
    d = diagonal.astype(float).copy()
    c = upper.astype(float).copy()
    b = rhs.astype(float).copy()

    for i in range(1, n):
        factor = a[i - 1] / d[i - 1]
        d[i] -= factor * c[i - 1]
        b[i] -= factor * b[i - 1]

    x = np.empty(n, dtype=float)
    x[-1] = b[-1] / d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (b[i] - c[i] * x[i + 1]) / d[i]
    return x


def proportional_scheme_1d(N: int, T: float = 0.1, kappa: float = 1.0, target_lam: float = 0.4):
    H = math.pi / N
    M = math.ceil(T / (target_lam * H * H / kappa))
    dt = T / M
    lam = kappa * dt / (H * H)
    X = np.linspace(0.0, math.pi, N + 1)
    U = 2.0 + np.sin(X)
    for n in range(M):
        U_next = U.copy()
        U_next[1:-1] = U[1:-1] + lam * (U[2:] - 2.0 * U[1:-1] + U[:-2])
        t_next = (n + 1) * dt
        U_next[0] = 2.0 + math.exp(-t_next) * math.sin(0.0)
        U_next[-1] = 2.0 + math.exp(-t_next) * math.sin(math.pi)
        U = U_next
    U_exact = 2.0 + math.exp(-T) * np.sin(X)
    diff = U - U_exact
    return H, dt, M, lam, float(np.max(np.abs(diff))), float(np.max(np.abs(np.exp(diff) - 1.0)))


def backward_euler_1d(N: int, T: float = 0.1, kappa: float = 1.0, target_lam: float = 2.0):
    H = math.pi / N
    M = math.ceil(T / (target_lam * H * H / kappa))
    dt = T / M
    lam = kappa * dt / (H * H)
    X = np.linspace(0.0, math.pi, N + 1)
    U = 2.0 + np.sin(X)
    m = N - 1
    diagonal = (1.0 + 2.0 * lam) * np.ones(m)
    lower = -lam * np.ones(m - 1)
    upper = -lam * np.ones(m - 1)
    for n in range(M):
        t_next = (n + 1) * dt
        left = 2.0 + math.exp(-t_next) * math.sin(0.0)
        right = 2.0 + math.exp(-t_next) * math.sin(math.pi)
        rhs = U[1:-1].copy()
        rhs[0] += lam * left
        rhs[-1] += lam * right
        U[1:-1] = solve_tridiagonal(lower, diagonal, upper, rhs)
        U[0], U[-1] = left, right
    U_exact = 2.0 + math.exp(-T) * np.sin(X)
    diff = U - U_exact
    return H, dt, M, lam, float(np.max(np.abs(diff))), float(np.max(np.abs(np.exp(diff) - 1.0)))


def crank_nicolson_1d(N: int, T: float = 0.1, kappa: float = 1.0, target_lam: float = 2.0):
    """Crank-Nicolson reference solver for U_t=kappa U_XX on the same X-grid."""
    H = math.pi / N
    M = math.ceil(T / (target_lam * H * H / kappa))
    dt = T / M
    lam = kappa * dt / (H * H)
    X = np.linspace(0.0, math.pi, N + 1)
    U = 2.0 + np.sin(X)
    m = N - 1
    lhs_diag = (1.0 + lam) * np.ones(m)
    lhs_lower = -(lam / 2.0) * np.ones(m - 1)
    lhs_upper = -(lam / 2.0) * np.ones(m - 1)
    for n in range(M):
        t_n = n * dt
        t_next = (n + 1) * dt
        left_n = 2.0 + math.exp(-t_n) * math.sin(0.0)
        right_n = 2.0 + math.exp(-t_n) * math.sin(math.pi)
        left_np1 = 2.0 + math.exp(-t_next) * math.sin(0.0)
        right_np1 = 2.0 + math.exp(-t_next) * math.sin(math.pi)
        rhs = (1.0 - lam) * U[1:-1] + (lam / 2.0) * (U[:-2] + U[2:])
        rhs[0] += (lam / 2.0) * left_np1
        rhs[-1] += (lam / 2.0) * right_np1
        U[1:-1] = solve_tridiagonal(lhs_lower, lhs_diag, lhs_upper, rhs)
        U[0], U[-1] = left_np1, right_np1
    U_exact = 2.0 + math.exp(-T) * np.sin(X)
    diff = U - U_exact
    return H, dt, M, lam, float(np.max(np.abs(diff))), float(np.max(np.abs(np.exp(diff) - 1.0)))


def source_driven_scheme_1d(N: int, T: float = 0.1, kappa: float = 1.0, target_lam: float = 0.4):
    if abs(kappa - 1.0) > 1e-14:
        raise ValueError("source_driven_scheme_1d is defined for kappa=1 in this benchmark")
    H = math.pi / N
    M = math.ceil(T / (target_lam * H * H / kappa))
    dt = T / M
    lam = kappa * dt / (H * H)
    X = np.linspace(0.0, math.pi, N + 1)

    def exact(t: float) -> np.ndarray:
        return 2.0 + math.exp(-t) * np.sin(X) + t * np.sin(2.0 * X)

    def source(t: float) -> np.ndarray:
        return (1.0 + 4.0 * t) * np.sin(2.0 * X)

    U = exact(0.0)
    for n in range(M):
        t_n = n * dt
        U_next = U.copy()
        U_next[1:-1] = U[1:-1] + lam * (U[2:] - 2.0 * U[1:-1] + U[:-2]) + dt * source(t_n)[1:-1]
        U_next[0] = exact((n + 1) * dt)[0]
        U_next[-1] = exact((n + 1) * dt)[-1]
        U = U_next
    diff = U - exact(T)
    return H, dt, M, lam, float(np.max(np.abs(diff))), float(np.max(np.abs(np.exp(diff) - 1.0)))


def variable_coefficient_scheme_1d(N: int, T: float = 0.1, target_lam: float = 0.2):
    H = math.pi / N
    max_a = 1.5
    M = math.ceil(T / (target_lam * H * H / max_a))
    dt = T / M
    lam = dt / (H * H)
    X = np.linspace(0.0, math.pi, N + 1)
    X_half = 0.5 * (X[:-1] + X[1:])
    a_half = 1.0 + 0.5 * np.cos(X_half)

    def exact(t: float) -> np.ndarray:
        return 2.0 + math.exp(-t) * np.sin(X)

    def source(t: float) -> np.ndarray:
        return math.exp(-t) * np.sin(X) * np.cos(X)

    U = exact(0.0)
    for n in range(M):
        t_n = n * dt
        flux_plus = a_half[1:] * (U[2:] - U[1:-1])
        flux_minus = a_half[:-1] * (U[1:-1] - U[:-2])
        U_next = U.copy()
        U_next[1:-1] = U[1:-1] + dt * ((flux_plus - flux_minus) / (H * H) + source(t_n)[1:-1])
        U_next[0] = exact((n + 1) * dt)[0]
        U_next[-1] = exact((n + 1) * dt)[-1]
        U = U_next
    diff = U - exact(T)
    return H, dt, M, lam, float(np.max(np.abs(diff))), float(np.max(np.abs(np.exp(diff) - 1.0)))


def additive_x_scheme(N: int, T: float = 0.1, kappa: float = 1.0, safety: float = 0.2):
    a = 1.0
    b = math.exp(math.pi)
    x = np.linspace(a, b, N + 1)
    dx = (b - a) / N
    dt_stable = safety * dx * dx / (kappa * b * b)
    M = math.ceil(T / dt_stable)
    dt = T / M
    W = 2.0 + np.sin(np.log(x))
    for n in range(M):
        W_next = W.copy()
        xi = x[1:-1]
        W_xx = (W[2:] - 2.0 * W[1:-1] + W[:-2]) / (dx * dx)
        W_x = (W[2:] - W[:-2]) / (2.0 * dx)
        W_next[1:-1] = W[1:-1] + dt * kappa * (xi * xi * W_xx + xi * W_x)
        t_next = (n + 1) * dt
        W_next[0] = 2.0 + math.exp(-t_next) * math.sin(math.log(a))
        W_next[-1] = 2.0 + math.exp(-t_next) * math.sin(math.log(b))
        W = W_next
    W_exact = 2.0 + math.exp(-T) * np.sin(np.log(x))
    diff = W - W_exact
    return dx, dt, M, float(np.max(np.abs(diff))), float(np.max(np.abs(np.exp(diff) - 1.0)))


def same_node_nonuniform_x_scheme(N: int, T: float = 0.1, kappa: float = 1.0, target_inverse_rho: float = 0.4):
    H = math.pi / N
    X = np.linspace(0.0, math.pi, N + 1)
    x = np.exp(X)
    m = N - 1
    L = np.zeros((m, m), dtype=float)
    coeffs: list[tuple[float, float, float]] = []
    for row, i in enumerate(range(1, N)):
        hm = x[i] - x[i - 1]
        hp = x[i + 1] - x[i]
        xi = x[i]
        w1m = -hp / (hm * (hm + hp))
        w10 = (hp - hm) / (hm * hp)
        w1p = hm / (hp * (hm + hp))
        w2m = 2.0 / (hm * (hm + hp))
        w20 = -2.0 / (hm * hp)
        w2p = 2.0 / (hp * (hm + hp))
        am = kappa * (xi * xi * w2m + xi * w1m)
        a0 = kappa * (xi * xi * w20 + xi * w10)
        ap = kappa * (xi * xi * w2p + xi * w1p)
        coeffs.append((am, a0, ap))
        if i - 1 >= 1:
            L[row, row - 1] = am
        L[row, row] = a0
        if i + 1 <= N - 1:
            L[row, row + 1] = ap
    rho = max(abs(np.linalg.eigvals(L)))
    M = math.ceil(T / (target_inverse_rho / float(rho)))
    dt = T / M
    lam = kappa * dt / (H * H)
    W = 2.0 + np.sin(np.log(x))
    for n in range(M):
        W_next = W.copy()
        for row, i in enumerate(range(1, N)):
            am, a0, ap = coeffs[row]
            W_next[i] = W[i] + dt * (am * W[i - 1] + a0 * W[i] + ap * W[i + 1])
        t_next = (n + 1) * dt
        W_next[0] = 2.0 + math.exp(-t_next) * math.sin(math.log(x[0]))
        W_next[-1] = 2.0 + math.exp(-t_next) * math.sin(math.log(x[-1]))
        W = W_next
    W_exact = 2.0 + math.exp(-T) * np.sin(np.log(x))
    diff = W - W_exact
    return H, dt, M, lam, float(np.max(np.abs(diff))), float(np.max(np.abs(np.exp(diff) - 1.0)))


def proportional_heat_2d(N: int, T: float = 0.05, kappa: float = 1.0, target_sum_lam: float = 0.4):
    H = math.pi / N
    target_lam = target_sum_lam / 2.0
    M = math.ceil(T / (target_lam * H * H / kappa))
    dt = T / M
    lam = kappa * dt / (H * H)
    X = np.linspace(0.0, math.pi, N + 1)
    Y = np.linspace(0.0, math.pi, N + 1)
    XX, YY = np.meshgrid(X, Y, indexing="ij")
    U = 2.0 + np.sin(XX) * np.sin(YY)
    for _ in range(M):
        U_next = U.copy()
        U_next[1:-1, 1:-1] = U[1:-1, 1:-1] + lam * (U[2:, 1:-1] - 2.0 * U[1:-1, 1:-1] + U[:-2, 1:-1]) + lam * (U[1:-1, 2:] - 2.0 * U[1:-1, 1:-1] + U[1:-1, :-2])
        U_next[0, :] = 2.0
        U_next[-1, :] = 2.0
        U_next[:, 0] = 2.0
        U_next[:, -1] = 2.0
        U = U_next
    U_exact = 2.0 + math.exp(-2.0 * T) * np.sin(XX) * np.sin(YY)
    return dt, lam, float(np.max(np.abs(U - U_exact)))


def mixed_operator_error(N: int) -> float:
    H = math.pi / N
    X = np.linspace(0.0, math.pi, N + 1)
    Y = np.linspace(0.0, math.pi, N + 1)
    XX, YY = np.meshgrid(X, Y, indexing="ij")
    U = np.sin(XX) * np.sin(YY)
    mixed_difference = (U[2:, 2:] - U[2:, :-2] - U[:-2, 2:] + U[:-2, :-2]) / (4.0 * H * H)
    exact = np.cos(XX[1:-1, 1:-1]) * np.cos(YY[1:-1, 1:-1])
    return float(np.max(np.abs(mixed_difference - exact)))


def smooth_x_derivative_diagnostic(N: int):
    a = 1.0
    b = math.exp(math.pi)
    length = b - a
    X = np.linspace(0.0, math.pi, N + 1)
    x_geo = np.exp(X)
    U_geo = np.sin(math.pi * (x_geo - a) / length)
    H = math.pi / N
    geom_derivative = (U_geo[2:] - U_geo[:-2]) / (2.0 * H)
    geom_exact = x_geo[1:-1] * (math.pi / length) * np.cos(math.pi * (x_geo[1:-1] - a) / length)
    geometric_error = float(np.max(np.abs(geom_derivative - geom_exact)))
    x_add = np.linspace(a, b, N + 1)
    W_add = np.sin(math.pi * (x_add - a) / length)
    h = length / N
    additive_derivative = x_add[1:-1] * (W_add[2:] - W_add[:-2]) / (2.0 * h)
    additive_exact = x_add[1:-1] * (math.pi / length) * np.cos(math.pi * (x_add[1:-1] - a) / length)
    additive_error = float(np.max(np.abs(additive_derivative - additive_exact)))
    return geometric_error, additive_error


def compute_all() -> dict[str, list[dict[str, float | int | None]]]:
    prop_raw = [proportional_scheme_1d(N) for N in NS]
    prop_orders = add_orders([row[4] for row in prop_raw])
    prop = [dict(N=N, H=H, dt=dt, M=M, lam=lam, Elog=Elog, Erel=Erel, order=order) for N, (H, dt, M, lam, Elog, Erel), order in zip(NS, prop_raw, prop_orders)]

    be_raw = [backward_euler_1d(N) for N in NS]
    be_orders = add_orders([row[4] for row in be_raw])
    be = [dict(N=N, H=H, dt=dt, M=M, lam=lam, Elog=Elog, Erel=Erel, order=order) for N, (H, dt, M, lam, Elog, Erel), order in zip(NS, be_raw, be_orders)]

    cn_raw = [crank_nicolson_1d(N) for N in NS]
    cn_orders = add_orders([row[4] for row in cn_raw])
    cn = [dict(N=N, H=H, dt=dt, M=M, lam=lam, Elog=Elog, Erel=Erel, order=order) for N, (H, dt, M, lam, Elog, Erel), order in zip(NS, cn_raw, cn_orders)]

    source_raw = [source_driven_scheme_1d(N) for N in NS]
    source_orders = add_orders([row[4] for row in source_raw])
    source = [dict(N=N, H=H, dt=dt, M=M, lam=lam, Elog=Elog, Erel=Erel, order=order) for N, (H, dt, M, lam, Elog, Erel), order in zip(NS, source_raw, source_orders)]

    var_raw = [variable_coefficient_scheme_1d(N) for N in NS]
    var_orders = add_orders([row[4] for row in var_raw])
    var = [dict(N=N, H=H, dt=dt, M=M, dt_over_H2=lam, Elog=Elog, Erel=Erel, order=order) for N, (H, dt, M, lam, Elog, Erel), order in zip(NS, var_raw, var_orders)]

    add_raw = [additive_x_scheme(N) for N in NS]
    add_orders_list = add_orders([row[3] for row in add_raw])
    add = [dict(N=N, dx=dx, dt=dt, M=M, Elog=Elog, Erel=Erel, order=order) for N, (dx, dt, M, Elog, Erel), order in zip(NS, add_raw, add_orders_list)]

    same_raw = [same_node_nonuniform_x_scheme(N) for N in NS]
    same_orders = add_orders([row[4] for row in same_raw])
    same = [dict(N=N, H=H, dt=dt, M=M, lam=lam, Elog=Elog, Erel=Erel, order=order) for N, (H, dt, M, lam, Elog, Erel), order in zip(NS, same_raw, same_orders)]

    heat_raw = [proportional_heat_2d(N) for N in NS]
    mixed_errors = [mixed_operator_error(N) for N in NS]
    heat_orders = add_orders([row[2] for row in heat_raw])
    mixed_orders = add_orders(mixed_errors)
    twod = [dict(N=N, dt=dt, lam=lam, heat_error=heat, heat_order=ho, mixed_error=me, mixed_order=mo) for N, (dt, lam, heat), ho, me, mo in zip(NS, heat_raw, heat_orders, mixed_errors, mixed_orders)]

    smooth_raw = [smooth_x_derivative_diagnostic(N) for N in NS]
    geom_orders = add_orders([row[0] for row in smooth_raw])
    addx_orders = add_orders([row[1] for row in smooth_raw])
    smooth = [dict(N=N, geometric_error=ge, geometric_order=go, additive_error=ae, additive_order=ao, ratio=ge / ae) for N, (ge, ae), go, ao in zip(NS, smooth_raw, geom_orders, addx_orders)]

    table8 = []
    table9 = []
    for rp, ra, rs in zip(prop, add, same):
        table8.append(
            dict(
                N=rp["N"],
                Elog_geometric=rp["Elog"],
                Elog_additive=ra["Elog"],
                additive_ratio=ra["Elog"] / rp["Elog"],
                additive_order=ra["order"],
            )
        )
        table9.append(
            dict(
                N=rp["N"],
                Elog_geometric=rp["Elog"],
                Elog_nonuniform=rs["Elog"],
                nonuniform_ratio=rs["Elog"] / rp["Elog"],
                nonuniform_order=rs["order"],
            )
        )

    return {
        "table3_explicit": prop,
        "table4_backward_euler": be,
        "table5_crank_nicolson": cn,
        "table6_source": source,
        "table7_variable_coefficient": var,
        "table8_additive_x": table8,
        "table9_same_node": table9,
        "table10_twod": twod,
        "table11_smooth_x": smooth,
    }


def fmt_float(x: float, kind: str = "sci") -> str:
    if kind == "fixed6":
        return f"{x:.6f}"
    if kind == "dt":
        return f"{x:.8g}"
    if kind == "sci":
        return f"{x:.4e}"
    if kind == "order":
        return "--" if x is None else f"{x:.2f}"
    return str(x)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def latex_rows(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    lines = []
    for row in rows:
        cells = []
        for key, kind in columns:
            val = row[key]
            if val is None:
                cells.append("--")
            elif isinstance(val, float):
                cells.append(fmt_float(val, kind))
            else:
                cells.append(str(val))
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def write_latex_tables(outdir: Path, tables: dict[str, list[dict[str, object]]]) -> None:
    table_dir = outdir / "latex_tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    specs = {
        "table3_explicit": [("N", "int"), ("H", "fixed6"), ("dt", "dt"), ("M", "int"), ("lam", "fixed6"), ("Elog", "sci"), ("Erel", "sci"), ("order", "order")],
        "table4_backward_euler": [("N", "int"), ("H", "fixed6"), ("dt", "dt"), ("M", "int"), ("lam", "fixed6"), ("Elog", "sci"), ("Erel", "sci"), ("order", "order")],
        "table5_crank_nicolson": [("N", "int"), ("H", "fixed6"), ("dt", "dt"), ("M", "int"), ("lam", "fixed6"), ("Elog", "sci"), ("Erel", "sci"), ("order", "order")],
        "table6_source": [("N", "int"), ("H", "fixed6"), ("dt", "dt"), ("M", "int"), ("lam", "fixed6"), ("Elog", "sci"), ("Erel", "sci"), ("order", "order")],
        "table7_variable_coefficient": [("N", "int"), ("H", "fixed6"), ("dt", "dt"), ("M", "int"), ("dt_over_H2", "fixed6"), ("Elog", "sci"), ("Erel", "sci"), ("order", "order")],
        "table8_additive_x": [("N", "int"), ("Elog_geometric", "sci"), ("Elog_additive", "sci"), ("additive_ratio", "fixed2"), ("additive_order", "order")],
        "table9_same_node": [("N", "int"), ("Elog_geometric", "sci"), ("Elog_nonuniform", "sci"), ("nonuniform_ratio", "fixed2"), ("nonuniform_order", "order")],
        "table10_twod": [("N", "int"), ("dt", "dt"), ("lam", "fixed6"), ("heat_error", "sci"), ("heat_order", "order"), ("mixed_error", "sci"), ("mixed_order", "order")],
        "table11_smooth_x": [("N", "int"), ("geometric_error", "sci"), ("geometric_order", "order"), ("additive_error", "sci"), ("additive_order", "order"), ("ratio", "fixed2")],
    }
    # Custom formatting for two decimal ratios
    global fmt_float
    old_fmt = fmt_float
    def fmt_float2(x: float, kind: str = "sci") -> str:
        if kind == "fixed2":
            return f"{x:.2f}"
        return old_fmt(x, kind)
    fmt_float = fmt_float2
    try:
        for name, cols in specs.items():
            (table_dir / f"{name}.tex").write_text(latex_rows(tables[name], cols), encoding="utf-8")
    finally:
        fmt_float = old_fmt


def write_outputs(outdir: Path, tables: dict[str, list[dict[str, object]]]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        write_csv(outdir / "csv" / f"{name}.csv", rows)
    write_latex_tables(outdir, tables)
    metadata = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "tables": list(tables.keys()),
    }
    (outdir / "environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    ref_json = json.dumps(tables, indent=2, sort_keys=True)
    (outdir / "computed_tables.json").write_text(ref_json, encoding="utf-8")
    sha = hashlib.sha256(ref_json.encode("utf-8")).hexdigest()
    (outdir / "checksums.txt").write_text(f"computed_tables.json sha256 {sha}\n", encoding="utf-8")


def plot_convergence(outdir: Path, tables: dict[str, list[dict[str, object]]]) -> None:
    if plt is None:
        return
    rows = tables["table3_explicit"]
    H = np.array([float(r["H"]) for r in rows])
    Elog = np.array([float(r["Elog"]) for r in rows])
    ref = Elog[-1] * (H / H[-1]) ** 2
    fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
    ax.loglog(H, Elog, marker="o", linewidth=1.6, markersize=5, label=r"$E_{\log}$")
    ax.loglog(H, ref, linestyle="--", linewidth=1.4, label="slope 2 reference")
    ax.set_xlabel(r"Mesh size $H$ in logarithmic coordinate")
    ax.set_ylabel(r"Logarithmic error $E_{\log}$")
    ax.set_title("Second-order convergence in logarithmic maximum norm", pad=8)
    # Use the actual mesh sizes as major ticks and suppress minor tick labels
    # to avoid crowded scientific-log labels in the manuscript PDF.
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter

    tick_values = np.sort(H)
    ax.xaxis.set_major_locator(FixedLocator(tick_values))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{v:.4f}" for v in tick_values]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", labelbottom=False)
    ax.tick_params(axis="x", which="major", labelsize=8)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.35)
    ax.legend(loc="best", frameon=False)
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / "convergence_loglog.pdf")
    fig.savefig(figdir / "convergence_loglog.png", dpi=240)
    plt.close(fig)


def print_summary(tables: dict[str, list[dict[str, object]]]) -> None:
    for name, rows in tables.items():
        print(f"\n{name}")
        print(", ".join(rows[0].keys()))
        for row in rows:
            print(", ".join("--" if v is None else (f"{v:.6g}" if isinstance(v, float) else str(v)) for v in row.values()))


def check_regression(tables: dict[str, list[dict[str, object]]], reference_path: Path, rtol: float = 5e-8, atol: float = 5e-10) -> None:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    for key, rows in tables.items():
        if key not in reference:
            raise AssertionError(f"Missing reference table {key}")
        ref_rows = reference[key]
        if len(rows) != len(ref_rows):
            raise AssertionError(f"Row-count mismatch in {key}")
        for i, (row, ref_row) in enumerate(zip(rows, ref_rows)):
            for field, val in row.items():
                ref_val = ref_row[field]
                if val is None and ref_val is None:
                    continue
                if isinstance(val, float):
                    if not math.isclose(val, float(ref_val), rel_tol=rtol, abs_tol=atol):
                        raise AssertionError(f"Mismatch {key}[{i}].{field}: {val} != {ref_val}")
                else:
                    if val != ref_val:
                        raise AssertionError(f"Mismatch {key}[{i}].{field}: {val} != {ref_val}")
    print("Regression check passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce manuscript tables and figures.")
    parser.add_argument("--outdir", default="Supplementary/output", help="Output directory for CSV, LaTeX tables and figures.")
    parser.add_argument("--check", action="store_true", help="Compare computed results with regression_reference.json.")
    parser.add_argument("--no-plot", action="store_true", help="Do not generate the convergence plot.")
    args = parser.parse_args()
    tables = compute_all()
    outdir = Path(args.outdir)
    write_outputs(outdir, tables)
    if not args.no_plot:
        plot_convergence(outdir, tables)
    print_summary(tables)
    if args.check:
        ref = Path(__file__).with_name("regression_reference.json")
        check_regression(tables, ref)


if __name__ == "__main__":
    main()
