#!/usr/bin/env python3
"""Illustrate PR #1025: python illustrateCuspSmoothing.py --output /tmp/cusps.png.

Requires numpy and matplotlib. Delta is measured along x, not arc length.
Illustration only: this does not run GUNDAM or modify its configuration.
"""

import argparse
from pathlib import Path

import numpy as np


def smoothGraph(x, knots, values, delta, method):
    """Return value, first derivative and second derivative (flat outside knots).

    Interior half-widths are capped at half of either adjacent knot spacing,
    so neighboring transition intervals cannot overlap.
    """
    knots = np.asarray(knots, dtype=float)
    values = np.asarray(values, dtype=float)
    spacing = np.diff(knots)
    if delta < 0 or not np.isfinite(delta) or np.any(spacing <= 0):
        raise ValueError("Require finite delta >= 0 and strictly increasing knots")
    slopes = np.diff(values) / spacing
    segment = np.clip(np.searchsorted(knots, x, side="right") - 1, 0, len(slopes) - 1)
    y = np.interp(x, knots, values)
    dy = slopes[segment].copy()
    dy[(x < knots[0]) | (x > knots[-1])] = 0
    ddy = np.zeros_like(x)
    if delta == 0:
        return y, dy, ddy
    for i in range(1, len(knots) - 1):
        d = min(delta, spacing[i - 1] / 2, spacing[i] / 2)
        mask = (x >= knots[i] - d) & (x <= knots[i] + d)
        t = (x[mask] - knots[i] + d) / (2 * d)
        left = slopes[i - 1]
        jump = slopes[i] - left
        if method == "bezier":
            integral, blend, derivative = t**2 / 2, t, np.ones_like(t)
        elif method == "quartic":
            integral = t**3 - t**4 / 2
            blend, derivative = 3 * t**2 - 2 * t**3, 6 * t * (1 - t)
        else:
            raise ValueError(method)
        y[mask] = values[i] - d * left + 2 * d * (left * t + jump * integral)
        dy[mask] = left + jump * blend
        ddy[mask] = jump * derivative / (2 * d)
    return y, dy, ddy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta", type=float, default=0.35)
    parser.add_argument("--output", type=Path, default=Path("cuspSmoothing.png"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    cases = [
        ("Lower response limit: max(0, x)", [-2, 0, 2], [0, 0, 2]),
        ("Peak: 1 - |x|", [-2, 0, 2], [-1, 1, -1]),
        ("Several cusps (short middle segment)", [-2, -0.4, 0, 0.5, 2], [0.3, 1.2, 0.8, 1.5, 1]),
    ]
    x = np.linspace(-1.8, 1.8, 2401)
    for column, (title, knots, values) in enumerate(cases):
        for method, delta, label, style in [
            ("bezier", 0, "Original", "k--"),
            ("bezier", args.delta, "Quadratic Bezier (C1)", "C0-"),
            ("quartic", args.delta, "Quartic (C2)", "C1-"),
        ]:
            curves = smoothGraph(x, knots, values, delta, method)
            for row, curve in enumerate(curves):
                axes[row, column].plot(x, curve, style, label=label)
                axes[row, column].grid(alpha=0.25)
                axes[row, column].set_xlabel("x (dial input; raw response in first column)")
        for knot in knots[1:-1]:
            for ax in axes[:, column]:
                ax.axvline(knot, color="grey", alpha=0.25)
        axes[0, column].set_title(title)
    for row, label in enumerate(["Response", "First derivative", "Second derivative"]):
        axes[row, 0].set_ylabel(label)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Local cusp smoothing: requested half-width delta = {args.delta:g}\n"
                 "Derivatives at original cusps are undefined; dashed curves show one-sided branches")
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    print(f"Saved {args.output.resolve()}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
