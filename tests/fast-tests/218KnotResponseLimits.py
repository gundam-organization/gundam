#!/usr/bin/env python3
"""Bounded dials must match dials built from explicitly clipped input knots."""
import math
import sys
from array import array
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True
import_module("005PythonSetup-utils").requireVenv(__file__)
evaluate = import_module("215ArrayDialBranchData").evaluate


def write_input(path, limits=None):
    import ROOT

    ROOT.gROOT.SetBatch(True)
    with ROOT.TFile(str(path), "RECREATE") as output:
        tree = ROOT.TTree("events", "events")
        coordinate = array("d", [0.5])
        graph = ROOT.TGraph()
        tree.Branch("X", coordinate, "X/D")
        tree.Branch("graph", graph, 32000, 0)
        # Empty, single, two-point, uniform, nonuniform, and constant dials.
        for xs, ys in [([], []), ([0.], [3.]), ([-1., 2.], [.25, 4.]),
                       ([-1., 0., 1., 2.], [.25, 1., 2., 4.]),
                       ([-1., 0., .75, 2.], [.25, 1., 2., 4.]),
                       ([-1., 0., 1., 2.], [3.] * 4)]:
            graph.Set(len(xs))
            for index, (x, y) in enumerate(zip(xs, ys)):
                if limits:
                    y = max(limits.get("min", -math.inf), min(y, limits.get("max", math.inf)))
                graph.SetPoint(index, x, y)
            tree.Fill()
        tree.Write()


def main():
    raw = Path.cwd() / "218KnotResponseLimits-raw.root"
    clipped = Path.cwd() / "218KnotResponseLimits-clipped.root"
    write_input(raw)
    cases = [([.75, 1.5], {"min": .75, "max": 1.5}),
             ({"min": .75}, {"min": .75}),
             ({"max": 1.5}, {"max": 1.5}),
             ([1., 1.], {"min": 1., "max": 1.}),
             ([0., 10.], {"min": 0., "max": 10.})]
    for limits, bounds in cases:
        write_input(clipped, bounds)
        for dial_type, options in [("Graph", ""), ("Spline", ""),
                                   ("Spline", "akima"), ("Spline", "ROOT"),
                                   ("Spline", "monotonic")]:
            reference = evaluate([clipped], "dialBranchData: graph", False, dial_type, options)
            # RootSpline is supported only by the legacy propagator.
            for backend in ([False] if options == "ROOT" else [False, True]):
                actual = evaluate([raw], "dialBranchData: graph", backend, dial_type, options, limits)
                for result, expected in zip(actual, reference):
                    assert all(math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-7)
                               for a, b in zip(result, expected)), (limits, dial_type, options, result, expected)
    try:
        evaluate([raw], "dialBranchData: graph", False, knot_limits={"min": 2., "max": 1.})
    except RuntimeError as error:
        assert "knotResponseLimits" in str(error), str(error)
    else:
        raise AssertionError("Inverted knotResponseLimits accepted")
    print("SUCCESS: knot limits match explicitly clipped graphs, including spline slopes and CPU propagation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
