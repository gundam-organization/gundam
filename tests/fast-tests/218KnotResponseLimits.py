#!/usr/bin/env python3
"""Bounded dials must match dials built from explicitly clipped input knots."""
import json
import math
import subprocess
import sys
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True
import_module("005PythonSetup-utils").requireVenv(__file__)
evaluate = import_module("215ArrayDialBranchData").evaluate


def write_input(path, limits=None):
    # ROOT object branches are not writable with uproot.
    macro = Path.cwd() / "218KnotResponseLimits-input.C"
    lower = repr(limits['min']) if limits and 'min' in limits else '-std::numeric_limits<double>::infinity()'
    upper = repr(limits['max']) if limits and 'max' in limits else 'std::numeric_limits<double>::infinity()'
    macro.write_text(r'''#include <TFile.h>
#include <TTree.h>
#include <TGraph.h>
#include <TSystem.h>
#include <algorithm>
#include <limits>
#include <vector>
{
 {
  TFile output(OUTPUT_PATH, "RECREATE");
  TTree tree("events", "events");
  double coordinate = .5;
  TGraph graphObject;
  TGraph* graph = &graphObject;
  tree.Branch("X", &coordinate, "X/D");
  tree.Branch("graph", "TGraph", &graph, 32000, 0);
  std::vector<std::vector<double>> xs{{}, {0.}, {-1., 2.}, {-1., 0., 1., 2.},
                                      {-1., 0., .75, 2.}, {-1., 0., 1., 2.}};
  std::vector<std::vector<double>> ys{{}, {3.}, {.25, 4.}, {.25, 1., 2., 4.},
                                      {.25, 1., 2., 4.}, {3., 3., 3., 3.}};
  for (int event = 0; event < int(xs.size()); ++event) {
    graph->Set(xs[event].size());
    for (int point = 0; point < int(xs[event].size()); ++point) {
      double y = std::max(LOWER_LIMIT, std::min(ys[event][point], UPPER_LIMIT));
      graph->SetPoint(point, xs[event][point], y);
    }
    tree.Fill();
  }
  tree.Write();
 }
 gSystem->Exit(0);
}
'''.replace("OUTPUT_PATH", json.dumps(str(path))).replace("LOWER_LIMIT", lower).replace("UPPER_LIMIT", upper))
    subprocess.run(["root", "-n", "-l", "-b", "-q", str(macro)], check=True)


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
