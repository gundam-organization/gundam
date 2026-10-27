#!/usr/bin/env python3
"""Validate one common set of variable names per binning, independently of smoothing."""
import json
import sys
from importlib import import_module
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fast-tests"))
import_module("005PythonSetup-utils").requireVenv(__file__)


def make_bin(*edges):
    return {"edgesList": [{"name": name, "bounds": bounds} for name, bounds in edges]}


def configure_samples(samples):
    import GUNDAM

    config = {"likelihoodInterfaceConfig": {
        "propagatorConfig": {"sampleSetConfig": {"sampleList": samples}},
    }}
    engine = GUNDAM.FitterEngine()
    engine.setConfig(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
    engine.configure()
    return engine


def configure_binning(binning, smooth=False):
    sample = {"name": "sample", "binning": binning}
    if smooth:
        sample["binningResponse"] = {"isEnabled": True, "variables": [{"name": "X", "width": 0.2}]}
    return configure_samples([sample])


def expect_rejection(binning, message, smooth=False):
    try:
        configure_binning(binning, smooth)
    except RuntimeError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"Accepted invalid binning: {binning}")


def main():
    import GUNDAM

    print("Testing module:", GUNDAM.__file__, flush=True)
    first = make_bin(("X", [0, 1]), ("Y", [0, 1]))
    reordered = make_bin(("Y", [0, 2]), ("X", [1, 3]))
    for smooth in [False, True]:
        engine = configure_binning({"binList": [first, reordered]}, smooth)
        sample = engine.getLikelihoodInterface().getModelPropagator().getSampleSet().getSampleList()[0]
        assert sample.getHistogram().getNbBins() == 2
        for second in [make_bin(("X", [2, 3])),
                       make_bin(("X", [2, 3]), ("Y", [0, 1]), ("Z", [0, 1])),
                       make_bin(("X", [2, 3]), ("Z", [0, 1]))]:
            expect_rejection({"binList": [first, second]}, "All bins must define the same variables", smooth)
        expect_rejection({"binningDefinition": [{"name": "X", "edges": [0, 1]}],
                          "binList": [make_bin(("Y", [2, 3]))]},
                         "All bins must define the same variables", smooth)

    expect_rejection({"binList": [make_bin(("X", [0, 1]), ("X", [2, 3]))]},
                     "Duplicate variable in bin")
    configure_binning({"sortBins": True, "binList": [reordered, first]})
    configure_binning({"binningDefinition": [{"name": "X", "edges": [0, 1, 2]},
                                            {"name": "Y", "edges": [0, 1, 2]}]})
    configure_samples([{"name": "first", "binning": {"binList": [make_bin(("X", [0, 1]))]}},
                       {"name": "second", "binning": {"binList": [make_bin(("Y", [0, 1]))]}}])

    valid = Path.cwd() / "219BinningVariables-valid.txt"
    valid.write_text("variables: X X Y Y\n0 1 0 1\nvariables: Y Y X X\n0 2 1 3\n")
    configure_binning(str(valid))
    invalid = Path.cwd() / "219BinningVariables-invalid.txt"
    invalid.write_text("variables: X X Y Y\n0 1 0 1\nvariables: X X\n2 3\n")
    expect_rejection(str(invalid), "All bins must define the same variables")
    print("SUCCESS: bin variables are consistent across JSON/YAML and text definitions.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
