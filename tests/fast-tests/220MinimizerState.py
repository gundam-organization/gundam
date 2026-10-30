#!/usr/bin/env python3
"""Check Python access to minimizer state before and after minimization."""

import json
import math
import sys
from importlib import import_module
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import_module("005PythonSetup-utils").requireVenv(__file__)


def checkMinimizerState(rootPath_, maxFcnCalls_):
    import GUNDAM

    config = {
        "likelihoodInterfaceConfig": {
            "jointProbabilityConfig": {"type": "PoissonLLH"},
            "datasetList": [{"name": "input", "model": {
                "tree": "events", "filePathList": [str(rootPath_)]}}],
            "propagatorConfig": {
                "sampleSetConfig": {"sampleList": [{
                    "name": "sample", "binning": {"binningDefinition": [
                        {"name": "X", "edges": [0, 1]}]}}]},
                "parametersManagerConfig": {"parameterSetList": [{
                    "name": "normalization", "isEnabled": True, "nominalStepSize": .1,
                    "parameterDefinitions": [{
                        "name": "scale", "priorValue": 1., "priorType": "Flat",
                        "dialSetDefinitions": [{"dialType": "Normalization"}]}]}]}}},
        "minimizerConfig": {
            "type": "RootMinimizer", "maxFcnCalls": maxFcnCalls_,
            "enablePostFitErrorFit": False, "print_level": 0},
        "enablePca": False, "scaleParStepWithChi2Response": False,
        "generateSamplePlots": False, "generateOneSigmaPlots": False,
    }
    engine = GUNDAM.FitterEngine()
    engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
    likelihood = engine.getLikelihoodInterface()
    likelihood.setDataType(GUNDAM.LikelihoodInterface.DataType.Asimov)
    engine.initialize()

    minimizer = engine.getMinimizer()
    assert isinstance(minimizer, GUNDAM.RootMinimizer)
    assert minimizer.isRootMinimizer() is True
    assert minimizer.isMinimizeDone() is False
    assert minimizer.hasConverged() is False
    assert minimizer.getMinimizerStatus() == -1

    parameter = (likelihood.getModelPropagator().getParametersManager()
                 .getParameterSetsList()[0].getParameterList()[0])
    parameter.setParameterValue(1.5)
    minimizer.minimize()

    assert minimizer.isMinimizeDone() is True
    if maxFcnCalls_ == 1:
        assert minimizer.hasConverged() is False
        assert minimizer.getMinimizerStatus() > 0
    else:
        assert minimizer.hasConverged() is True
        assert minimizer.getMinimizerStatus() == 0
        assert math.isclose(parameter.getParameterValue(), 1., abs_tol=1e-3)


def main():
    import GUNDAM
    import numpy as np
    import uproot

    GUNDAM.setNumberOfThreads(1)
    GUNDAM.setLightOutputMode(True)
    rootPath = Path.cwd() / "220MinimizerState-input.root"
    with uproot.recreate(rootPath) as output:
        tree = output.mktree("events", {"X": "float64"})
        tree.extend({"X": np.full(100, .5)})

    checkMinimizerState(rootPath, 10000)
    checkMinimizerState(rootPath, 1)
    print("SUCCESS: Python minimizer state reports initial, converged and failed fits.")


if __name__ == "__main__":
    main()
