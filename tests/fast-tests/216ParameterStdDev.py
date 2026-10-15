#!/usr/bin/env python3
"""Check individual Gaussian priors and overrides while preserving correlations.

Needs ROOT in PATH and the freshly built GUNDAM module in PYTHONPATH.
"""

import json
import math
import subprocess
from pathlib import Path


def assert_close(actual, expected):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise AssertionError(f"Expected {expected}, got {actual}")


def write_input(path):
    macro = path.with_name("makeStdDevInput.C")
    macro.write_text(f'''#include <TFile.h>
#include <TMatrixDSym.h>
#include <TTree.h>
void makeStdDevInput() {{
  TFile file("{path}", "RECREATE");
  TMatrixDSym covariance(2);
  covariance[0][0] = 4.; covariance[1][1] = 9.;
  covariance[0][1] = covariance[1][0] = 3.; // rho = 0.5
  covariance.Write("covariance");
  double x = 0.;
  TTree tree("tree_mc", "tree_mc");
  tree.Branch("X", &x); tree.Fill(); tree.Write();
}}
''')
    subprocess.run(["root", "-l", "-b", "-q", str(macro)], check=True)


def run_case(path, definitions, correlated=False, eigen=False, nominal_step=0.1):
    import GUNDAM

    parameter_set_config = {
        "name": "priors", "isEnabled": True, "parameterDefinitions": definitions,
        "enableEigenDecomp": eigen,
    }
    if nominal_step is not None:
        parameter_set_config["nominalStepSize"] = nominal_step
    if correlated:
        parameter_set_config.update(parameterDefinitionFilePath=str(path), covarianceMatrix="covariance")
    config = {"likelihoodInterfaceConfig": {
        "jointProbabilityConfig": {"type": "PoissonLLH"},
        "dataSetList": [{"name": "sample", "model": {
            "tree": "tree_mc", "filePathList": [str(path)],
        }}],
        "propagatorConfig": {
            "sampleSetConfig": {"sampleList": [{
                "name": "X", "binning": {"binningDefinition": [{"name": "X", "edges": [-1, 1]}]},
                "dataSets": ["sample"],
            }]},
            "parametersManagerConfig": {"parameterSetList": [parameter_set_config]},
        },
    }}
    # Normalization dials keep the parameters enabled through propagation setup.
    for index, definition in enumerate(definitions):
        definition.update(parameterIndex=index, name=f"par{index}", priorValue=1.,
                          dialSetDefinitions=[{"dialType": "Normalization", "applyOnDataSets": ["sample"]}])
    builder = GUNDAM.ConfigUtils.ConfigBuilder()
    builder.setConfigFromYamlString(json.dumps(config))
    engine = GUNDAM.FitterEngine()
    engine.setConfig(GUNDAM.ConfigUtils.ConfigReader(builder.getConfig()))
    engine.configure()
    likelihood = engine.getLikelihoodInterface()
    likelihood.initialize()
    propagator = likelihood.getModelPropagator()
    manager = propagator.getParametersManager()
    parameter_set = manager.getParameterSetsList()[0]
    parameters = parameter_set.getParameterList()
    for parameter in parameters:
        if parameter.isEnabled():
            parameter.setParameterValue(3.)
    if parameter_set.isEnableEigenDecomp():
        parameter_set.propagateOriginalToEigen()
    likelihood.propagateAndEvalLikelihood()
    penalty = likelihood.getLastPenaltyLikelihood()
    for parameter, definition in zip(parameters, definitions):
        if not parameter.isEnabled():
            continue
        if definition.get("priorType") == "Flat":
            assert math.isnan(parameter.getStdDevValue())
            assert parameter.isFree()
        elif "stdDev" in definition and definition["stdDev"] != "nan":
            assert_close(parameter.getStdDevValue(), definition["stdDev"])
        elif not correlated or definition.get("stdDev", 0) == "nan":
            assert math.isnan(parameter.getStdDevValue())
            assert parameter.isFree()
    if correlated and all(not p.isFree() for p in parameters):
        assert_close(parameter_set.getPriorCorrelationMatrix()[0, 1], 0.5)
        covariance = parameter_set.getPriorCovarianceMatrix()
        sigmas = [p.getStdDevValue() for p in parameters]
        assert_close(covariance[0, 1], 0.5 * sigmas[0] * sigmas[1])
    return penalty


def main():
    import GUNDAM

    GUNDAM.setRuntimeWorkingDirectory(str(Path.cwd()))
    GUNDAM.setLightOutputMode(True)
    GUNDAM.setNumberOfThreads(1)
    path = Path.cwd() / "216ParameterStdDev.root"
    write_input(path)
    for eigen in (False, True):
        # z = (1, 2/3), rho = 1/2: chi2 = (z0^2 - z0*z1 + z1^2)/(3/4).
        assert_close(run_case(path, [{}, {}], correlated=True, eigen=eigen), 28./27.)
        # Both sigmas doubled; same correlations, quarter of the original penalty.
        assert_close(run_case(path, [{"stdDev": 4.}, {"stdDev": 6.}],
                              correlated=True, eigen=eigen), 7./27.)
        assert_close(run_case(path, [{"stdDev": 2.}, {"stdDev": 4.}], eigen=eigen), 1.25)
        assert_close(run_case(path, [{}, {}], eigen=eigen), 0.)
    # Free parameters interleaved with constrained parameters exercise compact matrix ordering.
    assert_close(run_case(path, [{}, {"stdDev": 2.}, {}, {"stdDev": 4.},
                                 {"stdDev": 1., "isPenaltyDisabled": True},
                                 {"stdDev": 1., "isEnabled": False}]), 1.25)
    assert_close(run_case(path, [{"stdDev": 2., "priorType": "Flat"},
                                 {"stdDev": 4., "isFixed": True}]), 0.25)
    assert_close(run_case(path, [{"priorType": "Flat"}, {"stdDev": 6.}], correlated=True), 1./9.)
    # The string "nan" explicitly removes an existing prior constraint.
    assert_close(run_case(path, [{"stdDev": "nan"}, {"stdDev": 6.}], correlated=True), 1./9.)
    for invalid_sigma in (0., -1.):
        try:
            run_case(path, [{"stdDev": invalid_sigma}])
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"Accepted invalid stdDev: {invalid_sigma}")
    # Free parameters need an explicitly supplied scale, including covariance overrides.
    for correlated in (False, True):
        definitions = [{"stdDev": "nan"}, {"stdDev": 6.}]
        for invalid_step in (None, 0., -0.1):
            try:
                run_case(path, definitions, correlated=correlated, nominal_step=invalid_step)
            except RuntimeError as error:
                assert "nominalStepSize" in str(error), str(error)
            else:
                raise AssertionError(f"Accepted free parameter with nominalStepSize={invalid_step}")
        assert_close(run_case(path, [{"stdDev": "nan", "parameterStepSize": 0.2},
                                    {"stdDev": 6.}], correlated=correlated, nominal_step=None), 1./9.)
        assert_close(run_case(path, [{"priorType": "Gaussian", "stdDev": 2.},
                                    {"stdDev": 4.}], correlated=correlated, nominal_step=None),
                     1. if correlated else 1.25)
    print("SUCCESS: individual std-devs, preserved correlations, eigen priors, and free parameters.")


if __name__ == "__main__":
    main()
