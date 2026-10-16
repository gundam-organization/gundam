#!/usr/bin/env python3
"""Check individual Gaussian priors and overrides while preserving correlations.

Needs ROOT in PATH and the freshly built GUNDAM module in PYTHONPATH.
"""

import json
import math
import os
import sys
from contextlib import contextmanager
import subprocess
from pathlib import Path


@contextmanager
def suppress_cpp_output():
    import GUNDAM

    GUNDAM.flushOutput()
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout, saved_stderr = os.dup(1), os.dup(2)
    try:
        with open(os.devnull, "w") as output:
            os.dup2(output.fileno(), 1)
            os.dup2(output.fileno(), 2)
            try:
                yield
            finally:
                GUNDAM.flushOutput()
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)


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
  TMatrixDSym covariance3(3);
  covariance3[0][0] = 4.; covariance3[1][1] = 9.; covariance3[2][2] = 16.;
  covariance3[0][1] = covariance3[1][0] = 3.;   // rho = 0.5
  covariance3[0][2] = covariance3[2][0] = 2.;   // rho = 0.25
  covariance3[1][2] = covariance3[2][1] = -2.4; // rho = -0.2
  covariance3.Write("covariance3");
  double x = 0.;
  TTree tree("tree_mc", "tree_mc");
  tree.Branch("X", &x); tree.Fill(); tree.Write();
}}
''')
    subprocess.run(["root", "-l", "-b", "-q", str(macro)], check=True)


def run_case(path, definitions, correlated=False, eigen=False, nominal_step=0.1,
             check_throws=False, global_throws=False, expected_steps=None):
    import GUNDAM

    parameter_set_config = {
        "name": "priors", "isEnabled": True, "parameterDefinitions": definitions,
        "enableEigenDecomp": eigen,
    }
    if nominal_step is not None:
        parameter_set_config["nominalStepSize"] = nominal_step
    if correlated:
        parameter_set_config.update(parameterDefinitionFilePath=str(path),
                                    covarianceMatrix="covariance3" if len(definitions) == 3 else "covariance")
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
            "parametersManagerConfig": {"parameterSetList": [parameter_set_config],
                                        "throwToyParametersWithGlobalCov": global_throws},
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
    if expected_steps is not None:
        for parameter, expected_step in zip(parameters, expected_steps):
            assert_close(parameter.getStepSize(), expected_step)
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
    if correlated:
        input_correlations = ((1., 0.5, 0.25), (0.5, 1., -0.2), (0.25, -0.2, 1.))
        constrained = [(i, p) for i, p in enumerate(parameters)
                       if p.isEnabled() and not p.isFree() and not definitions[i].get("isPenaltyDisabled", False)]
        covariance = parameter_set.getPriorCovarianceMatrix()
        correlation = parameter_set.getPriorCorrelationMatrix()
        for i, (original_i, parameter_i) in enumerate(constrained):
            for j, (original_j, parameter_j) in enumerate(constrained):
                expected = input_correlations[original_i][original_j]
                if i != j and (definitions[original_i].get("releasePriorCorrelations", False)
                               or definitions[original_j].get("releasePriorCorrelations", False)):
                    expected = 0.
                assert_close(correlation[i, j], expected)
                assert_close(covariance[i, j], expected * parameter_i.getStdDevValue() * parameter_j.getStdDevValue())
    if check_throws:
        GUNDAM.FitterEngine.setRandomSeed(12345)
        with suppress_cpp_output():
            throws = []
            for _ in range(4000):
                manager.throwParameters()
                throws.append([p.getParameterValue() for p in parameters])
        means = [sum(row[i] for row in throws) / len(throws) for i in range(len(parameters))]
        sample_covariance = [[sum((row[i] - means[i]) * (row[j] - means[j]) for row in throws)
                              / (len(throws) - 1) for j in range(len(parameters))]
                             for i in range(len(parameters))]
        for i, parameter in enumerate(parameters):
            sigma = parameter.getStdDevValue()
            assert abs(means[i] - parameter.getPriorValue()) < 0.08 * sigma
            assert abs(math.sqrt(sample_covariance[i][i]) / sigma - 1.) < 0.08
            for j in range(len(parameters)):
                empirical_correlation = sample_covariance[i][j] / math.sqrt(
                    sample_covariance[i][i] * sample_covariance[j][j])
                assert abs(empirical_correlation - correlation[i, j]) < 0.08
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
    for eigen in (False, True):
        # Release either side of a correlation, preserving inherited or overridden std-devs.
        for released_index in (0, 1):
            definitions = [{}, {}]
            definitions[released_index]["releasePriorCorrelations"] = True
            assert_close(run_case(path, definitions, correlated=True, eigen=eigen), 13./9.)
            definitions[0]["stdDev"] = 4.
            assert_close(run_case(path, definitions, correlated=True, eigen=eigen), 25./36.)
        # Releasing one parameter must preserve correlations between the other two.
        for released_index, expected in enumerate((1609./864., 68./45., 139./108.)):
            definitions = [{}, {}, {}]
            definitions[released_index]["releasePriorCorrelations"] = True
            assert_close(run_case(path, definitions, correlated=True, eigen=eigen), expected)
        assert_close(run_case(path, [{"stdDev": 2., "releasePriorCorrelations": True},
                                    {"stdDev": 4.}], eigen=eigen), 1.25)
    assert_close(run_case(path, [{"stdDev": "nan", "releasePriorCorrelations": True},
                                {"stdDev": 6.}], correlated=True), 1./9.)
    # Both throw paths must retain variances and the unaffected pair correlation.
    for global_throws in (False, True):
        assert_close(run_case(path, [{"stdDev": 4., "releasePriorCorrelations": True}, {}, {}],
                              correlated=True, check_throws=True, global_throws=global_throws),
                     961./864.)
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
    # Free priors retain covariance-derived steps without requiring an explicit scale.
    for free_prior in ({"priorType": "Flat"}, {"stdDev": "nan"}):
        for nominal_step in (None, 0.1):
            assert_close(run_case(path, [dict(free_prior), {}], correlated=True,
                                  nominal_step=nominal_step, expected_steps=[2., 3.]), 4./9.)
        assert_close(run_case(path, [dict(free_prior, parameterStepSize=0.2), {}],
                              correlated=True, nominal_step=None,
                              expected_steps=[0.2, 3.]), 4./9.)
    # Without covariance, free parameters still need an explicitly supplied scale.
    for invalid_step in (None, 0., -0.1):
        try:
            run_case(path, [{"stdDev": "nan"}], nominal_step=invalid_step)
        except RuntimeError as error:
            assert "nominalStepSize" in str(error), str(error)
        else:
            raise AssertionError(f"Accepted free parameter with nominalStepSize={invalid_step}")
    for correlated in (False, True):
        for invalid_step in ("nan", 0., -0.1):
            try:
                run_case(path, [{"stdDev": "nan", "parameterStepSize": invalid_step},
                                {"stdDev": 6.}], correlated=correlated)
            except RuntimeError as error:
                assert "parameterStepSize" in str(error), str(error)
            else:
                raise AssertionError(f"Accepted free parameter with parameterStepSize={invalid_step}")
        assert_close(run_case(path, [{"stdDev": "nan", "parameterStepSize": 0.2},
                                    {"stdDev": 6.}], correlated=correlated, nominal_step=None,
                              expected_steps=[0.2, 6.]), 1./9.)
        assert_close(run_case(path, [{"priorType": "Gaussian", "stdDev": 2.},
                                    {"stdDev": 4.}], correlated=correlated, nominal_step=None),
                     1. if correlated else 1.25)
    print("SUCCESS: individual std-devs, preserved correlations, eigen priors, and free parameters.")


if __name__ == "__main__":
    main()
