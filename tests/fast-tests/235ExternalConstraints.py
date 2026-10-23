#!/usr/bin/env python3
"""Exercise external Gaussian measurements through the full likelihood interface."""
import copy
import json
import math
import os
import subprocess
import sys
from array import array
from importlib import import_module
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import_module("005PythonSetup-utils").requireVenv(__file__)



def write_input_root_file(root_path: Path) -> None:
    import uproot

    with uproot.recreate(root_path) as root_file:
        tree = root_file.mktree(
            "tree_mc",
            {
                "X": "float64",
                "C": "float64",
            },
        )
        tree.extend(
            {
                "X": array("d", [-0.5, -0.4, -0.3, 0.2, 0.3, 0.4]),
                "C": array("d", [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]),
            }
        )


def build_config_text(root_path: Path, enable_backend: bool) -> str:
    backend_config = ""
    if enable_backend:
        backend_config = """
    backendManagerConfig:
      isEnabled: true
      type: CPU
"""

    return f"""
fitterEngineConfig:
  enablePca: false
  scaleParStepWithChi2Response: false
  generateSamplePlots: false
  generateOneSigmaPlots: false

  likelihoodInterfaceConfig:
    jointProbabilityConfig:
      type: LeastSquares
      lsqPoissonianApproximation: true

    dataSetList:
      - name: "BackendSample"
        isEnabled: true
        model:
          tree: tree_mc
          selectionCutFormula: "(1)"
          nominalWeightFormula: "(1.0)"
          filePathList:
            - "{root_path}"
{backend_config}
    propagatorConfig:
      sampleSetConfig:
        sampleList:
          - name: X
            isEnabled: true
            binning: {{ binningDefinition: [{{ name: "X", edges: [-1, 0, 1] }}] }}
            dataSets: [ "BackendSample" ]

      parametersManagerConfig:
        parameterSetList:
          - name: Normalizations
            isEnabled: true
            nominalStepSize: 0.1

            parameterDefinitions:
              - parameterName: "Positive_C"
                isEnabled: true
                priorValue: 2.0
                priorType: Flat
              - parameterName: "Negative_C"
                isEnabled: true
                priorValue: 3.0
                priorType: Flat

            dialSetDefinitions:
              - dialType: Normalization
                applyCondition: "[C] > 0"
                dialInputList:
                  - name: "Positive_C"
              - dialType: Normalization
                applyCondition: "[C] <= 0"
                dialInputList:
                  - name: "Negative_C"
"""


def set_normalization_parameters(likelihood_interface, positive_value: float, negative_value: float) -> None:
    parameter_set = (
        likelihood_interface
        .getModelPropagator()
        .getParametersManager()
        .getParameterSetsList()[0]
    )
    parameter_list = parameter_set.getParameterList()
    parameter_list[0].setParameterValue(positive_value)
    parameter_list[1].setParameterValue(negative_value)


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1.e-10, abs_tol=1.e-10), (actual, expected)


def main():
    import GUNDAM
    GUNDAM.setNumberOfThreads(1)
    GUNDAM.setLightOutputMode(True)
    GUNDAM.setRuntimeWorkingDirectory(str(Path.cwd()))
    root_path = Path.cwd() / f"235ExternalConstraints-{os.getpid()}.root"
    write_input_root_file(root_path)
    base = build_config_text(root_path, False)

    def make_engine(external=None, config=base):
        if external is not None:
            config = config.replace('  likelihoodInterfaceConfig:',
                                    '  likelihoodInterfaceConfig:\n    externalConstraints: ' + json.dumps(external))
        builder = GUNDAM.ConfigUtils.ConfigBuilder()
        builder.setConfigFromYamlString(config)
        reader = GUNDAM.ConfigUtils.ConfigReader(builder.getConfig())
        reader.defineField(GUNDAM.ConfigUtils.ConfigReader.FieldDefinition('fitterEngineConfig'))
        engine = GUNDAM.FitterEngine()
        engine.configure(reader.fetchValueConfigReader('fitterEngineConfig'))
        engine.getLikelihoodInterface().initialize()
        return engine

    if len(sys.argv) > 1 and sys.argv[1] == '--reject':
        try:
            make_engine({'isEnabled': True, 'constraintsList': [json.loads(sys.argv[2])]})
        except RuntimeError as error:
            assert sys.argv[3] in str(error), str(error)
            return
        raise AssertionError('Accepted invalid constraint')

    def evaluate(engine, positive, negative):
        li = engine.getLikelihoodInterface()
        set_normalization_parameters(li, positive, negative)
        li.propagateAndEvalLikelihood()
        return li.getLastPenaltyLikelihood()

    constraint = {
        'name': 'mixingAmplitude', 'isEnabled': True,
        'parameters': [{'name': 'thetaValue', 'parameterSetName': 'Normalizations',
                        'parameterName': 'Positive_C'}],
        'formula': 'TMath::Sin(2*[thetaValue])^2',
        'type': {'name': 'gaussian', 'mean': 0.87, 'stdDev': 0.02},
    }
    # Reverse alias order relative to the formula, and exercise parameterIndex.
    combined = {
        'name': 'combination',
        'parameters': [
            {'name': 'second', 'parameterSetName': 'Normalizations', 'parameterIndex': 1},
            {'name': 'first', 'parameterSetName': 'Normalizations', 'parameterName': 'Positive_C'},
        ],
        'formula': '[first] - 2*[second]',
        'type': {'name': 'gaussian', 'mean': -4., 'stdDev': 2.},
    }
    disabled = {'name': 'disabled', 'isEnabled': False}
    external = {'isEnabled': True, 'constraintsList': [constraint, combined, disabled]}
    baseline = make_engine()
    engine = make_engine(external)
    li = engine.getLikelihoodInterface()
    manager = li.getModelPropagator().getParametersManager()
    constraints = li.getExternalConstraints().getGaussianConstraintList()
    assert len(constraints) == 3
    for positive, negative in [(2., 3.), (2.4, 3.2), (0.6, 0.8)]:
        internal = evaluate(baseline, positive, negative)
        amplitude = math.sin(2 * positive) ** 2
        residual = (amplitude - .87) / .02
        combination = ((positive - 2 * negative + 4.) / 2.) ** 2
        close(evaluate(engine, positive, negative), internal + residual ** 2 + combination)
        close(constraints[0].evalFormula(), amplitude)
        close(constraints[0].evalResidual(), residual)
        close(li.getExternalConstraints().eval(), residual ** 2 + combination)
        close(li.getLastLikelihood(), li.getLastStatLikelihood() + li.getLastPenaltyLikelihood())

    # Fixed status does not suppress external measurements.
    manager.getParameterSetsList()[0].getParameterList()[0].setIsFixed(True)
    close(constraints[0].eval(), residual ** 2)
    li.setCurrentParameterValuesAsPrior()
    close(constraints[0].eval(), residual ** 2)

    for config in [None, {'constraintsList': [constraint]},
                   {'isEnabled': False, 'constraintsList': [constraint]},
                   {'isEnabled': True, 'constraintsList': [disabled]}]:
        close(evaluate(make_engine(config), 2.4, 3.2), evaluate(baseline, 2.4, 3.2))

    # External constraints also apply with no internally constrained parameters.
    free_engine = make_engine(external, base)
    close(evaluate(free_engine, .6, .8), residual ** 2 + combination)
    penalty_disabled = base.replace('priorType: Flat', 'priorType: Flat\n                isPenaltyDisabled: true', 1)
    close(evaluate(make_engine(external, penalty_disabled), .6, .8), residual ** 2 + combination)

    def rejected(change, expected):
        bad = copy.deepcopy(constraint)
        change(bad)
        # Keep rejection cases independent of ROOT interpreter state and of
        # the engine instances used for numerical checks.
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                 '--reject', json.dumps(bad), expected],
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr

    rejected(lambda c: c['type'].update(stdDev=0), 'stdDev')
    rejected(lambda c: c['type'].update(name='unknown'), 'Unsupported')
    rejected(lambda c: c['parameters'][0].update(parameterSetName='missing'), 'resolve')
    rejected(lambda c: c['parameters'][0].update(parameterName='missing'), 'resolve')
    rejected(lambda c: c['parameters'][0].update(parameterIndex=0), 'exactly one')
    rejected(lambda c: c['parameters'].append(copy.deepcopy(c['parameters'][0])), 'Duplicate')
    rejected(lambda c: c.update(formula='[unknown]'), 'Unused')
    rejected(lambda c: c.update(formula='[thetaValue] + x'), 'declared')
    rejected(lambda c: c.update(formula='sin('), 'Invalid formula')

    singular = copy.deepcopy(constraint)
    singular['formula'] = 'sqrt([thetaValue]-10)'
    close_engine = make_engine({'isEnabled': True, 'constraintsList': [singular]})
    assert math.isinf(evaluate(close_engine, 2., 3.))
    print('SUCCESS: external Gaussian constraints add to the penalty and validate named formula inputs.')


if __name__ == '__main__':
    main()
