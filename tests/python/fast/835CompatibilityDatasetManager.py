#!/usr/bin/env python3
"""Dataset manager configuration, legacy fallback and list precedence."""
import json
import math
from importlib import import_module
from pathlib import Path

import GUNDAM

fixture = import_module('510NumericsIgnoreZeroPredictionAtPrior')


def check_configuration():
    dataset = {'name': 'disabled', 'isEnabled': False, 'model': {}}
    cases = [
        ({'datasetManagerConfig': {'datasetList': [dataset]}}, 1),
        ({'datasetList': [dataset]}, 1),
        ({'propagatorConfig': {'dataSetList': [dataset]}}, 1),
        ({'datasetList': [dataset], 'datasetManagerConfig': {'datasetList': []}}, 0),
        ({'datasetManagerConfig': {'datasetList': [dataset]}, 'datasetList': [dataset, dataset]}, 1),
        ({}, 0),
    ]
    for config, expected in cases:
        engine = GUNDAM.FitterEngine()
        engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(
            {'likelihoodInterfaceConfig': config}))))
        likelihood = engine.getLikelihoodInterface()
        assert len(likelihood.getDatasetList()) == expected
        assert len(likelihood.getDatasetManager().getDatasetList()) == expected


def check_fraction(config_text, first_path):
    second_path = first_path.with_name('835CompatibilityDatasetManager-second.root')
    fixture.write_input_root_file(second_path)
    config_text = config_text.replace('      datasetList:', '      debugLoadDatasetFraction: 0.15\n      datasetList:')
    config_text = config_text.replace(f'              - "{first_path}"', f'''              - "{first_path}"
              - "{second_path}"
          data:
            - name: measurement
              fromModel: true
          selectedDataEntry: measurement''')

    builder = GUNDAM.ConfigUtils.ConfigBuilder()
    builder.setConfigFromYamlString(config_text)
    config = GUNDAM.ConfigUtils.ConfigReader(builder.getConfig())
    config.defineField(GUNDAM.ConfigUtils.ConfigReader.FieldDefinition('fitterEngineConfig'))
    for override, expected_events in ((None, 2), (0.5, 10)):
        engine = GUNDAM.FitterEngine()
        engine.configure(config.fetchValueConfigReader('fitterEngineConfig'))
        likelihood = engine.getLikelihoodInterface()
        if override is not None:
            likelihood.getDatasetManager().setDebugLoadDatasetFraction(fraction=override)
        likelihood.initialize()
        for propagator in (likelihood.getModelPropagator(), likelihood.getDataPropagator()):
            events = propagator.getSampleSet().getSampleList()[0].getEventList()
            assert len(events) == expected_events, f'Expected {expected_events} events, got {len(events)}'


def main():
    check_configuration()
    root_path = Path.cwd() / '835CompatibilityDatasetManager.root'
    fixture.write_input_root_file(root_path)
    legacy = fixture.build_config_text(root_path, True)
    modern = legacy.replace('    dataSetList:\n', '    datasetManagerConfig:\n      datasetList:\n')
    start = modern.index('      - name:')
    end = modern.index('\n    propagatorConfig:', start)
    modern = modern[:start] + '\n'.join('  ' + line if line else line
                                           for line in modern[start:end].split('\n')) + modern[end:]
    old_value = fixture.evaluate_config(legacy, Path.cwd(), 'legacy')
    new_value = fixture.evaluate_config(modern, Path.cwd(), 'manager')
    assert math.isclose(old_value, new_value, rel_tol=1e-12, abs_tol=1e-12)
    check_fraction(modern, root_path)


if __name__ == '__main__':
    main()
