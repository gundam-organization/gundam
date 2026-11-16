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


if __name__ == '__main__':
    main()
