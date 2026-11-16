#!/usr/bin/env python3
"""Dataset templates merge in order before the dataset's own overrides."""
import json
import math
from importlib import import_module
from pathlib import Path

import GUNDAM

fixture = import_module('510NumericsIgnoreZeroPredictionAtPrior')


def configure(manager):
    engine = GUNDAM.FitterEngine()
    engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps({
        'likelihoodInterfaceConfig': {'datasetManagerConfig': manager}}))))
    return engine


def expect_error(manager):
    try:
        configure(manager)
    except RuntimeError:
        return
    raise AssertionError('Expected invalid template configuration to fail')


def main():
    disabled = {'name': 'input', 'model': {}, 'isEnabled': False}
    expect_error({'datasetTemplateList': [{'name': 'duplicate'}, {'name': 'duplicate'}]})
    expect_error({'datasetTemplateList': [{}]})
    expect_error({'datasetList': [dict(disabled, useTemplates=[{'name': 'missing'}])]})
    engine = configure({'datasetList': [dict(disabled, useTemplates=[
        {'name': 'missing', 'isEnabled': False}])]})
    assert len(engine.getLikelihoodInterface().getDatasetList()) == 1

    path = Path.cwd() / '836CompatibilityDatasetTemplates.root'
    fixture.write_input_root_file(path)
    original = fixture.build_config_text(path, True)
    begin = original.index('    dataSetList:')
    end = original.index('    propagatorConfig:', begin)
    for own_override in (False, True):
        # The second template overrides the first tree. In the second case,
        # the dataset must override the deliberately invalid second tree.
        second_tree = 'missing' if own_override else 'tree_mc'
        manager = f'''    datasetManagerConfig:
      datasetTemplateList:
        - name: base
          model:
            tree: missing
            filePathList: ["{path}"]
        - name: selection
          model:
            tree: {second_tree}
      datasetList:
        - name: TestSample
          useTemplates:
            - name: base
            - name: missing
              isEnabled: false
            - name: selection
'''
        if own_override:
            manager += '          model:\n            tree: tree_mc\n'
        value = fixture.evaluate_config(original[:begin] + manager + original[end:],
                                        Path.cwd(), str(own_override))
        reference = fixture.evaluate_config(original, Path.cwd(), 'reference')
        assert math.isclose(value, reference, rel_tol=1e-12, abs_tol=1e-12)


if __name__ == '__main__':
    main()
