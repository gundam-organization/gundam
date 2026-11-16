#!/usr/bin/env python3
"""Shared model/data templates merge before each dispenser's own overrides."""
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
    expect_error({'datasetList': [dict(disabled, model={'useTemplates': [{'name': 'missing'}]})]})
    engine = configure({'datasetList': [dict(disabled, model={'useTemplates': [
        {'name': 'missing', 'isEnabled': False}]})]})
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
          tree: missing
          filePathList: ["{path}"]
        - name: selection
          tree: {second_tree}
      datasetList:
        - name: TestSample
          model:
            useTemplates:
              - name: base
              - name: missing
                isEnabled: false
              - name: selection
          data:
            - name: measurement
              useTemplates:
                - name: base
                - name: selection
              tree: tree_mc
              fromModel: {str(own_override).lower()}
          selectedDataEntry: measurement
'''
        if own_override:
            manager = manager.replace('          data:', '            tree: tree_mc\n          data:')
        value = fixture.evaluate_config(original[:begin] + manager + original[end:],
                                        Path.cwd(), str(own_override))
        reference = fixture.evaluate_config(original, Path.cwd(), 'reference')
        assert math.isclose(value, reference, rel_tol=1e-12, abs_tol=1e-12)
        builder = GUNDAM.ConfigUtils.ConfigBuilder()
        builder.setConfigFromYamlString(original[:begin] + manager + original[end:])
        reader = GUNDAM.ConfigUtils.ConfigReader(builder.getConfig())
        reader.defineField(GUNDAM.ConfigUtils.ConfigReader.FieldDefinition('fitterEngineConfig'))
        engine = GUNDAM.FitterEngine()
        engine.configure(reader.fetchValueConfigReader('fitterEngineConfig'))
        likelihood = engine.getLikelihoodInterface()
        likelihood.setDataType(GUNDAM.LikelihoodInterface.DataType.RealData)
        likelihood.initialize()
        sample = likelihood.getDataPropagator().getSampleSet().getSampleList()[0]
        assert sum(b.sumWeights for b in sample.getHistogram().getBinContentList()) == 10



if __name__ == '__main__':
    main()
