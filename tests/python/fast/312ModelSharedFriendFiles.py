#!/usr/bin/env python3
"""Shared friends preserve cuts and weights across files and reader boundaries."""
from array import array
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True


def make_inputs(work):
    import uproot

    inputs = []
    for index, count in enumerate((3, 5, 4)):
        main = work / f'main{index}.root'
        friends = work / f'friends{index}.root'
        with uproot.recreate(main) as output:
            tree = output.mktree('events', {'raw': 'float64'})
            tree.extend({'raw': array('d', range(count))})
            tree = output.mktree('local', {'value': 'float64'})
            tree.extend({'value': array('d', [0.25] * count)})
        with uproot.recreate(friends) as output:
            tree = output.mktree('nested/values', {'value': 'float64'})
            tree.extend({'value': array('d', (10 * index + entry for entry in range(count)))})
            tree = output.mktree('weights', {'offset': 'float64', 'weight': 'float64'})
            tree.extend({'offset': array('d', [0.25] * count),
                         'weight': array('d', [index + 1.] * count)})
        inputs.append({
            'name': f'input{index}',
            'path': str(main) + ':events',
            'friendList': [
                {'name': 'a', 'path': str(friends) + ':nested/values'},
                {'name': 'b', 'path': str(friends) + ':weights'},
                {'name': 'c', 'path': str(main) + ':local'},
            ],
        })
    return inputs


def evaluate(inputs, threads):
    import GUNDAM

    GUNDAM.setNumberOfThreads(threads)
    builder = GUNDAM.ConfigUtils.ConfigBuilder()
    builder.setConfigFromYamlString(json.dumps({
        'likelihoodInterfaceConfig': {
            'jointProbabilityConfig': {'type': 'PoissonLLH'},
            'backendManagerConfig': {'isEnabled': False},
            'datasetManagerConfig': {'datasetList': [{
                'name': 'shared',
                'model': {
                    'filePathList': inputs,
                    'nominalWeightFormula': 'b.weight',
                    'variableDict': [{'name': 'X', 'expr': 'a.value + b.offset + c.value'}],
                },
            }]},
            'propagatorConfig': {'sampleSetConfig': {'sampleList': [{
                'name': 'sample',
                'selectionCutStr': 'a.value != 11',
                'binning': {'binningDefinition': [{'name': 'X', 'edges': [0, 10, 20, 30]}]},
            }]}},
        },
    }))
    engine = GUNDAM.FitterEngine()
    engine.setConfig(GUNDAM.ConfigUtils.ConfigReader(builder.getConfig()))
    engine.configure()
    likelihood = engine.getLikelihoodInterface()
    likelihood.initialize()
    likelihood.propagateAndEvalLikelihood()
    sample = likelihood.getModelPropagator().getSampleSet().getSampleList()[0]
    result = [entry.sumWeights for entry in sample.getHistogram().getBinContentList()]
    assert result == [3., 8., 12.], (threads, result)
    # Readers can start inside a later file; all stored entries must retain
    # their logical chain indices and their local main-tree entry numbers.
    assert len(sample.getEventList()) == 11
    actual = sorted((event.getIndices().entry, event.getIndices().treeFile,
                     event.getIndices().treeEntry) for event in sample.getEventList())
    expected = [(start + local, index, local)
                for index, (start, count) in enumerate(((0, 3), (3, 5), (8, 4)))
                for local in range(count) if (index, local) != (1, 1)]
    assert actual == expected, (threads, actual, expected)
    return result


def main():
    import GUNDAM

    GUNDAM.setRuntimeWorkingDirectory(str(Path.cwd()))
    GUNDAM.setLightOutputMode(True)
    inputs = make_inputs(Path.cwd())
    assert evaluate(inputs, 1) == evaluate(inputs, 4)
    print('SUCCESS: shared friends preserve weighted bins across unequal files with one and four readers.')


if __name__ == '__main__':
    main()
