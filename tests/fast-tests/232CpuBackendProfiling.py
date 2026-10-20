#!/usr/bin/env python3
"""Check sampled CPU diagnostics, cache accounting and unchanged predictions."""
import json
import os
import sys
from importlib import import_module
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import_module('005PythonSetup-utils').requireVenv(__file__)


def run(path, profile):
    import GUNDAM
    helper = import_module('211CpuBackendPropagation')
    output = profile.with_name(f'{profile.stem}-{os.getpid()}{profile.suffix}') if profile is not None else None
    prefix = output.read_text().splitlines() if output is not None and output.exists() else []
    text = helper.build_config_text(path, True)
    if profile is not None:
        text = text.replace('      type: CPU', f'''      type: CPU
      backendConfig:
        profileOutputPath: "{profile}"
        profileSampleInterval: 3
        profileMaxSamples: 8''')
    else:
        text = text.replace('      type: CPU', '''      type: CPU
      backendConfig:
        weightScheduling: Fixed''')
    builder = GUNDAM.ConfigUtils.ConfigBuilder()
    builder.setConfigFromYamlString(text)
    reader = GUNDAM.ConfigUtils.ConfigReader(builder.getConfig())
    reader.defineField(GUNDAM.ConfigUtils.ConfigReader.FieldDefinition('fitterEngineConfig'))
    engine = GUNDAM.FitterEngine()
    engine.configure(reader.fetchValueConfigReader('fitterEngineConfig'))
    engine.initialize()
    li = engine.getLikelihoodInterface()
    predictions = []
    for index in range(24):
        # Alternating changed and unchanged inputs exercises empty cache jobs.
        helper.set_normalization_parameters(li, 2. + (index // 2) * .1, 3.)
        engine.evaluateLikelihood()
        bins = li.getModelPropagator().getSampleSet().getSampleList()[0].getHistogram().getBinContentList()
        predictions.append([(v.sumWeights, v.sqrtSumSqWeights) for v in bins])
    if profile is not None:
        # Records must be readable while the engine and output stream are alive.
        lines = output.read_text().splitlines()
        assert lines[:len(prefix)] == prefix
        records = [json.loads(line) for line in lines[len(prefix):]]
        metadata = records[0]
        samples = records[1:]
        assert metadata['kind'] == 'metadata'
        assert metadata['outputPath'] == str(output)
        assert metadata['configuredOutputPath'] == str(profile)
        assert metadata['system']['pid'] == os.getpid()
        assert metadata['threads'] == 4
        assert metadata['events'] == 18
        assert metadata['weightScheduling'] == 'Dynamic'
        assert metadata['weightRangesRole'] == 'fixedReference'
        assert metadata['weightBlockCount'] == metadata['dynamicBlockTarget'] == 8
        assert metadata['weightBlocks'][0]['begin'] == 0
        assert metadata['weightBlocks'][-1]['end'] == 18
        assert all(a['end'] == b['begin'] for a, b in zip(
            metadata['weightBlocks'], metadata['weightBlocks'][1:]))
        assert sum(b['dialReferences'] for b in metadata['weightBlocks']) == metadata['eventDialReferences']
        assert sum(b['events'] for b in metadata['weightBlocks']) == 18
        assert all(b['events'] > 0 for b in metadata['weightBlocks'])
        assert sum(r['events'] for r in metadata['weightRanges']) == 18
        assert sum(metadata['referencedDialTypes']) == metadata['eventDialReferences']
        assert [r['call'] for r in samples] == [1, 2, 3, 4, 5, 6, 9, 12]
        assert len({r['runId'] for r in records}) == 1
        assert any(s['dirtyCachedDials'] == 0 for s in samples)
        assert any(s['dirtyCachedDials'] > 0 for s in samples)
        for sample in samples:
            assert sample['process']['wallSeconds'] >= 0
            stages = {s['name']: s for s in sample['stages']}
            cache = stages['cache']['workers']
            assert sum(w['evaluatedDials'] + w['unchangedInputs'] for w in cache) == sample['dirtyCachedDials']
            for stage in stages.values():
                assert stage['wallSeconds'] >= 0
                for worker in stage['workers']:
                    assert worker['endOffsetSeconds'] >= worker['startOffsetSeconds'] >= 0
                    if stage['name'] == 'cache':
                        assert sum(worker['evaluatedDialTypes']) == worker['evaluatedDials']
            assert len(stages['weights']['workers']) == 4
            assert sum(w['weightEvents'] for w in stages['weights']['workers']) == 18
            assert sum(w['weightBlocks'] for w in stages['weights']['workers']) == 8
            assert len(stages['histograms']['workers']) == 4
    return predictions


def main():
    import GUNDAM
    if not hasattr(GUNDAM.FitterEngine, 'getBackendsManager'):
        print('SKIP: GUNDAM was built without backends')
        return
    GUNDAM.setNumberOfThreads(4)
    GUNDAM.setLightOutputMode(True)
    root = Path.cwd() / '232CpuBackendProfiling-input.root'
    profile = Path.cwd() / '232CpuBackendProfiling.jsonl'
    profile.with_name(f'{profile.stem}-{os.getpid()}{profile.suffix}').unlink(missing_ok=True)
    profile.write_text('Existing unsuffixed report must be preserved.\n')
    import numpy as np
    import uproot
    with uproot.recreate(root) as saved:
        tree = saved.mktree('tree_mc', {'X': 'float64', 'C': 'float64'})
        tree.extend({'X': np.tile([-.5, -.4, -.3, .2, .3, .4], 3),
                     'C': np.tile([1., -1., 1., -1., 1., -1.], 3)})
    expected = run(root, None)
    assert expected == run(root, profile)
    assert expected == run(root, profile)
    assert profile.read_text() == 'Existing unsuffixed report must be preserved.\n'
    print('SUCCESS: CPU profiling preserves predictions and exports bounded, flushed cache/worker diagnostics.')


if __name__ == '__main__':
    main()
