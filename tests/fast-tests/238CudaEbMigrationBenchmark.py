#!/usr/bin/env python3
"""Exercise the migration benchmark's configuration, replay and reports on CPU."""
import copy
import importlib.util
from importlib import import_module
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import_module('005PythonSetup-utils').requireVenv(__file__)


def main():
    import GUNDAM
    if not hasattr(GUNDAM.FitterEngine, 'getBackendsManager'):
        print('SKIP: backend support was not compiled')
        return
    path = Path(__file__).resolve().parents[1] / 'benchmarks' / '232CudaEbMigrationBenchmark.py'
    spec = importlib.util.spec_from_file_location('eb_benchmark', path)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    helpers = import_module('230CompositeDials')
    with tempfile.TemporaryDirectory(prefix='gundam-eb-benchmark-') as temporary:
        folder = Path(temporary)
        previous = Path.cwd()
        try:
            os.chdir(folder)
            helpers.write_input(folder / 'input.root')
        finally:
            os.chdir(previous)
        formula = helpers.formula('shift', '[arg0]+[arg1]+.001*[arg2]',
            {'parameterName': 'p'}, {'parameterName': 'q'}, {'eventVariable': 'q3'})
        dial = helpers.composite([formula], 'shift', applyToVariable='X', variableResponseMode='Override')
        config = {'fitterEngineConfig': helpers.make_config(folder / 'input.root', dial)}
        original = copy.deepcopy(config)
        hard, coverage = bench.prepare_config(config, 'hard', 'X', .05, 'sample')
        smooth, other = bench.prepare_config(config, 'smooth', 'X', .05, 'sample')
        samples = lambda c: c['fitterEngineConfig']['likelihoodInterfaceConfig']['propagatorConfig']['sampleSetConfig']['sampleList']
        assert not samples(hard)[0]['binningResponse']['isEnabled']
        assert samples(hard)[0]['useDynamicBinning'] is False
        assert samples(smooth)[0]['binningResponse']['variables'] == [{'name': 'X', 'width': .05}]
        assert coverage['configuredDialDefinitions'] == {'Composite': 1, 'Formula': 1}
        assert config == original
        try:
            bench.prepare_config(config, 'smooth', 'X', .05, 'does-not-match')
        except RuntimeError:
            pass
        else:
            raise AssertionError('Empty sample selection was accepted')
        (folder / 'config.json').write_text(json.dumps(config))
        arguments = ['--work-dir', str(folder), '--config', 'config.json']
        configured, inferred = bench.prepare_config(smooth, 'configured', None, None, 'sample')
        assert configured == smooth  # No binning settings are rewritten by default.
        assert inferred['variable'] == 'X' and inferred['hasSmoothedBinning']
        original_response = samples(smooth)[0]['binningResponse']
        original_response['variables'] = [{'name': 'X', 'widthFraction': .05}]
        configured, inferred = bench.prepare_config(smooth, 'configured', None, None, 'sample')
        assert configured == smooth  # Relative widths are preserved too.
        original_response['variables'] = [{'name': 'X', 'width': .05}]
        (folder / 'bin-response.json').write_text(json.dumps({'fitterEngineConfig': {
            'likelihoodInterfaceConfig': {'propagatorConfig': {'sampleSetConfig': {
                'sampleList': [{'name': 'sample', 'binningResponse': original_response}]}}}}}))
        parsed = bench.parse_args(arguments)
        assert parsed.override == []
        assert parsed.parameter_match == '.*'
        assert parsed.variants == ['configured'] and parsed.variable is None and parsed.smoothing_width is None
        assert bench.parse_args(arguments + ['--override', 'first.yaml', 'second.yaml']).override == ['first.yaml', 'second.yaml']
        # Hard binning needs no smoothing width; inputs have no dataset defaults.
        hard_args = arguments + ['--variants', 'hard']
        assert bench.parse_args(hard_args).smoothing_width is None
        default_output = folder / 'configured-results'
        with (folder / 'configured-run.log').open('w') as log:
            completed = subprocess.run([sys.executable, str(path), '--work-dir', str(folder),
                '--config', 'config.json', '--override', 'bin-response.json', '--cases', 'CPU',
                '--calls', '5', '--global-calls', '3', '--trials', '1', '--warmup', '1',
                '--output', str(default_output)], stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            print((folder / 'configured-run.log').read_text())
            raise RuntimeError('Configured benchmark smoke run failed')
        default_summary = json.loads((default_output / 'summary.json').read_text())
        assert default_summary['complete'] and set(default_summary['variants']) == {'configured'}
        configured_dir = default_output / 'configured' / 'CPU'
        effective = json.loads((configured_dir / 'input-config.json').read_text())
        assert samples(effective)[0]['binningResponse'] == original_response
        assert 'useDynamicBinning' not in samples(effective)[0]
        configured_coverage = json.loads((configured_dir / 'coverage.json').read_text())
        assert configured_coverage['variable'] == 'X'
        assert configured_coverage['samples'][0]['smoothed']
        output = folder / 'results'
        with (folder / 'run.log').open('w') as log:
            completed = subprocess.run([sys.executable, str(path), '--work-dir', str(folder),
                '--config', 'config.json', '--override', 'bin-response.json', '--cases', 'CPU',
                '--variants', 'configured', 'hard', 'smooth', '--calls', '5',
                '--global-calls', '3', '--trials', '1', '--warmup', '1', '--output', str(output)],
                stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            print((folder / 'run.log').read_text())
            for native in output.glob('*/CPU/native.log'):
                print(native.read_text()[-6000:])
            raise RuntimeError('CPU benchmark smoke run failed')
        summary = json.loads((output / 'summary.json').read_text())
        assert summary['complete'] and summary['gpuComparisonPerformed'] is False
        assert json.loads((output / 'settings.json').read_text())['override'] == ['bin-response.json']
        assert set(summary['variants']) == {'configured', 'hard', 'smooth'}
        for variant in ('configured', 'hard', 'smooth'):
            assert json.loads((output / variant / 'comparison.json').read_text())['passed']
            directory = output / variant / 'CPU'
            loaded = json.loads((directory / 'coverage.json').read_text())
            assert loaded['samples'][0]['events'] == 6
            assert loaded['samples'][0]['smoothed'] == (variant != 'hard')
            if variant != 'hard':
                assert loaded['hostFractionalProbe']['fractionalEvents'] > 0
            movement = json.loads((directory / 'migration.json').read_text())
            assert movement['global'][0]['eventsWithChangedObservable'] > 0
            assert (directory / 'local1-coordinate-costs.json').is_file()
            counts = summary['variants'][variant]['CPU']['scenarios']['constant']['counts']
            assert counts['cacheCandidateCount']['max'] == 0
            assert counts['rebinnedEventCount']['max'] == 0  # Preserve automatic-dial cache reuse.

        assert 'Smoothing overhead' in (output / 'report.md').read_text()
        # GUNDAM accepts relocated/aliased fields. Inspect them without migrating
        # the user's config, including the mixed layout created by a backend override.
        legacy = copy.deepcopy(config)
        fitter = legacy['fitterEngineConfig']
        fitter['generateSamplePlots'] = True  # Preserve user flags; light output suppresses actual plots.
        likelihood = fitter.pop('likelihoodInterfaceConfig')
        propagator = likelihood.pop('propagatorConfig')
        propagator['fitSampleSetConfig'] = propagator.pop('sampleSetConfig')
        sample_set = propagator['fitSampleSetConfig']
        sample_set['fitSampleList'] = sample_set.pop('sampleList')
        sample_set['fitSampleList'][0]['binningResponse'] = copy.deepcopy(original_response)
        propagator['parameterSetListConfig'] = propagator.pop('parametersManagerConfig')['parameterSetList']
        propagator['dataSetList'] = likelihood.pop('dataSetList')
        joint = likelihood.pop('jointProbabilityConfig')
        propagator['llhStatFunction'] = joint.pop('type')
        propagator['llhConfig'] = joint
        fitter['propagatorConfig'] = propagator
        for with_backend in (False, True):
            source = copy.deepcopy(legacy)
            if with_backend:
                source['fitterEngineConfig']['likelihoodInterfaceConfig'] = {
                    'backendManagerConfig': {'isEnabled': True, 'type': 'CUDA'}}
            prepared, legacy_coverage = bench.prepare_config(source, 'configured', None, None, 'sample')
            assert prepared == source
            assert legacy_coverage['variable'] == 'X'
            name = 'legacy-backend' if with_backend else 'legacy'
            (folder / f'{name}.json').write_text(json.dumps(source))
            legacy_output = folder / f'{name}-results'
            with (folder / f'{name}.log').open('w') as log:
                code = subprocess.run([sys.executable, str(path), '--work-dir', str(folder),
                    '--config', f'{name}.json', '--cases', 'CPU', '--calls', '3',
                    '--global-calls', '3', '--trials', '1', '--warmup', '1',
                    '--output', str(legacy_output)], stdout=log, stderr=subprocess.STDOUT).returncode
            if code:
                print((folder / f'{name}.log').read_text())
                for native in legacy_output.glob('*/CPU/native.log'):
                    print(native.read_text()[-6000:])
                raise RuntimeError(f'{name} benchmark failed')
            resolved = json.loads((legacy_output / 'configured' / 'CPU' / 'resolved-fitter-config.json').read_text())
            expected = copy.deepcopy(source['fitterEngineConfig'])
            expected.setdefault('likelihoodInterfaceConfig', {})['backendManagerConfig'] = {
                'isEnabled': True, 'type': 'CPU', 'backendConfig': {}}
            assert resolved == expected  # Only the backend changes, not physics/initialization settings.

        # Cross-variant comparison must reject different replay coordinates.
        mismatch = output / 'smooth' / 'CPU' / 'global-trial1.json'
        values = json.loads(mismatch.read_text())
        values['rows'][0]['sequenceDigest'] = 'wrong'
        mismatch.write_text(json.dumps(values))
        args = type('Arguments', (), dict(output=output, variants=['hard', 'smooth'],
                                        scenarios=['global'], trials=1))()
        try:
            bench.verify_variant_replay(args)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Different parameter points were accepted')
    print('SUCCESS: migration benchmark validates hard/smooth migration and replay.')


if __name__ == '__main__':
    main()
