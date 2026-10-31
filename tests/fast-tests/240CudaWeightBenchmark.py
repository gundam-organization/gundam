#!/usr/bin/env python3
"""Check CUDA capture boundaries and resident launch commands without a GPU."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
from unittest.mock import patch

sys.dont_write_bytecode = True


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1]/'benchmarks'/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    profiling = load('weight_profiling', '229CudaBackendProfiling.py')
    wrapper = load('weight_benchmark', '236CudaWeightBenchmark.py')
    events = []
    timing = SimpleNamespace(**{field: 0 for field in (*profiling.SECONDS_FIELDS, *profiling.COUNT_FIELDS)})
    manager = SimpleNamespace(getLastTimingSummary=lambda: timing, getLastPreparationSeconds=lambda: 0.)
    likelihood = SimpleNamespace(getLastLikelihood=lambda: 1., getLastStatLikelihood=lambda: 1.)
    engine = SimpleNamespace(evaluateLikelihood=lambda: events.append('evaluate'),
                             getBackendsManager=lambda: manager, getLikelihoodInterface=lambda: likelihood)
    sequence = SimpleNamespace(reset=lambda: events.append('reset'), coordinates=lambda scenario, call: [call],
                               apply=lambda *args: 1)
    args = SimpleNamespace(worker='CUDA-cachedResponses-interleaved', profile_call=1,
        profiler=SimpleNamespace(start=lambda: events.append('start'), stop=lambda: events.append('stop')),
        event_variables=None)
    with patch.object(profiling, 'histogram', return_value=[]), patch.object(profiling, 'outside_histogram', return_value=[]):
        profiling.evaluate_pass(args, engine, sequence, 'local1', 3, 42)
        assert 'start' not in events  # Warmup never opens the gate.
        events.clear()
        result = profiling.evaluate_pass(args, engine, sequence, 'local1', 3, 42, capture=True)
        assert events == ['reset', 'evaluate', 'evaluate', 'start', 'evaluate', 'stop', 'evaluate'], events
        assert [row['hardwareProfiled'] for row in result['rows']] == [False, True, False]
        events.clear()
        def fail_in_capture():
            if events and events[-1] == 'start':
                raise RuntimeError('evaluation failed')
            events.append('evaluate')
        engine.evaluateLikelihood = fail_in_capture
        try:
            profiling.evaluate_pass(args, engine, sequence, 'local1', 3, 42, capture=True)
        except RuntimeError as error:
            assert str(error) == 'evaluation failed'
        else:
            raise AssertionError('Evaluation failure was swallowed')
        assert events[-1] == 'stop'  # Exception must close the gate too.
    with tempfile.TemporaryDirectory(prefix='gundam-weight-cli-') as tmp:
        folder = Path(tmp)
        commands = []
        def run(command, **kwargs):
            commands.append(command)
            if command[0] == 'ncu':
                (folder/'profile'/'weights.ncu-rep').write_bytes(b'test report')
        with patch.object(wrapper.subprocess, 'run', side_effect=run), patch.object(wrapper.shutil, 'which', return_value='/bin/ncu'):
            wrapper.main(['--mode', 'benchmark', '--output', str(folder/'baseline'), '--work-dir', '/inputs', '--config', 'config.yaml'])
            assert commands[-1][commands[-1].index('--execution-mode')+1] == 'resident'
            wrapper.main(['--mode', 'ncu', '--output', str(folder/'profile'), '--work-dir', '/inputs', '--config', 'config.yaml'])
            command = commands[-1]
            assert command[command.index('--replay-mode')+1] == 'kernel'
            assert command[command.index('--profile-from-start')+1] == 'off'
            assert '--profile-capture' in command and '--worker' not in command
            assert json.loads((folder/'profile'/'command.json').read_text()) == command
    print('SUCCESS: capture gates, exception cleanup and resident Nsight commands')


if __name__ == '__main__':
    main()
