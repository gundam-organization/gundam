#!/usr/bin/env python3
"""Exercise the V100 benchmark matrix and stat-only comparisons without a GPU."""
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
    path = Path(__file__).resolve().parents[1]/'benchmarks'/'235CudaV100LikelihoodBenchmark.py'
    spec = importlib.util.spec_from_file_location('v100_benchmark', path)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    helpers = import_module('230CompositeDials')
    with tempfile.TemporaryDirectory(prefix='gundam-v100-benchmark-') as tmp:
        folder = Path(tmp)
        previous = Path.cwd()
        try:
            os.chdir(folder)
            helpers.write_input(folder/'input.root')
        finally:
            os.chdir(previous)
        dial = helpers.formula('weight', '1+.1*[arg0]+.02*[arg1]',
                               {'parameterName': 'p'}, {'parameterName': 'q'})
        config = {'fitterEngineConfig': helpers.make_config(folder/'input.root', dial)}
        (folder/'config.json').write_text(json.dumps(config))
        args = bench.parse_args(['--work-dir', str(folder), '--config', 'config.json'])
        assert args.block_sizes == ['auto','256'] and args.reuse_modes == ['on']
        assert args.likelihoods == ['configured'] and args.execution_mode == 'isolated'
        assert not args.allow_other_gpu
        command = bench.command(args, 'PoissonLLH', 'auto', 'off', folder/'planned')
        assert '--require-device-likelihood' in command and '--require-v100' in command
        assert command[command.index('--likelihood-block-size')+1] == '0'
        assert '--disable-prediction-reuse' in command
        args.worker = 'CUDA-cachedResponses-interleaved'
        args.likelihood_block_size = 64
        args.disable_prediction_reuse = True
        config = bench.profiling.backend_config(args)
        assert config['backendConfig']['likelihoodBlockSize'] == 64
        assert not config['backendConfig']['reusePrediction']
        output = folder/'results'
        with (folder/'run.log').open('w') as log:
            result = subprocess.run([sys.executable, str(path), '--work-dir', str(folder),
                '--config', 'config.json', '--cases', 'CPU', '--block-sizes', 'auto','64',
                '--reuse-modes', 'on','off', '--likelihoods', 'PoissonLLH','BarlowLLH_BANFF_OA2021',
                '--calls', '3', '--global-calls', '3', '--trials', '1', '--warmup', '1',
                '--output', str(output)], stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            print((folder/'run.log').read_text())
            raise RuntimeError('V100 benchmark CPU smoke run failed')
        summary = json.loads((output/'summary.json').read_text())
        assert summary['complete'] and len(summary['experiments']) == 8
        assert (output/'report.md').is_file()
        reference = output/summary['experiments'][0]['directory']/'CPU'
        candidate = output/summary['experiments'][1]['directory']
        rows = candidate/'CPU'/'constant-trial1.json'
        measurement = json.loads(rows.read_text())
        measurement['rows'][0]['statLikelihood'] += 1.  # Total unchanged: must still fail.
        rows.write_text(json.dumps(measurement))
        try:
            bench.profiling.compare(candidate,['CPU'],['constant'],1,1e-9,1e-7,
                reference_directory=reference,comparison_name='negative-stat-comparison.json')
        except RuntimeError:
            pass
        else:
            raise AssertionError('A stat-only mismatch was accepted')
    print('SUCCESS: V100 benchmark CLI, matrix replay and stat-only validation pass on CPU.')


if __name__ == '__main__':
    main()
