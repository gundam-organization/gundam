#!/usr/bin/env python3
"""Check the real PythonWorker host/MPS buffers and backend weight consumption."""
import json
from pathlib import Path
import sys


def main():
    import GUNDAM
    if sys.platform != 'darwin' or not hasattr(GUNDAM.FitterEngine, 'getBackendsManager'):
        print('SKIP: GUNDAM was built without the MPS backend')
        return 0
    import numpy as np
    import uproot
    import torch
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS PythonWorker test requires a usable Metal device and Torch MPS')

    fixtures = Path(__file__).resolve().parent / 'inputs' / 'PythonWorkerBuffers'
    work = Path.cwd()
    x = np.asarray([0., .25, .5, 1., 2., 4., 8.], dtype=np.float64)
    rootPath = work / 'Events.root'
    with uproot.recreate(rootPath) as file:
        tree = file.mktree('tree_mc', {'X': 'float64'})
        tree.extend({'X': x})
    GUNDAM.setRuntimeWorkingDirectory(str(work))
    GUNDAM.setLightOutputMode(True)
    GUNDAM.setNumberOfThreads(1)

    for backend, location, expectedDevice in [
            ('MPS', 'onDevice', 'mps'), ('MPS', 'onHost', 'cpu'), ('CPU', 'onDevice', 'cpu')]:
        auditPath = work / f'{backend}-{location}.jsonl'
        auditPath.write_text('')
        builder = GUNDAM.ConfigUtils.ConfigBuilder()
        builder.setConfigFromYamlString((fixtures / 'Config.yaml').read_text())
        config = json.loads(builder.getConfig().toString())['fitterEngineConfig']
        likelihood = config['likelihoodInterfaceConfig']
        likelihood['backendManagerConfig']['type'] = backend
        likelihood['dataSetList'][0]['model']['filePathList'] = [str(rootPath)]
        worker = likelihood['propagatorConfig']['parametersManagerConfig']['parameterSetList'][0]['dialSetDefinitions'][0]['options']['workerConfig']
        worker.update(pythonExecutable=sys.executable, evalScript=str(fixtures / 'Reweight.py'),
                      backendBufferMemoryLocation=location, scriptArgs=[str(auditPath)])
        engine = GUNDAM.FitterEngine()
        engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
        engine.initialize()
        manager = engine.getBackendsManager()
        if not manager.hasBackend() or manager.getType() != backend:
            raise RuntimeError(f'Requested backend {backend} was not activated')
        interface = engine.getLikelihoodInterface()
        propagator = interface.getModelPropagator()
        parameters = propagator.getParametersManager().getParameterSetsList()[0].getParameterList()
        # Check initial activation, updates, a repeated point and a return to prior.
        for scale, offset in [(2., .25), (.5, 1.), (.5, 1.), (2., .25)]:
            parameters[0].setParameterValue(scale)
            parameters[1].setParameterValue(offset)
            engine.evaluateLikelihood()
            records = [json.loads(line) for line in auditPath.read_text().splitlines()]
            if records[-1] != {'command': 'evaluate', 'device': expectedDevice}:
                raise RuntimeError(f'{backend}/{location}: worker did not evaluate on {expectedDevice}: {records}')
            bins = propagator.getSampleSet().getSampleList()[0].getHistogram().getBinContentList()
            weights = scale * x + offset
            masks = [x < 1., x >= 1.]
            np.testing.assert_allclose([b.sumWeights for b in bins],
                                       [weights[mask].sum() for mask in masks], rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose([b.sqrtSumSqWeights**2 for b in bins],
                                       [(weights[mask]**2).sum() for mask in masks], rtol=1e-6, atol=1e-6)
            GUNDAM.flushOutput()
        print(f'PASS {backend}/{location}: {expectedDevice} buffers and analytic histogram moments')
        del engine
    return 0


if __name__ == '__main__':
    sys.exit(main())
