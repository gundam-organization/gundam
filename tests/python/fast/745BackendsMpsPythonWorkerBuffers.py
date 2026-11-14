#!/usr/bin/env python3
"""Check the real PythonWorker host/MPS buffers and backend weight/gradient consumption."""
import json
from pathlib import Path
import sys


CONFIG_YAML = r'''
# The fast test supplies absolute runtime paths and selects host/device buffers.
fitterEngineConfig:
  enablePca: false
  scaleParStepWithChi2Response: false
  generateSamplePlots: false
  generateOneSigmaPlots: false
  minimizerConfig:
    useAnalyticGradient: true
    useNormalizedFitSpace: false
    print_level: 0

  likelihoodInterfaceConfig:
    jointProbabilityConfig:
      type: BarlowBeestonLiteGaussian

    backendManagerConfig:
      isEnabled: true
      type: MPS

    dataSetList:
      - name: BufferSample
        isEnabled: true
        model:
          tree: tree_mc
          nominalWeightFormula: "1.0"
          filePathList: [Events.root]

    propagatorConfig:
      sampleSetConfig:
        sampleList:
          - name: X
            isEnabled: true
            dataSets: [BufferSample]
            binning:
              binningDefinition:
                - name: X
                  edges: [-0.5, 1.0, 10.0]

      parametersManagerConfig:
        parameterSetList:
          - name: AffineWeights
            isEnabled: true
            nominalStepSize: 0.1
            parameterDefinitions:
              - name: scale
                priorValue: 2.0
                priorType: Flat
              - name: offset
                priorValue: 0.25
                priorType: Flat
            dialSetDefinitions:
              - dialType: ExternalWeight
                applyOnDataSets: [BufferSample]
                dialInputList:
                  - name: scale
                  - name: offset
                options:
                  type: PythonWorker
                  inputEventVarList: ["[X]"]
                  workerConfig:
                    # Replaced by the test with its own venv's executable.
                    pythonExecutable: ./WorkerPython
                    evalScript: ./Reweight.py
                    backendBufferMemoryLocation: onDevice
'''


WORKER_SCRIPT = r'''
"""PythonWorker fixture: write affine weights into GUNDAM's existing buffers."""
import json
from pathlib import Path

import numpy as np
import torch


def configure(arguments):
    global auditPath, configuredMode
    auditPath = Path(arguments[0])
    configuredMode = arguments[1] if len(arguments) > 1 else 'ok'


def run(command):
    action = command['command']
    if action == 'shutdown':
        return {'status': 'ok'}
    device = command['bufferDevice']
    mode = configuredMode if device == 'mps' else 'ok'
    arrays = [command['inputs']['X'], command['parameterBuffer'], command['weights']]
    if command.get('withGradient'):
        arrays.append(command['gradients'])
    for array in arrays:
        if device == 'mps':
            if not isinstance(array, torch.Tensor) or array.device.type != 'mps' or array.dtype != torch.float32:
                raise RuntimeError('Expected an existing float32 MPS tensor')
        elif device == 'cpu':
            if not isinstance(array, np.ndarray) or array.dtype != np.float64:
                raise RuntimeError('Expected a float64 NumPy host buffer')
        else:
            raise RuntimeError(f'Unexpected buffer device: {device}')
    if action == 'initialize':
        if [entry['name'] for entry in command['parameterInfo']] != ['scale', 'offset']:
            raise RuntimeError('Unexpected parameter order')
    elif action == 'evaluate':
        x, parameters, weights = map(torch.as_tensor, arrays[:3])
        # No allocation or transfer of the output; Torch writes the supplied view.
        with torch.no_grad():
            torch.mul(x, parameters[0], out=weights)
            weights.add_(parameters[1])
            if command['withGradient']:
                gradients = torch.as_tensor(command['gradients'])
                if tuple(gradients.shape) != (x.numel(), 2):
                    raise RuntimeError('Unexpected gradient shape')
                gradients[:, 0].copy_(x)
                if mode != 'partial':
                    gradients[:, 1].fill_(1.)
                if mode == 'nonfinite':
                    gradients[0, 0] = float('inf')
    else:
        raise RuntimeError(f'Unexpected command: {action}')
    with auditPath.open('a') as stream:
        stream.write(json.dumps({'command': action, 'device': device}) + '\n')
    result = {'status': 'ok', 'supportsGradient': True}
    if command.get('withGradient') and mode != 'missingAck':
        result['gradientGeneration'] = command['generation'] + (mode == 'staleAck')
    return result
'''


def main():
    import GUNDAM
    if sys.platform != 'darwin' or not hasattr(GUNDAM.FitterEngine, 'getBackendsManager'):
        print('SKIP: GUNDAM was built without the MPS backend')
        return 0
    import numpy as np
    import uproot
    import torch
    if not torch.backends.mps.is_available():
        print('SKIP: Torch MPS has no usable Metal device')
        return 77

    work = Path.cwd()
    workerPath = work / 'Reweight.py'
    workerPath.write_text(WORKER_SCRIPT)
    x = np.asarray([0., .25, .5, 1., 2., 4., 8.], dtype=np.float64)
    rootPath = work / 'Events.root'
    with uproot.recreate(rootPath) as file:
        tree = file.mktree('tree_mc', {'X': 'float64'})
        tree.extend({'X': x})
    GUNDAM.setRuntimeWorkingDirectory(str(work))
    GUNDAM.setLightOutputMode(True)
    GUNDAM.setNumberOfThreads(1)

    referenceGradients = []
    for backend, location, expectedDevice in [
            ('CPU', 'onDevice', 'cpu'), ('MPS', 'onHost', 'cpu'), ('MPS', 'onDevice', 'mps')]:
        auditPath = work / f'{backend}-{location}.jsonl'
        auditPath.write_text('')
        builder = GUNDAM.ConfigUtils.ConfigBuilder()
        builder.setConfigFromYamlString(CONFIG_YAML)
        config = json.loads(builder.getConfig().toString())['fitterEngineConfig']
        likelihood = config['likelihoodInterfaceConfig']
        likelihood['backendManagerConfig']['type'] = backend
        likelihood['dataSetList'][0]['model']['filePathList'] = [str(rootPath)]
        worker = likelihood['propagatorConfig']['parametersManagerConfig']['parameterSetList'][0]['dialSetDefinitions'][0]['options']['workerConfig']
        worker.update(pythonExecutable=sys.executable, evalScript=str(workerPath),
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
        points = [(2., .25), (.5, 1.), (.5, 1.), (2., .25)]
        for index, (scale, offset) in enumerate(points):
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
            minimizer = engine.getMinimizer()
            gradient = list(minimizer.evalFitGradientAt([scale, offset]))
            records = [json.loads(line) for line in auditPath.read_text().splitlines()]
            if records[-1] != {'command': 'evaluate', 'device': expectedDevice}:
                raise RuntimeError(f'Gradient worker used the wrong device: {records}')
            if backend == 'CPU':
                referenceGradients.append(gradient)
                for i in range(2):
                    up, down = [scale, offset], [scale, offset]
                    up[i] += 1e-5
                    down[i] -= 1e-5
                    numeric = (minimizer.evalFit(up) - minimizer.evalFit(down)) / 2e-5
                    np.testing.assert_allclose(gradient[i], numeric, rtol=3e-5, atol=3e-6)
            else:
                np.testing.assert_allclose(gradient, referenceGradients[index], rtol=3e-5, atol=3e-6)
            GUNDAM.flushOutput()
        print(f'PASS {backend}/{location}: {expectedDevice} buffers, histogram moments and gradients')
        del minimizer, engine
    # A transport restart must also remap the lazily allocated gradient table.
    engine = GUNDAM.FitterEngine()
    engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
    engine.initialize()
    manager = engine.getBackendsManager()
    for backend in ('MPS', 'CPU', 'MPS'):
        manager.activate(backend)
        gradient = engine.getMinimizer().evalFitGradientAt([.5, 1.])
        np.testing.assert_allclose(gradient, referenceGradients[1], rtol=3e-5, atol=3e-6)
    del engine
    for mode, expected in [('partial', 'missing/non-finite gradient'),
                           ('nonfinite', 'missing/non-finite gradient'),
                           ('missingAck', 'gradientGeneration'), ('staleAck', 'gradientGeneration')]:
        worker['scriptArgs'] = [str(auditPath), mode]
        engine = GUNDAM.FitterEngine()
        engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
        try:
            engine.initialize()
            engine.getMinimizer().evalFitGradientAt([.5, 1.])
        except RuntimeError as error:
            if expected not in str(error):
                raise
        else:
            raise AssertionError(f'Invalid device gradient accepted: {mode}')
        del engine
    return 0


if __name__ == '__main__':
    sys.exit(main())
