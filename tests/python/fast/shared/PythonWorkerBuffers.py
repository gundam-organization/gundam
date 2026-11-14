#!/usr/bin/env python3
"""Shared PythonWorker host/device buffer, weight and gradient checks."""
import json
from pathlib import Path
import sys
import time


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
      type: CPU

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
    global auditPath, configuredMode, addresses
    auditPath = Path(arguments[0])
    configuredMode = arguments[1] if len(arguments) > 1 else 'ok'
    addresses = {}


def run(command):
    action = command['command']
    if action == 'shutdown':
        return {'status': 'ok'}
    device = command['bufferDevice']
    mode = configuredMode if device in ('mps', 'cuda') else 'ok'
    arrays = [command['inputs']['X'], command['parameterBuffer'], command['weights']]
    if command.get('withGradient'):
        arrays.append(command['gradients'])
    for index, array in enumerate(arrays):
        if device in ('mps', 'cuda'):
            dtype = torch.float32 if device == 'mps' else torch.float64
            if not isinstance(array, torch.Tensor) or array.device.type != device or array.dtype != dtype:
                raise RuntimeError(f'Expected an existing {dtype} {device} tensor')
        elif device == 'cpu':
            if not isinstance(array, np.ndarray) or array.dtype != np.float64:
                raise RuntimeError('Expected a float64 NumPy host buffer')
        else:
            raise RuntimeError(f'Unexpected buffer device: {device}')
        address = array.data_ptr() if isinstance(array, torch.Tensor) else array.ctypes.data
        if index in addresses and addresses[index] != address:
            raise RuntimeError('A supplied buffer changed address within one worker session')
        addresses[index] = address
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


def run(deviceBackend, propagationMode=None):
    started = time.perf_counter()

    def progress(stage):
        elapsed = time.perf_counter() - started
        print(f'[{deviceBackend}/{propagationMode or "default"} +{elapsed:.1f}s] {stage}', flush=True)

    progress('Starting buffer contract')
    import GUNDAM
    if not hasattr(GUNDAM.FitterEngine, 'getBackendsManager'):
        print('SKIP: GUNDAM was built without backends')
        return 77
    import numpy as np
    import uproot
    import torch
    available = torch.backends.mps.is_available() if deviceBackend == 'MPS' else torch.cuda.is_available()
    if not available:
        raise RuntimeError(f'Torch in {sys.executable} cannot use the compiled {deviceBackend} backend')
    gradientRtol = 3e-5 if deviceBackend == 'MPS' else 1e-8
    gradientAtol = 3e-6 if deviceBackend == 'MPS' else 1e-9

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
            ('CPU', 'onDevice', 'cpu'), (deviceBackend, 'onHost', 'cpu'),
            (deviceBackend, 'onDevice', deviceBackend.lower())]:
        auditPath = work / f'{backend}-{location}.jsonl'
        auditPath.write_text('')
        builder = GUNDAM.ConfigUtils.ConfigBuilder()
        builder.setConfigFromYamlString(CONFIG_YAML)
        config = json.loads(builder.getConfig().toString())['fitterEngineConfig']
        likelihood = config['likelihoodInterfaceConfig']
        likelihood['backendManagerConfig']['type'] = backend
        if backend != 'CPU' and propagationMode is not None:
            likelihood['backendManagerConfig']['options'] = [
                {'name': backend, 'propagationMode': propagationMode}]
        likelihood['dataSetList'][0]['model']['filePathList'] = [str(rootPath)]
        worker = likelihood['propagatorConfig']['parametersManagerConfig']['parameterSetList'][0]['dialSetDefinitions'][0]['options']['workerConfig']
        worker.update(pythonExecutable=sys.executable, evalScript=str(workerPath),
                      backendBufferMemoryLocation=location, scriptArgs=[str(auditPath)])
        progress(f'Initializing {backend}/{location}')
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
            progress(f'{backend}/{location}: weights and gradient at point {index + 1}/{len(points)}')
            parameters[0].setParameterValue(scale)
            parameters[1].setParameterValue(offset)
            engine.evaluateLikelihood()
            records = [json.loads(line) for line in auditPath.read_text().splitlines()]
            if records[-1] != {'command': 'evaluate', 'device': expectedDevice}:
                raise RuntimeError(f'{backend}/{location}: worker did not evaluate on {expectedDevice}: {records}')
            if location == 'onDevice' and backend != 'CPU':
                assert manager.getLastTimingSummary().externalWeightUploadBytes == 0
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
                np.testing.assert_allclose(gradient, referenceGradients[index], rtol=gradientRtol, atol=gradientAtol)
            GUNDAM.flushOutput()
        progress(f'PASS {backend}/{location}: {expectedDevice} buffers, histogram moments and gradients')
        del minimizer, engine
    # A transport restart must also remap the lazily allocated gradient table.
    progress('Initializing backend-switch checks')
    engine = GUNDAM.FitterEngine()
    engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
    engine.initialize()
    manager = engine.getBackendsManager()
    for backend in (deviceBackend, 'CPU', deviceBackend):
        progress(f'Switching to {backend} and evaluating gradient')
        manager.activate(backend)
        gradient = engine.getMinimizer().evalFitGradientAt([.5, 1.])
        np.testing.assert_allclose(gradient, referenceGradients[1], rtol=gradientRtol, atol=gradientAtol)
    del engine
    for mode, expected in [('partial', 'missing/non-finite gradient'),
                           ('nonfinite', 'missing/non-finite gradient'),
                           ('missingAck', 'gradientGeneration'), ('staleAck', 'gradientGeneration')]:
        progress(f'Checking rejection of {mode}')
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
    progress('PASS complete buffer contract')
    return 0
