"""PythonWorker fixture: write affine weights into GUNDAM's existing buffers."""
import json
from pathlib import Path

import numpy as np
import torch


def configure(arguments):
    global auditPath
    auditPath = Path(arguments[0])


def run(command):
    action = command['command']
    if action == 'shutdown':
        return {'status': 'ok'}
    device = command['bufferDevice']
    arrays = [command['inputs']['X'], command['parameterBuffer'], command['weights']]
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
        x, parameters, weights = map(torch.as_tensor, arrays)
        # No allocation or transfer of the output; Torch writes the supplied view.
        with torch.no_grad():
            torch.mul(x, parameters[0], out=weights)
            weights.add_(parameters[1])
    else:
        raise RuntimeError(f'Unexpected command: {action}')
    with auditPath.open('a') as stream:
        stream.write(json.dumps({'command': action, 'device': device}) + '\n')
    return {'status': 'ok', 'supportsGradient': False}
