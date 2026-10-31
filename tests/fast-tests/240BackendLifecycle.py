#!/usr/bin/env python3
"""Switch resident backends and the original engine without reopening dynamic inputs."""
import json
import math
import os
from importlib import import_module
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import_module('005PythonSetup-utils').requireVenv(__file__)
fixture = import_module('219VariableDials')
capture = import_module('220CpuDynamicBinning').capture


def write_input(path):
    macro = path.with_suffix('.C')
    macro.write_text('''#include <TFile.h>
#include <TTree.h>
#include <TSystem.h>
{
 {
 TFile output(OUTPUT_PATH, "RECREATE");
 TTree tree("events", "events");
 double raw, Y = .5, weight;
 int aux, category;
 tree.Branch("raw", &raw, "raw/D");
 tree.Branch("Y", &Y, "Y/D");
 tree.Branch("weight", &weight, "weight/D");
 tree.Branch("aux", &aux, "aux/I");
 tree.Branch("category", &category, "category/I");
 for (int i = 0; i < 4; ++i) {
  raw = -.25 + i; weight = i + 1; aux = 10 + i; category = i % 2;
  tree.Fill();
 }
 tree.Write();
 }
 gSystem->Exit(0);
}
'''.replace('OUTPUT_PATH', json.dumps(str(path))))
    subprocess.run(['root', '-l', '-b', '-q', str(macro)], check=True)


def assert_same(actual, expected):
    assert len(actual) == len(expected)
    assert all(math.isclose(a, b, rel_tol=2e-5, abs_tol=2e-5)
               for a, b in zip(actual, expected)), (actual, expected)


def exercise(probability):
    import GUNDAM
    if not hasattr(GUNDAM.FitterEngine, 'getBackendsManager'):
        print('SKIP: backend support was not compiled')
        return
    GUNDAM.setLightOutputMode(True)
    GUNDAM.setNumberOfThreads(2)
    path = Path.cwd() / '240BackendLifecycle.root'
    write_input(path)
    reference = fixture.make_engine(path, explicit_dynamic=True, probability=probability)
    engine = fixture.make_engine(path, explicit_dynamic=True, backend='CPU', probability=probability)
    # A manager initialized without an active backend can be activated later.
    reference_manager = reference.getBackendsManager()
    assert reference_manager.getActiveBackendName() == ''
    assert 'CPU' in reference_manager.getAvailableBackendNames()
    reference_manager.activate('CPU')
    reference_manager.deactivateBackend()
    manager = engine.getBackendsManager()
    manager.setEnableAutoMaterialize(False)
    assert manager.getActiveBackendName() == 'CPU'
    path.unlink()  # Every following operation must use the already-loaded events and dials.
    transitions = [('CPU', None), ('', None), ('CPU', None)]
    if sys.platform == 'darwin':
        transitions += [('MPS', {'propagationMode': 'recomputeResponses'}), ('CPU', None),
                        ('MPS', None), ('', None), ('MPS', {'propagationMode': 'cachedResponses'})]
    if os.environ.get('GUNDAM_TEST_CUDA') == '1':
        transitions += [('CUDA', {'propagationMode': 'recomputeResponses'}), ('CPU', None),
                        ('CUDA', None), ('', None), ('CUDA', {'propagationMode': 'cachedResponses'})]
    transitions += [('CPU', None), ('', None), ('CPU', None)]
    points = [(1., .5, 1.25, 2.), (2., -.25, -.5, 1.5), (.5, .1, 1.5, 3.)]
    for index, (name, config) in enumerate(transitions):
        if name:
            if config is None:
                manager.activate(name)
            else:
                manager.activate(name, GUNDAM.JsonType(json.dumps(config)))
        else:
            manager.deactivateBackend()
            manager.deactivateBackend()  # Idempotent, also after an evaluation in the original engine.
        assert manager.getActiveBackendName() == name
        assert manager.hasBackend() == bool(name)
        for current in (reference, engine):
            propagator = current.getLikelihoodInterface().getModelPropagator()
            parameters = {p.getName(): p for p in propagator.getParametersManager()
                          .getParameterSetsList()[0].getParameterList()}
            for key, value in zip(('multiply', 'add', 'override', 'weight'), points[index % len(points)]):
                parameters[key].setParameterValue(value)
            # An explicitly dynamic coordinate is independent of the variable responses.
            sample = propagator.getSampleSet().getSampleList()[2]
            sample.getEventList()[0].getVariables().fetchVariable('Y').setValue(2. if index % 2 else .5)
            current.evaluateLikelihood()
        if name in ('MPS', 'CUDA') and probability is not None:
            timing = manager.getLastTimingSummary()
            assert timing.likelihoodReadbackBytes == 8
            assert timing.eventStateReadbackBytes == 0
            assert timing.eventWeightReadbackBytes == 0
            assert timing.histogramReadbackBytes == 0
        # Deactivation itself must materialize state, weights and histograms, even with auto materialization off.
        manager.deactivateBackend()
        assert manager.getActiveBackendName() == ''
        assert_same(capture(engine), capture(reference))
        engine.evaluateLikelihood()  # Original engine must reproduce the last resident result.
        assert_same(capture(engine), capture(reference))
        if name:
            manager.activate(name)  # Stored overrides survive unload/reload.
            engine.evaluateLikelihood()
            for output in ('EventState', 'EventWeights', 'Histograms'):
                manager.materialize(output)
            assert_same(capture(engine), capture(reference))
        try:
            manager.activate('NotABackend')
        except RuntimeError:
            pass
        else:
            raise AssertionError('Unknown backend activation must fail')
        assert manager.getActiveBackendName() == name
    manager.deactivateBackend()
    print(f'SUCCESS: {probability or "LeastSquares"} lifecycle preserves observables, binning, weights, errors and LLH without disk reads', flush=True)


def main():
    for probability in (None, 'PoissonLLH', 'BarlowLLH_BANFF_OA2021'):
        exercise(probability)


if __name__ == '__main__':
    main()
