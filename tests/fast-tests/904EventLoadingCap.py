#!/usr/bin/env python3
"""An early event-loading stop must release the reader/filler handshake."""
from importlib import import_module
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
if '--worker' not in sys.argv:
    import_module('005PythonSetup-utils').requireVenv(__file__)


def worker(path, threads):
    import GUNDAM
    GUNDAM.setLightOutputMode(True)
    GUNDAM.setNumberOfThreads(threads)
    config = {'generateSamplePlots': False, 'enablePca': False,
              'scaleParStepWithChi2Response': False,
              'likelihoodInterfaceConfig': {
                  'jointProbabilityConfig': {'type': 'PoissonLLH'},
                  'datasetList': [{'name': 'input', 'model': {
                      'tree': 'events', 'filePathList': [str(path)],
                      'debugNbMaxEventsToLoad': 1}}],
                  'propagatorConfig': {
                      'sampleSetConfig': {'sampleList': [{'name': 'sample', 'binning': {
                          'binningDefinition': [{'name': 'X', 'edges': [0, 1]}]}}]},
                      'parametersManagerConfig': {'parameterSetList': []}}}}
    # Repeat to exercise different reader/filler schedules.
    for _ in range(8):
        engine = GUNDAM.FitterEngine()
        engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(config))))
        engine.getLikelihoodInterface().setDataType(GUNDAM.LikelihoodInterface.DataType.Asimov)
        engine.initialize()
        samples = engine.getLikelihoodInterface().getModelPropagator().getSampleSet().getSampleList()
        count = sum(len(s.getEventList()) for s in samples)
        # The cap is checked concurrently after insertion, so workers can overshoot.
        assert 1 <= count <= threads, count


def main():
    path = Path.cwd() / '904EventLoadingCap.root'
    macro = path.with_suffix('.C')
    macro.write_text('''#include <TFile.h>
#include <TTree.h>
#include <TSystem.h>
{ { TFile file(INPUT, "RECREATE"); TTree tree("events", "events");
double x=.5; tree.Branch("X", &x, "X/D");
for(int i=0;i<4000;++i) tree.Fill(); tree.Write(); } gSystem->Exit(0); }
'''.replace('INPUT', json.dumps(str(path))))
    subprocess.run(['root', '-n', '-l', '-b', '-q', str(macro)], check=True)
    for threads in (1, 4):
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(path), str(threads)],
                       check=True, timeout=30)
    print('SUCCESS: early loading cap releases readers with 1 and 4 threads')


if __name__ == '__main__':
    if '--worker' in sys.argv:
        worker(Path(sys.argv[2]), int(sys.argv[3]))
    else:
        main()
