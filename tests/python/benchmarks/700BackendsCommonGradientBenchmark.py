#!/usr/bin/env python3
"""Quick, deterministic CPU fits comparing numerical and supplied gradients."""
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import time


def write_input(path, events, bins):
    macro = path.with_suffix('.C')
    macro.write_text(r'''#include <TFile.h>
#include <TTree.h>
#include <TGraph.h>
#include <TSystem.h>
#include <cmath>
{ {
 TFile file(PATH, "RECREATE"); TTree tree("events", "events");
 double x, weight; TGraph first, second; TGraph* a=&first; TGraph* b=&second;
 tree.Branch("X", &x, "X/D"); tree.Branch("weight", &weight, "weight/D");
 tree.Branch("graphA", "TGraph", &a, 32000, 0); tree.Branch("graphB", "TGraph", &b, 32000, 0);
 for(int i=0;i<EVENTS;++i) {
  int bin=i%BINS; x=bin+.5; weight=.8+.1*(i%5);
  for(int k=0;k<4;++k) { double p=-3.+2*k;
   a->SetPoint(k,p,1.+(.03+.06*std::sin(bin+.2))*p);
   b->SetPoint(k,p,1.+(.03+.05*std::cos(.3*bin))*p);
  } tree.Fill();
 } tree.Write();
 } gSystem->Exit(0); }
'''.replace('PATH', json.dumps(str(path))).replace('EVENTS', str(events)).replace('BINS', str(bins)))
    subprocess.run(['root', '-n', '-l', '-b', '-q', str(macro)], check=True)


def configuration(path, bins, analytic, strategy):
    pars = [{'name': f'norm{i}', 'priorValue': 1., 'stdDev': .3,
             'parameterLimits': [.2, 2.],
             'dialSetDefinitions': [{'dialType': 'Normalization', 'applyCondition': f'X >= {i} && X < {i+1}'}]}
            for i in range(bins)]
    pars += [{'name': name, 'priorValue': 0., 'stdDev': .8, 'parameterLimits': [-2., 2.],
              'dialSetDefinitions': [{'dialType': 'Graph', 'dialLeafName': leaf}]}
             for name, leaf in [('shapeA', 'graphA'), ('shapeB', 'graphB')]]
    return {'enablePca': False, 'scaleParStepWithChi2Response': False,
            'generateSamplePlots': False, 'generateOneSigmaPlots': False,
            'minimizerConfig': {'useAnalyticGradient': analytic, 'useNormalizedFitSpace': True,
                'enablePostFitErrorFit': False, 'writeLlhHistory': False, 'print_level': 0,
                'strategy': strategy, 'tolerance': 1e-5, 'functionPrecision': 1e-12, 'maxFcnCalls': 10000},
            'likelihoodInterfaceConfig': {'jointProbabilityConfig': {'type': 'BarlowBeestonLiteGaussian'},
                'datasetList': [{'name': 'input', 'model': {'tree': 'events', 'filePathList': [str(path)],
                                                          'nominalWeightFormula': 'weight'}}],
                'propagatorConfig': {'sampleSetConfig': {'sampleList': [{'name': 'spectrum',
                    'binning': {'binningDefinition': [{'name': 'X', 'edges': list(range(bins+1))}]}}]},
                    'parametersManagerConfig': {'parameterSetList': [{'name': 'parameters', 'isEnabled': True,
                        'nominalStepSize': .05, 'parameterDefinitions': pars}]}}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events', type=int, default=12000)
    parser.add_argument('--bins', type=int, default=12)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--strategy', type=int, choices=[0, 1, 2], default=1)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.events, args.bins, args.threads, args.trials) <= 0 or args.events < args.bins:
        parser.error('Positive counts and at least one event per bin are required')
    import GUNDAM
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    input_path = args.output/'input.root'
    write_input(input_path, args.events, args.bins)
    GUNDAM.setLightOutputMode(True)
    GUNDAM.setNumberOfThreads(args.threads)
    report = {'events': args.events, 'parameters': args.bins+2, 'threads': args.threads,
              'strategy': args.strategy, 'module': GUNDAM.__file__, 'complete': False, 'runs': []}
    for trial in range(args.trials):
        for analytic in ([False, True] if trial%2 == 0 else [True, False]):
            engine = GUNDAM.FitterEngine()
            engine.configure(GUNDAM.ConfigUtils.ConfigReader(GUNDAM.JsonType(json.dumps(configuration(input_path, args.bins, analytic, args.strategy)))))
            engine.getLikelihoodInterface().setDataType(GUNDAM.LikelihoodInterface.DataType.Asimov)
            engine.initialize()
            minimizer = engine.getMinimizer()
            point = [.2*math.sin(i+.4) for i in range(args.bins+2)]
            # Equal value/gradient workloads, with all initialization outside the timers.
            minimizer.evalFit(point)
            started = time.perf_counter()
            for _ in range(8): minimizer.evalFit(point)
            value_ms = (time.perf_counter()-started)*1000/8
            started = time.perf_counter()
            for _ in range(8): minimizer.evalFitGradientAt(point)
            gradient_ms = (time.perf_counter()-started)*1000/8
            minimizer.evalFit(point)
            minimizer.updateFitParameterSettings()
            started = time.perf_counter()
            minimizer.minimize()
            elapsed = time.perf_counter()-started
            assert minimizer.hasConverged(), 'MIGRAD did not converge'
            value = engine.getLikelihoodInterface().getLastLikelihood()
            assert math.isfinite(value) and value < 1e-4, value
            run = {'trial': trial, 'analytic': analytic, 'fitSeconds': elapsed,
                   'valueMs': value_ms, 'gradientMs': gradient_ms,
                   'gradientCalls': minimizer.getNbGradientCalls(),
                   'valueCalls': minimizer.getNbFunctionCalls(), 'edm': minimizer.getEdm(), 'likelihood': value,
                   'parameters': [p.getParameterValue() for p in minimizer.getMinimizerFitParameterList()]}
            assert (run['gradientCalls'] > 0) == analytic
            assert math.isfinite(run['edm']) and run['edm'] < 1e-5, run['edm']
            partner = next((r for r in report['runs'] if r['trial'] == trial), None)
            if partner is not None:
                assert all(math.isclose(a, b, rel_tol=1e-4, abs_tol=1e-4)
                           for a, b in zip(run['parameters'], partner['parameters'])), 'Fit parameter mismatch'
            report['runs'].append(run)
            (args.output/'summary.json').write_text(json.dumps(report, indent=2))
            print('RESULT', json.dumps(run), flush=True)
    medians = {str(mode): statistics.median(r['fitSeconds'] for r in report['runs'] if r['analytic']==mode)
               for mode in [False, True]}
    report.update(complete=True, medianFitSeconds=medians, speedup=medians['False']/medians['True'])
    (args.output/'summary.json').write_text(json.dumps(report, indent=2))
    print('SPEEDUP', report['speedup'], flush=True)


if __name__ == '__main__':
    main()
