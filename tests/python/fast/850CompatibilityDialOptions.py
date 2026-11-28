#!/usr/bin/env python3
"""Historical dialConfig options preserve Formula, Composite and compiled responses."""
import copy
from importlib import import_module
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
composites = import_module('470DialsComposite')
formulas = import_module('480DialsEventVarInputForFormula')


def legacy_options(dial):
    result = copy.deepcopy(dial)
    if result['dialType'] == 'Composite':
        result['options']['nodes'] = [legacy_options(node) for node in result['options']['nodes']]
    if result['dialType'] in ('Composite', 'Formula', 'CompiledLibDial'):
        result['dialConfig'] = result.pop('options')
    return result


def main():
    import GUNDAM
    GUNDAM.setLightOutputMode(True)
    path = Path.cwd() / '850CompatibilityDialOptions.root'
    composites.write_input(path)
    first = {'parameterName': 'p'}
    second = {'parameterIndex': 1}
    upstream = {'dialResponse': 'source'}
    dag = composites.composite([
        composites.formula('output', '2+[arg0]+[arg1]+[arg2]',
                           {'dialResponse': 'left'}, {'dialResponse': 'right'}, second),
        composites.formula('right', '[arg0]*[arg0]', upstream),
        composites.formula('source', '[arg0]', first),
        composites.formula('left', '2*[arg0]', upstream)], 'output')
    inner = copy.deepcopy(dag)
    inner['name'] = 'nested'
    nested = composites.composite([inner], 'nested')
    for threads in (1, 2):
        GUNDAM.setNumberOfThreads(threads)
        for dial in (dag, nested):
            for options in (dial, legacy_options(dial)):
                composites.exercise(path, options, lambda p, q, i: max(0., 2+2*p+p*p+q))
    source = Path.cwd() / '850CompatibilityDialOptions.cpp'
    library = source.with_suffix('.so')
    source.write_text('extern "C" double evalVariable(double* x) { return 2+x[0]+3*x[1]; }\n')
    subprocess.run(['c++', '-shared', '-fPIC', str(source), '-o', str(library)], check=True)
    compiled = dict(name='compiled', dialType='CompiledLibDial', dialInputList=[upstream, second],
                    options={'libraryFile': str(library)})
    graph = composites.composite([composites.formula('source', '[arg0]*[arg0]', first), compiled], 'compiled')
    for options in (graph, legacy_options(graph)):
        composites.exercise(path, options, lambda p, q, i: 2+p*p+3*q)
    event_path = Path.cwd() / '850CompatibilityDialOptions-events.root'
    formulas.write_input(event_path)
    mixed = formulas.formula('2+[offset]*[offset]+[slope]*[slope]+.001*([left]+[momentum])', [
        {'name': 'left', 'eventVariable': '[q3]'},
        {'name': 'offset', 'parameterName': 'delta'},
        {'name': 'slope', 'parameterIndex': 1},
        {'name': 'momentum', 'eventVariable': 'q3'}], printDialSummary=True)
    for threads in (1, 2):
        GUNDAM.setNumberOfThreads(threads)
        for options in (mixed, legacy_options(mixed)):
            formulas.exercise(event_path, options, lambda d, a, i: 2+d*d+a*a+.5*i)
    print('SUCCESS: dialConfig and options preserve composite, compiled and event Formula responses.')


if __name__ == '__main__':
    main()
