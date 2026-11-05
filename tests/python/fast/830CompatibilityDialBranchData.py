#!/usr/bin/env python3
"""Historical dial branch keys produce the same responses as dialBranchData."""
from importlib import import_module
from pathlib import Path
import math
import sys

sys.dont_write_bytecode = True
fixture = import_module('410DialsArrayBranchData')


def main():
    paths = [Path.cwd() / f'830CompatibilityDialBranchData-{i}.root' for i in range(2)]
    for i, path in enumerate(paths):
        fixture.write_input(path, i * 3)
    reference = fixture.evaluate(paths, 'dialBranchData: graph', False)
    for backend in (False, True):
        for config in ('dialLeafName: graph', 'treeExpression: graph'):
            actual = fixture.evaluate(paths, config, backend)
            assert len(actual) == len(reference)
            for result, expected in zip(actual, reference):
                assert len(result) == len(expected)
                assert all(math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-7)
                           for a, b in zip(result, expected)), (config, backend, actual, reference)
    print('SUCCESS: historical dial branch keys preserve histogram sums and errors.')


if __name__ == '__main__':
    main()
