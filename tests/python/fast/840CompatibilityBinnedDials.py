#!/usr/bin/env python3
"""Historical Graph/Spline defaults retain responses and emit deprecation warnings."""
from importlib import import_module
from pathlib import Path
import json
import subprocess
import sys

sys.dont_write_bytecode = True
fixture = import_module('490DialsBinnedBranch')


def main():
    path = Path.cwd() / '840CompatibilityBinnedDials.root'
    fixture.write_input(path)
    warning = 'Implicit binned dial branch selection is deprecated'
    for mode in ('explicit', 'legacy'):
        cases = []
        for kind, branch, scale in (('Graph', 'Graph', .1), ('Spline', 'Spline', .2)):
            source = {'dialsFilePath': str(path), 'dialsTreePath': 'legacy'}
            if mode == 'explicit':
                source['dialsBranch'] = branch
            cases.append((kind, source, scale, None))
        result = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()),
                                 '--exercise', str(path), json.dumps(cases)],
                                capture_output=True, text=True)
        log = result.stdout + result.stderr
        Path(f'840CompatibilityBinnedDials-{mode}.log').write_text(log)
        assert result.returncode == 0, log
        assert len(result.stdout.split('BEGIN_CASE ')) - 1 == len(cases), log
        if mode == 'legacy':
            assert log.count(warning) >= 2, log
        else:
            assert warning not in log, log
    print('SUCCESS: implicit historical branches retain responses and deprecation warnings.')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--exercise':
        fixture.exercise(Path(sys.argv[2]), json.loads(sys.argv[3]))
    else:
        main()
