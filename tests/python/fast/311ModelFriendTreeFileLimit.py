#!/usr/bin/env python3
"""Report descriptor exhaustion before ROOT compiles formulas on friend chains."""
from importlib import import_module
from pathlib import Path
import subprocess
import shutil
import sys

sys.dont_write_bytecode = True
fixture = import_module('310ModelFriendTrees')


def worker(mode):
    import resource
    import GUNDAM

    work = Path.cwd()
    GUNDAM.setRuntimeWorkingDirectory(str(work))
    GUNDAM.setLightOutputMode(True)
    GUNDAM.setNumberOfThreads(1 if mode in ('single', 'occupied') else 4)
    friends = ''.join(
        f'                - name: friend{i}\n'
        f'                  path: "{work / ("friends.root" if mode == "shared" else f"friends{i}.root")}:friend_events"\n'
        for i in range(40)
    )
    config = fixture.build_config(work / 'main.root', friends, 'friend0.friendValue')
    if mode == 'capped':
        config = config.replace('name: FriendTreeSample\n', 'name: FriendTreeSample\n        nbMaxThreadsForLoad: 1\n')
    if mode == 'serial':
        config = config.replace(
            'name: FriendTreeSample\n',
            'name: FriendTreeSample\n        devSingleThreadEventSelection: true\n'
            '        devSingleThreadEventLoaderAndIndexer: true\n',
        )
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    limit = 32 if mode == 'shared' else 512 if mode == 'sufficient' else 128
    resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))
    # Keep these descriptors alive while the loader checks its remaining budget.
    occupied = [open('/dev/null') for _ in range(90)] if mode == 'occupied' else []
    fixture.check_histogram(config, [2., 2.])
    if mode in ('insufficient', 'occupied'):
        raise AssertionError('Expected the four readers to exceed the file limit')


def main():
    try:
        import resource
    except ImportError:
        print('SKIP: POSIX file descriptor limits are unavailable.')
        return 77
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard != resource.RLIM_INFINITY and hard < 512:
        print('SKIP: hard descriptor limit is below 512.')
        return 77
    work = Path.cwd()
    fixture.write_root_files(work / 'main.root', work / 'friends.root')
    for index in range(40):
        shutil.copyfile(work / 'friends.root', work / f'friends{index}.root')
    for mode in ('insufficient', 'occupied', 'sufficient', 'shared', 'single', 'capped', 'serial'):
        result = subprocess.run(
            [sys.executable, '-B', str(Path(__file__).resolve()), '--worker', mode],
            capture_output=True, text=True, timeout=60,
        )
        expected_status = 1 if mode in ('insufficient', 'occupied') else 0
        if result.returncode != expected_status:
            raise AssertionError(f'{mode}: exit {result.returncode}\n{result.stdout}\n{result.stderr}')
        if mode in ('insufficient', 'occupied'):
            for expected in ('Open-file limit is insufficient',
                             '1 reader(s)' if mode == 'occupied' else '4 reader(s)',
                             '41 distinct input files', 'ulimit -n', 'nbMaxThreadsForLoad'):
                if expected not in result.stdout + result.stderr:
                    raise AssertionError(f'Missing diagnostic {expected!r}\n{result.stdout}\n{result.stderr}')
    print('SUCCESS: descriptor exhaustion is reported; sufficient and serial readers preserve bins.')
    return 0


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        worker(sys.argv[2])
    else:
        sys.exit(main())
