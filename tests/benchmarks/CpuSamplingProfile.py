"""Optional Linux perf sampling after model loading, outside timing trials."""
import os
import shutil
import signal
import subprocess
import sys
import time


def record_profile(output, evaluate, calls, write_json, progress):
    status_path = output / 'perf-status.json'
    executable = shutil.which('perf') if sys.platform.startswith('linux') else None
    if executable is None:
        status = {'available': False, 'reason': 'Linux perf executable unavailable; stage/counter results retained'}
        write_json(status_path, status)
        progress(status['reason'])
        return
    data = output / 'perf.data'
    command = [executable, 'record', '-e', 'cycles:u', '-F', '199', '--call-graph', 'dwarf,8192',
               '-p', str(os.getpid()), '-o', str(data)]
    status = {'available': True, 'command': command, 'complete': False, 'calls': 0,
              'scope': 'whole resident likelihood evaluations, including Python wrapper; initialization excluded'}
    write_json(status_path, status)
    progress(f'Separate perf sampling pass: {calls} evaluations; main timings already saved')
    with (output / 'perf-record.log').open('w') as log:
        try:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        except OSError as error:
            status.update(available=False, reason=str(error))
            write_json(status_path, status)
            return
        try:
            # Allow perf to attach. Exact boundaries include this short idle
            # startup interval; it is not part of the uninstrumented timings.
            time.sleep(.25)
            if child.poll() is None:
                start = time.monotonic()
                for _ in range(calls):
                    evaluate()
                    status['calls'] += 1
                status['evaluationWallSeconds'] = time.monotonic() - start
        finally:
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
                status['reason'] = 'perf did not stop promptly'
            status['recordExitCode'] = child.returncode
            status['complete'] = status['calls'] == calls and child.returncode == 0
            write_json(status_path, status)
    if status['complete'] and data.exists():
        report = [executable, 'report', '--stdio', '--no-children', '--sort', 'dso,symbol',
                  '--percent-limit', '0.5', '-g', 'none', '-i', str(data)]
        annotate = [executable, 'annotate', '--stdio', '--no-source', '--percent-limit', '1',
                    '-i', str(data)]
        for name, path, arguments in [('report', 'perf-report.txt', report),
                                      ('annotate', 'perf-annotate.txt', annotate)]:
            with (output / path).open('w') as text:
                try:
                    result = subprocess.run(arguments, stdout=text, stderr=subprocess.STDOUT, timeout=30)
                    status[name + 'ExitCode'] = result.returncode
                except (OSError, subprocess.TimeoutExpired) as error:
                    status[name + 'Error'] = str(error)
        write_json(status_path, status)
    progress(f'perf sampling status saved: {status_path}')
