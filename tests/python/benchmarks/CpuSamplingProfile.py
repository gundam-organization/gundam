"""Optional Linux perf sampling after model loading, outside timing trials."""
import os
import shutil
import signal
import subprocess
import sys
import time


def check_profile_support(output, write_json):
    executable = shutil.which('perf') if sys.platform.startswith('linux') else None
    status = {'available': False, 'phase': 'preflight'}
    if executable is None:
        status['reason'] = 'Linux perf executable unavailable in PATH'
    else:
        data = output / 'perf-preflight.data'
        command = [executable, 'record', '-e', 'cycles:u', '-F', '199', '--call-graph', 'dwarf,8192',
                   '-o', str(data), '--', sys.executable, '-c', 'sum(range(1000000))']
        status['command'] = command
        with (output / 'perf-preflight.log').open('w') as log:
            try:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=15)
                status['exitCode'] = result.returncode
                status['available'] = result.returncode == 0
                if not status['available']:
                    status['reason'] = 'perf preflight failed; see perf-preflight.log'
            except (OSError, subprocess.TimeoutExpired) as error:
                status['reason'] = str(error)
        data.unlink(missing_ok=True)
    write_json(output / 'perf-status.json', status)
    return status


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
        stop_requested = False
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
                stop_requested = True
                child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
                status['reason'] = 'perf did not stop promptly'
            status['recordExitCode'] = child.returncode
            # perf can flush its recording and then terminate with the SIGINT
            # we sent. Only accept that signal when we requested the stop.
            stopped_normally = child.returncode == 0 or (
                stop_requested and child.returncode == -signal.SIGINT)
            status['complete'] = (status['calls'] == calls and stopped_normally
                                  and data.exists() and data.stat().st_size > 0)
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
