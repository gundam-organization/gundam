#!/usr/bin/env python3
"""Regression checks for Python runner and CTest pass/fail/skip propagation."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

TESTS = Path(__file__).resolve().parent


class RunnerStatus(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='gundam-runner-status-')
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.fast = self.folder / 'fast'
        self.fast.mkdir()
        shutil.copy(TESTS / 'run.py', self.folder / 'run.py')
        # Deterministic absent-device probe; these checks need neither GUNDAM nor a GPU.
        (self.fast / 'BackendTestSupport.py').write_text('def mps_available(): return False\n')
        (self.folder / 'GUNDAM.py').write_text('class FitterEngine: getBackendsManager = None\n')

    def run_script(self, code, name='100Status.py'):
        (self.fast / name).write_text(code)
        return subprocess.run([sys.executable, '-B', str(self.folder / 'run.py'),
                               '--test', name], capture_output=True, text=True)

    def test_pass(self):
        result = self.run_script('pass\n')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('PASS: 100Status.py', result.stdout)

    def test_failure(self):
        result = self.run_script('raise RuntimeError("numerical mismatch")\n')
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('numerical mismatch', result.stdout)

    def test_explicit_skip(self):
        result = self.run_script('raise SystemExit(77)\n')
        self.assertEqual(result.returncode, 77, result.stdout + result.stderr)
        self.assertNotIn('PASS:', result.stdout)

    def test_no_device_does_not_run_test(self):
        result = self.run_script('raise RuntimeError("must not run")\n', '740Device.py')
        self.assertEqual(result.returncode, 77, result.stdout + result.stderr)
        self.assertIn('no usable Metal device', result.stdout)
        self.assertNotIn('must not run', result.stdout)

    def test_cuda_launcher_statuses(self):
        launcher = self.folder / 'cuda-runner'
        (self.fast / '760Device.py').write_text(
            "import os; assert os.environ['GUNDAM_TEST_CUDA'] == '1'\n")
        for code in ('exit 77', 'exit 1', 'exec "$@"'):
            with self.subTest(launcher=code):
                launcher.write_text('#!/bin/sh\n' + code + '\n')
                launcher.chmod(0o755)
                result = subprocess.run([sys.executable, '-B', str(self.folder / 'run.py'),
                    '--cuda-runner', str(launcher), '--test', '760Device.py'],
                    capture_output=True, text=True)
                self.assertEqual(result.returncode, {'exit 77': 77, 'exit 1': 1,
                    'exec "$@"': 0}[code], result.stdout + result.stderr)

    def test_ctest_statuses(self):
        (self.folder / 'selection.cmake').write_text(f'set(selectedPython "{sys.executable}")\n')
        (self.folder / 'config.cmake').write_text(
            f'set(testPythonSelection "{self.folder}/selection.cmake")\n'
            f'set(testPythonRunner "{self.folder}/run.py")\n')
        lines = ['cmake_minimum_required(VERSION 3.18)', 'project(Status NONE)', 'enable_testing()']
        for name, code in [('100Pass', 'pass'), ('101Fail', 'raise SystemExit(1)'),
                           ('102Skip', 'raise SystemExit(77)')]:
            (self.fast / (name + '.py')).write_text(code + '\n')
            lines.append(f'add_test(NAME {name} COMMAND "${{CMAKE_COMMAND}}" '
                         f'-DTEST_CONFIG={self.folder}/config.cmake -DTEST_MODE=RUN '
                         f'-DTEST_SUITE=fast -DTEST_NAME={name}.py '
                         f'-P "{TESTS.parent}/cmake/RunPythonTest.cmake")')
            lines.append(f'set_tests_properties({name} PROPERTIES '
                         'SKIP_REGULAR_EXPRESSION "GUNDAM_PYTHON_TEST_SKIPPED:")')
        (self.folder / 'CMakeLists.txt').write_text('\n'.join(lines))
        subprocess.run(['cmake', '-S', str(self.folder), '-B', str(self.folder / 'build')],
                       check=True, capture_output=True)
        result = subprocess.run(['ctest', '--test-dir', str(self.folder / 'build')],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        for status in ('Passed', 'Failed', 'Skipped'):
            self.assertIn(status, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
