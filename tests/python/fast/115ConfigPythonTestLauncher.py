#!/usr/bin/env python3
"""Exercise CTest's cross-node Python selection without ROOT or pip installs."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest


class PythonTestLauncher(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='gundam-python-launcher-')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.module = self.directory / 'modules with spaces'
        self.module.mkdir()
        (self.module / 'GUNDAM.py').write_text('value = 42\n')
        self.launcher = Path(__file__).resolve().parents[2] / 'cmake' / 'RunPythonTest.cmake'
        self.selection = self.directory / 'selected.cmake'
        self.runner = self.directory / 'runner.py'
        self.runner.write_text("import sys\nfrom pathlib import Path\nPath(__file__).with_suffix('.args').write_text(repr(sys.argv[1:]))\n")
        self.cmake = shutil.which('cmake')
        self.assertIsNotNone(self.cmake)

    def config(self, candidates, version=None):
        values = dict(testPythonCandidates=';'.join(map(str, candidates)),
                      testPythonVersions=version or f'{sys.version_info.major}.{sys.version_info.minor}',
                      testPythonModuleDirectory=self.module, testPythonSelection=self.selection,
                      testPythonRunner=self.runner, testPythonUseVenv='OFF')
        path = self.directory / 'config.cmake'
        path.write_text('\n'.join(f'set({key} [==[{value}]==])' for key, value in values.items()))
        return path

    def run_launcher(self, config, mode, *arguments):
        return subprocess.run([self.cmake, f'-DTEST_CONFIG={config}', f'-DTEST_MODE={mode}',
                               *arguments, '-P', str(self.launcher)],
                              capture_output=True, text=True, timeout=60,
                              env={**os.environ, 'PATH': str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')})

    def test_missing_preferred_interpreter_falls_back_and_runs_selected_python(self):
        config = self.config([self.directory / 'unmounted-python', sys.executable])
        result = self.run_launcher(config, 'PREPARE')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('Python test fallback:', result.stdout)
        self.assertIn(sys.executable, self.selection.read_text())
        result = self.run_launcher(config, 'RUN', '-DTEST_SUITE=fast', '-DTEST_NAME=script with spaces.py')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.runner.with_suffix('.args').read_text(),
                         repr(['--suite', 'fast', '--test', 'script with spaces.py']))

    def test_node_path_interpreter_can_replace_unmounted_configured_paths(self):
        config = self.config([self.directory / 'unmounted-python'])
        result = self.run_launcher(config, 'PREPARE')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('Python test fallback:', result.stdout)

    def test_rejects_python_versions_without_a_built_interface(self):
        result = self.run_launcher(self.config([sys.executable], version='0.0'), 'PREPARE')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Python version has no interface in this build', result.stderr)
        self.assertFalse(self.selection.exists())

    def test_import_failure_clears_stale_selection_and_reports_dependency(self):
        self.selection.write_text('stale interpreter')
        (self.module / 'GUNDAM.py').write_text("raise ImportError('missing test dependency')\n")
        result = self.run_launcher(self.config([sys.executable]), 'PREPARE')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('missing test dependency', result.stderr)
        self.assertFalse(self.selection.exists())

    def test_failed_test_is_not_hidden_by_launcher(self):
        config = self.config([sys.executable])
        result = self.run_launcher(config, 'PREPARE')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.runner.write_text('raise SystemExit(19)\n')
        result = self.run_launcher(config, 'RUN', '-DTEST_SUITE=fast', '-DTEST_NAME=failed.py')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('failed (19)', result.stderr)


if __name__ == '__main__':
    unittest.main()
