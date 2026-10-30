"""Configuration regression tests; run with python3 tests/cmake/testPythonInterfaces.py.

Requires CMake >= 3.18, pybind11 and CPython development files.
"""
import pathlib
import re
import shutil
import subprocess
import tempfile
import unittest


class PythonInterfacesTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = pathlib.Path(self.temporary.name)
        source = pathlib.Path(__file__).resolve().parents[2] / 'src/PythonInterface'
        options = pathlib.Path(__file__).resolve().parents[2] / 'cmake/options.cmake'
        messages = options.parent / 'utils/cmessage.cmake'
        # Exercise the real options and module configuration without ROOT or
        # compiling the GUNDAM core. GundamFitter is only needed at link time.
        (self.directory / 'CMakeLists.txt').write_text(f'''
cmake_minimum_required(VERSION 3.18)
project(PythonInterfacesTest LANGUAGES CXX)
include("{messages.as_posix()}")
set(WITH_PYTHON_INTERFACE ON CACHE BOOL "")
include("{options.as_posix()}")
add_library(GundamFitter INTERFACE)
set(GUNDAM_FULL_VERSION_STR test)
add_subdirectory("{source.as_posix()}" interfaces)
''')

    def configure(self, selection=None):
        command = ['cmake', '-S', str(self.directory), '-B', str(self.directory / 'build')]
        if selection is not None:
            command.append('-DGUNDAM_PYTHON_VERSIONS=' + selection)
        return subprocess.run(command, capture_output=True, text=True)

    def testAuto(self):
        result = self.configure()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        modules = list((self.directory / 'build/interfaces').glob('python*/cmake_install.cmake'))
        self.assertTrue(modules)
        # Each successful interpreter has a single installation rule.
        installs = [path.read_text() for path in modules]
        targets = [text for text in installs if 'GUNDAM.' in text]
        self.assertTrue(targets)
        selected = re.findall(r'Building GUNDAM_python(\d+_\d+) ', result.stdout)
        self.assertLessEqual(len(selected), 3)
        self.assertEqual(selected, sorted(selected, key=lambda v: tuple(map(int, v.split('_'))), reverse=True))
        versions = self.availableVersions()
        if versions:
            explicit = self.configure(';'.join(versions))
            self.assertEqual(explicit.returncode, 0, explicit.stdout + explicit.stderr)
            expected = [v.replace('.', '_') for v in reversed(versions[-3:])]
            self.assertEqual(selected, expected)

    def testExplicitInterpreterAndDuplicate(self):
        executable = shutil.which('python3')
        self.assertIsNotNone(executable)
        result = self.configure(executable + ';' + executable)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.count('Building GUNDAM_python'), 1)

    def availableVersions(self):
        versions = []
        for minor in range(9, 20):
            version = f'3.{minor}'
            executable = shutil.which('python' + version)
            if executable:
                probe = subprocess.run([executable, '-c',
                    "import pathlib, sysconfig; assert (pathlib.Path(sysconfig.get_path('include')) / 'Python.h').is_file()"],
                    capture_output=True)
                if probe.returncode == 0:
                    versions.append(version)
        return versions

    def testExplicitVersions(self):
        versions = self.availableVersions()
        if len(versions) < 2:
            self.skipTest('Two CPython versions with development files are required')
        result = self.configure(';'.join(versions))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.count('Building GUNDAM_python'), len(versions))

    def testMissingVersion(self):
        result = self.configure('3.999')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Requested Python 3.999 was not found', result.stderr)


if __name__ == '__main__':
    unittest.main()
