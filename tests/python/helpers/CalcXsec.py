"""I/O and assertions for executable CalcXsec tests (no configuration defaults)."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import uproot


def executable(name):
    override = os.environ.get('GUNDAM_' + ('CALC_XSEC' if name == 'gundamCalcXsec' else 'FITTER'))
    if override:
        path = Path(override).resolve()
        if not path.is_file():
            raise RuntimeError(f'Executable does not exist: {path}')
        return str(path)
    # Prefer the application built alongside the imported interface, not an
    # unrelated installation earlier on PATH. CTest supplies explicit paths.
    import GUNDAM
    path = Path(GUNDAM.__file__).resolve().parent.parent / 'Applications' / name
    if path.is_file():
        return str(path)
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f'{name} not found; set GUNDAM_CALC_XSEC / GUNDAM_FITTER')
    return path


def run(command, label, error=None):
    log = Path(label + '.log')
    environment = os.environ.copy()
    for key in ('DYLD_LIBRARY_PATH', 'LD_LIBRARY_PATH'):
        environment.pop(key, None)
    with log.open('w') as output:
        result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT,
                                env=environment, timeout=180)
    if error is None:
        if result.returncode:
            raise AssertionError(f'{command} failed ({result.returncode}):\n{log.read_text()[-16000:]}')
    elif result.returncode == 0 or error not in log.read_text():
        raise AssertionError(f'Expected failure containing {error!r}:\n{log.read_text()[-8000:]}')
    return log


def write_config(label, config):
    path = Path(label + '.json').resolve()
    path.write_text(json.dumps(config, indent=2) + '\n')
    return path


def calculate(label, fitter, config, n=32, seed=12345, options=(), error=None, threads=1):
    config_path = write_config(label, config)
    output = Path(label + '.root').resolve()
    run([executable('gundamCalcXsec'), '-c', str(config_path), '-f', str(fitter),
         '-n', str(n), '-s', str(seed), '-t', str(threads), '-o', str(output), *options], label, error)
    return output


def write_events():
    """Sums 120 and 240; third bin empty. A second category gives a second sample."""
    path = Path('events.root').resolve()
    with uproot.recreate(path) as output:
        tree = output.mktree('events', {key: 'float64' for key in ('x', 'y', 'category', 'weight')})
        tree.extend({'x': np.array([1., 1., 4., 4.] * 2),
                     'y': np.full(8, 1.5), 'category': np.repeat([0., 1.], 4),
                     'weight': np.array([40., 80., 80., 160.] * 2)})
    return path


def write_covariance():
    """ROOT stores this as TMatrixDSym, the native ParameterSet input format."""
    path = Path('covariance.root').resolve()
    macro = Path('makeCalcXsecCovariance.C').resolve()
    macro.write_text(f'''#include <TFile.h>
#include <TMatrixDSym.h>
void makeCalcXsecCovariance() {{
  TFile output("{path}", "RECREATE");
  TMatrixDSym covariance(2);
  covariance[0][0] = 0.01; covariance[1][1] = 0.04;
  covariance[0][1] = covariance[1][0] = -0.01;
  covariance.Write("covariance");
}}
''')
    run(['root', '-l', '-b', '-q', str(macro)], 'makeCovariance')
    return path


def tree_values(tree):
    """Flatten ROOT branches with multiple named scalar leaves, in stored order."""
    columns = []
    for branch in tree.branches:
        values = branch.array(library='np')
        if values.dtype.names:
            columns.extend(values[name] for name in values.dtype.names)
        else:
            columns.append(values)
    return np.column_stack(columns)


def read_output(path, n, best_fit=False):
    with uproot.open(path) as output:
        toys = tree_values(output['calcXsec/throws/xsecThrow_TTree'])
        nominal = tree_values(output['calcXsec/throws/xsecAtBestFitTree_TTree'])
        assert len(toys) == n and len(nominal) == 1
        assert np.isfinite(toys).all() and np.isfinite(nominal).all()
        matrices = output['calcXsec/matrices']
        covariance = matrices['covarianceMatrix_TH2D'].values()
        empirical = np.cov(toys, rowvar=False, ddof=0)
        np.testing.assert_allclose(covariance, empirical, atol=1e-10, rtol=1e-10)
        histograms = list(output['calcXsec/histograms'].values())
        nbins = sum(len(hist.values()) for hist in histograms)
        central = nominal[0] if best_fit else toys.mean(axis=0)
        start = 0
        for hist in histograms:
            stop = start + len(hist.values())
            np.testing.assert_allclose(hist.values(), central[start:stop], atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(hist.errors() ** 2, np.diag(empirical)[start:stop], atol=1e-10)
            start = stop
        for prefix, selection in [('bins', slice(0, nbins)), ('pars', slice(nbins, None))]:
            np.testing.assert_allclose(matrices[prefix + 'CovarianceMatrix_TH2D'].values(),
                                       empirical[selection, selection], atol=1e-10)
        positive = np.diag(empirical) > 1e-20
        expected_correlation = empirical[np.ix_(positive, positive)] / np.sqrt(
            np.outer(np.diag(empirical)[positive], np.diag(empirical)[positive]))
        np.testing.assert_allclose(matrices['correlationMatrix_TH2D'].values()[np.ix_(positive, positive)],
                                   expected_correlation, atol=1e-10)
        assert matrices['covarianceMatrix_TH2D'].axis(0).labels() is not None
    return toys, nominal[0], covariance


def check_moments(values, mean, covariance, label):
    """Six standard errors, for these small analytic Gaussian/Poisson models.

    For covariance use the empirical variance of centered products, which also
    handles non-Gaussian throws. No relative tolerance on zero covariances.
    """
    values = np.asarray(values)
    mean = np.asarray(mean)
    covariance = np.asarray(covariance)
    n = len(values)
    empirical_mean = values.mean(axis=0)
    mean_error = np.sqrt(np.diag(covariance) / n)
    assert np.all(np.abs(empirical_mean - mean) <= 6 * mean_error + 1e-10), (label, empirical_mean, mean)
    products = (values - mean)[:, :, None] * (values - mean)[:, None, :]
    covariance_error = products.std(axis=0, ddof=1) / np.sqrt(n)
    # Correct the known 1/N bias of the exported population covariance for this comparison.
    empirical_covariance = np.cov(values, rowvar=False, ddof=1)
    assert np.all(np.abs(empirical_covariance - covariance) <= 6 * covariance_error + 1e-10), (
        label, empirical_covariance, covariance)
    print(f'PASS {label}: {n} throws, mean={empirical_mean.tolist()}')
