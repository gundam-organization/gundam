#!/usr/bin/env python3
"""Generate documentation figures using NumPy and Matplotlib; no GUNDAM needed."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


COLORS = ['#2563eb', '#d97706', '#0f766e']


def cumulative(position):
    t = np.clip((position + 1.) / 2., 0., 1.)
    return t**3 * (10. + t * (-15. + 6. * t))


def kernel(position):
    t = np.clip((position + 1.) / 2., 0., 1.)
    return 15. * t**2 * (1. - t)**2


def fractions(x, edges, width):
    """Return f_b(x) and df_b/dx; final dimension indexes histogram bins."""
    upper = (edges[1:] - np.asarray(x)[..., None]) / width
    lower = (edges[:-1] - np.asarray(x)[..., None]) / width
    return cumulative(upper) - cumulative(lower), (kernel(lower) - kernel(upper)) / width


def save(figure, directory, name):
    figure.savefig(directory / name, dpi=160, facecolor='white')
    plt.close(figure)
    print(directory / name)


def plot_fractions(directory):
    x = np.linspace(-.4, 2.4, 1000)
    edges = np.array([0., 1., 2.])
    width = .2
    values, slopes = fractions(x, edges, width)
    figure, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True, layout='constrained')
    for b in range(2):
        axes[0].plot(x, values[:, b], color=COLORS[b], label=f'Bin {b}: [{b}, {b+1}]')
        axes[1].plot(x, slopes[:, b], color=COLORS[b], label=f'Bin {b}')
    axes[0].plot(x, 1. - values.sum(axis=1), '--', color='#64748b', label='Outside')
    axes[1].plot(x, -slopes.sum(axis=1), '--', color='#64748b', label='Outside')
    for axis in axes:
        for edge in edges:
            axis.axvline(edge, color='#94a3b8', lw=.8, ls=':')
        axis.axvspan(1. - width, 1. + width, color='#cbd5e1', alpha=.3)
        axis.legend(loc='upper right', ncols=3, fontsize=9)
    axes[0].set(ylabel=r'Fraction $f_b(x)$', ylim=(-.06, 1.12),
                title='A moving event is shared continuously between bins (width h = 0.2)')
    axes[1].set(xlabel='Observable x', ylabel=r'Derivative $df_b/dx$')
    save(figure, directory, 'ObservableBinFractions.png')


def plot_contribution(directory):
    theta = np.linspace(-.35, .35, 1000)
    x = .95 + theta
    w = 2. * (1. + .3 * theta)
    f, dfdx = fractions(x, np.array([0., 1., 2.]), .2)
    c = w[:, None] * f
    weight_term = .6 * f
    migration_term = w[:, None] * dfdx  # dx/dtheta = 1
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    for b in range(2):
        axes[0].plot(theta, c[:, b], color=COLORS[b], label=f'Bin {b}')
    axes[0].plot(theta, c.sum(axis=1), '--', color='#64748b', label='Total = event weight')
    axes[0].set(xlabel=r'Parameter $\theta$', ylabel=r'Contribution $c_b=w f_b$',
                title='One event: x = 0.95 + θ, w = 2(1 + 0.3θ)')
    # Detail one bin so that both chain-rule terms remain readable.
    axes[1].plot(theta, weight_term[:, 1], '--', color=COLORS[0], label=r'Weight term: $f_1\,dw/d\theta$')
    axes[1].plot(theta, migration_term[:, 1], ':', lw=2.5, color=COLORS[1],
                 label=r'Migration term: $w\,(df_1/dx)\,(dx/d\theta)$')
    axes[1].plot(theta, (weight_term + migration_term)[:, 1], color=COLORS[2],
                 label=r'Total: $dc_1/d\theta$')
    axes[1].set(xlabel=r'Parameter $\theta$', ylabel='Derivative in bin 1',
                title='The derivative includes both paths')
    for axis in axes:
        axis.legend(fontsize=8, loc='upper left')
    save(figure, directory, 'ObservableContributionGradient.png')


def toy_prediction(theta):
    """Three fixed reference events plus three moving events; fixed MC sample."""
    theta = np.asarray(theta)
    x0 = np.array([.5, 1.5, 2.5, .95, 1.95, 2.7])
    speed = np.array([0., 0., 0., 1., .6, -.5])
    base = np.array([10., 10., 10., 3., 4., 5.])
    rate = np.array([0., 0., 0., .25, .25, .25])
    x = x0 + theta[..., None] * speed
    w = base * (1. + theta[..., None] * rate)
    f, dfdx = fractions(x, np.array([0., 1., 2., 3.]), .22)
    contribution = w[..., None] * f
    weight_term = (base * rate)[..., None] * f
    migration_term = w[..., None] * dfdx * speed[:, None]
    derivative = weight_term + migration_term
    return (contribution.sum(axis=-2), (contribution**2).sum(axis=-2),
            derivative.sum(axis=-2), (2. * contribution * derivative).sum(axis=-2))


def plot_likelihood(directory):
    theta = np.linspace(-.3, .3, 801)
    mu, q, dmu, _ = toy_prediction(theta)
    data = toy_prediction(0.)[0]  # Fixed Asimov reference, never shifted in the scan.
    # Illustrative Poisson deviance only: no MC nuisance profiling in this plot.
    residual = (mu - data) / data
    llh = 2. * (data * (residual - np.log1p(residual))).sum(axis=-1)
    dllh = (2. * (mu - data) / mu * dmu).sum(axis=-1)
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    for b in range(3):
        axes[0, 0].plot(theta, mu[:, b], color=COLORS[b], label=f'Bin {b}')
        axes[0, 1].plot(theta, q[:, b], color=COLORS[b], label=f'Bin {b}')
    axes[0, 0].set(ylabel=r'$\mu_b=\sum_e c_{eb}$', title='Predicted bin contents')
    axes[0, 1].set(ylabel=r'$Q_b=\sum_e c_{eb}^2$', title='Diagonal MC variance (not used in this toy LLH)')
    axes[1, 0].plot(theta, llh, color=COLORS[0], label='Poisson deviance')
    index = np.argmin(np.abs(theta - .12))
    tangent_x = np.linspace(theta[index] - .06, theta[index] + .06, 100)
    axes[1, 0].plot(tangent_x, llh[index] + dllh[index] * (tangent_x - theta[index]),
                   '--', color=COLORS[1], label='Tangent from analytic gradient')
    axes[1, 0].scatter([theta[index]], [llh[index]], color=COLORS[1], s=25)
    axes[1, 0].set(ylabel='Statistical LLH', title='Fixed data, varying prediction')
    axes[1, 1].plot(theta, dllh, color=COLORS[2], label=r'$\sum_b 2(\mu_b-n_b)\,\partial_\theta\mu_b/\mu_b$')
    axes[1, 1].axhline(0., color='#94a3b8', lw=.8)
    axes[1, 1].set(ylabel=r'$\partial LLH/\partial\theta$', title='Sum of the bin contributions')
    for axis in axes.flat:
        axis.set_xlabel(r'Parameter $\theta$')
        axis.legend(fontsize=8)
    save(figure, directory, 'ObservableHistogramLikelihood.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'figures')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .18, 'lines.linewidth': 2})
    plot_fractions(args.output_dir)
    plot_contribution(args.output_dir)
    plot_likelihood(args.output_dir)


if __name__ == '__main__':
    main()
