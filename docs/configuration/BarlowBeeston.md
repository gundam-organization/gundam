# Monte Carlo statistical likelihoods

## BarlowBeestonLiteGaussian

```yaml
jointProbabilityConfig:
  type: BarlowBeestonLiteGaussian
```

The current model uncertainty `sqrt(sum(w^2))` is used at every evaluation.
There is no option to freeze nominal uncertainties, no nominal cache, and no
BANFF compatibility options. Scalar and sample calls use the same convention:
the uncertainty argument is a standard deviation, not a variance.
The historical `BarlowLLH_BANFF_OA2021` implementation remains available unchanged.
Legacy thresholds, fractional-error caps and finite replacements of infinity
are not reproduced.

For observation n, prediction mu and MC standard deviation sigma, the objective is

    min over x >= 0: 2*(x-n+n*log(n/x)) + (x-mu)^2/sigma^2.

For mu>0, x=beta*mu gives Conway's equations (10)-(11). The absolute-mean
parameterization also defines the continuous extension at mu=0 with sigma>0.
For n=0 the logarithmic term is zero; for sigma=0 the MC mean is fixed to mu.
An empty data/MC bin contributes zero. Invalid inputs (negative or nonfinite)
return positive infinity. Impossible observations with zero fixed expectation
also return infinity; undefined derivatives return NaN. There is no logging or
exception policy option. These are nonnegative-weight models.

`evaluate` returns the objective and analytic partial derivatives with respect to
mu and sigma using the envelope theorem. Sample gradients respect ignored bins. A full parameter gradient combines these partials with derivatives
of the propagated histogram. No finite-difference mutation or shared scratch
buffer is used.

Reference: J. S. Conway, *Incorporating Nuisance Parameters in Likelihoods for
Multisource Spectra*, [arXiv:1103.0354](https://arxiv.org/abs/1103.0354), section 5.

## Exact Barlow-Beeston solver

`BarlowBeeston::evaluateSources(n, sources, fixedPrediction)` profiles the model

    n ~ Poisson(fixedPrediction + sum_j scale_j * A_j)
    count_j ~ Poisson(A_j).

Each `Source` contains an observed unweighted integer MC `count` and its current
nonnegative `scale`. The optional fixed prediction has no MC uncertainty.
The result contains the saturated deviance, fitted prediction, fitted source
means and envelope derivatives with respect to source scales.

A monotonic one-dimensional equation is solved by bisection. Empty MC sources,
including an active solution on the boundary associated with the largest scale,
are retained. In a degenerate boundary solution the allocation between identical
empty sources is nonunique; reported scale derivatives follow the selected
allocation and are not a unique ordinary gradient at that point.

**Fit integration needs per-source inputs.** The existing histogram interface
only supplies sum(w) and sqrt(sum(w^2)); these do not determine the individual
counts and scales. Consequently `BarlowBeeston` refuses scalar/histogram evaluations
with nonzero uncertainty rather than silently constructing an effective single
source. With zero uncertainty it evaluates the fixed-mean Poisson deviance.
The new source API implements the complete statistical model, but is not yet
connected to source decomposition during propagation. No backend changes are made.

Reference: R. Barlow and C. Beeston, *Fitting using finite Monte Carlo samples*,
Comput. Phys. Commun. 77 (1993) 219-228,
[doi:10.1016/0010-4655(93)90005-W](https://doi.org/10.1016/0010-4655(93)90005-W).

## BarlowBeestonLitePoisson

```yaml
jointProbabilityConfig:
  type: BarlowBeestonLitePoisson
```

Implements the improved Dembinski-Abdelmotteleb approximation. With unweighted
Poisson data and the current weighted-MC moments, it uses

    a_eff = mu^2 / variance
    beta = (n + a_eff)/(mu + a_eff)
    q = PoissonDeviance(n, beta*mu) + PoissonDeviance(a_eff, beta*a_eff).

It keeps one nuisance per bin and an analytic profile, replacing the Gaussian
penalty by a Poisson deviance. It preserves asymmetry at low effective MC counts
and agrees with the Gaussian penalty near beta=1. It remains an approximation
for a mixture of templates. It is exact for one unweighted Poisson template;
the effective-count extension to varying event weights is still approximate.
Improvements in fit bias or coverage are problem-dependent and require toy
validation; it does not solve bins with no simulated support automatically.

Reference: H. Dembinski and A. Abdelmotteleb, *A fast and stable approximate
maximum-likelihood method for template fits*,
[arXiv:2206.12346](https://arxiv.org/abs/2206.12346), especially equations (17)-(18).

The MC uncertainty is always current, as in `BarlowBeestonLiteGaussian`. Both
classes have no class-specific configuration options. `ignoreBinsWithZeroPredictionAtPrior`
is inherited from `JointProbabilityBase`.

`evaluate` returns analytic partial derivatives with respect to mu and sigma.
These include the dependence of a_eff on both inputs, which must be included
when differentiating weighted predictions. Products involving an extreme
a_eff or beta are evaluated using logarithmic scales to avoid intermediate
overflows. A series preserves small displacements near an Asimov minimum.

For sigma=0, the result is the ordinary Poisson deviance. The empty input
(n,mu,sigma)=(0,0,0) contributes zero; mu=0 with n>0 or sigma>0 returns infinity.
The latter has no valid effective-count parameterization for nonnegative MC
weights. Negative/nonfinite inputs return infinity, with NaN derivatives.
At (0,0,0), the prediction derivative is the one-sided fixed-sigma value 2.

## Shared numerical helper

`src/Utils/include/PoissonDeviance.h` is a numerical utility, not another configurable likelihood.
Its `GundamUtils::halfPoissonDeviance(n, mu)` returns

    mu - n + n*log(n/mu),

half the saturated Poisson deviance (the likelihood classes supply the factor 2).
It handles n=0 and uses a series or log1p near mu=n so that subtracting almost
equal numbers does not destroy the quadratic signal needed by Hesse and
gradient checks. An overload accepts the residual (mu-n)/n separately when
profiling would otherwise round away a small displacement. It is shared by
the exact Barlow-Beeston solver and both lite likelihoods.
