---
layout: default
title: FAQs 
next_page: "LineageandLegacy.html"
---

<style>
details0 {
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 10px;
  margin-bottom: 10px;
  background: #f9f9f9;
}

summary0 {
  font-weight: bold;
  cursor: pointer;
  font-size: 16px;
}

details0[open] {
  background: #eef6ff;
}
</style>

<details0>
  <summary0><strong>Is a non-converging Gundam fit a sign of a software problem?</strong></summary0>
  <p>Not usually. Gundam uses MINUIT to minimize the negative log-likelihood (NLL) function and find the best-fit parameters. For this process to work reliably, the NLL surface must be smooth, well-behaved, and contain a clear global minimum. Attributes related to the model, parameterization or input uncertainties/correlations can create a likelihood surface that is difficult to optimize. Thus, non-convergence is often a diagnostic signal that the model or inputs need refinement, rather than a failure of Gundam.</p>
</details0>

<style>
details1 {
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 10px;
  margin-bottom: 10px;
  background: #f9f9f9;
}

summary1 {
  font-weight: bold;
  cursor: pointer;
  font-size: 16px;
}

details1[open] {
  background: #eef6ff;
}
</style>

<details1>
  <summary1><strong>How does Gundam handle spline extrapolation?</strong></summary1>
  <p>Gundam features multiple spline interpolation methods, and they have different approaches for when the spline extends beyond the boundaries. A not-a-knot spline performs cubic extrapolation by continuing the cubic polynomial defined by the first/last two points of the dataset. This maintains agreement with splines generated using ROOT's TSpline3 class. A Catmull-Rom spline extrapolates linearly beyond the defined knots.</p>
</details1>
