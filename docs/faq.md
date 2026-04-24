---
layout: default
title: FAQs 
next_page: "LineageandLegacy.html"
---

<details>
  <summary><strong>Is a non-converging Gundam fit a sign of a software problem?</strong></summary>
  <p>Not usually. Gundam uses MINUIT to minimize the negative log-likelihood (NLL) function and find the best-fit parameters. For this process to work reliably, the NLL surface must be smooth, well-behaved, and contain a clear global minimum. Attributes related to the model, parameterization or input uncertainties/correlations can create a likelihood surface that is difficult to optimize. Thus, non-convergence is often a diagnostic signal that the model or inputs need refinement, rather than a failure of Gundam.</p>
</details>
