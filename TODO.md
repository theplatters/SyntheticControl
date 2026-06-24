# Todos to implement

## Priority 1

- Classic Synthetic Control (Abadie et al.):
  The original nested optimization framework to solve for weights W and V that minimize pre-treatment MSPE.
- Placebo Inference (In-space & In-time):
  - In-space: Iteratively applying the SCM to all untreated units to construct a distribution of placebo effects for p-value calculation.
  - In-time: Applying the SCM to a fake pre-treatment date to test for parallel trends/anticipation effects.
- Standard Visualization Suite:
  Path plots (Actual vs. Synthetic).
  - Gap plots (Treatment effect over time).
  - P-value/Placebo distribution plots.
- Data Prep Handling: A robust data structure (similar to R's Synth::dataprep) that handles panel data imbalances, missing values, and allows easy specification of predictors, treat id, and time dimensions.

## Priority 2

- Penalized & Regularized SCM (Doudchenko & Imbens): Allowing for negative weights and an intercept by incorporating Lasso, Ridge, or Elastic Net penalties to combat over-fitting when the donor pool is large.
- Synthetic Difference-in-Differences (SDID) (Arkhangelsky et al.): Combining the strengths of DiD (unit and time fixed effects) with SCM (regularized unit weights) to achieve double robustness.
- Generalized Synthetic Control (GSC) (Xu / Gobillon & Magnac): Utilizing latent factor models (Interactive Fixed Effects) to handle multiple treated units and time-varying unobserved confounders.
- Matrix Completion Methods (Athey et al.): Treating causal inference as a matrix completion problem, leveraging nuclear norm regularization to predict counterfactuals.
- Proximal / De-biased SCM (Miao et al. / Chernozhukov et al.): Adjusting for measurement errors in proxies and providing valid asymptotic inference (confidence intervals) without relying solely on placebos.
- Staggered Adoption Handling: Built-in support for settings where different units receive treatment at different times, aggregating individual synthetic controls into a coherent cohort effect.

## Priority 3

- Autotuning / Cross-Validation:
  Automated cross-validation loops to pick hyper-parameters
  (like λ in Ridge/Lasso SCM or the number of latent factors in GSC).
- Interactive Diagnostics:
  Out-of-sample validation metrics (e.g., Mean Absolute Scaled Error)
  checks for "donor pool sparsity" to warn users if their synthetic control is built on too many microscopic weights.
- Sensivity Analysis / Bounds:
  Implementing Rosenbaum-style bounds or general sensitivity analyses  
  test how robust the results are to hidden bias (unobserved confounders).
- Cross-Language Bridges:
  Standardized API wrappers or exported schemas
  so users can call your hyper-fast Julia backendfrom Python (PyCall) or R (RCall).
- GPU Acceleration:
  For massive datasets (e.g., large-scale geo-experiments in tech tech firms)
  utilizing CUDA.jl to parallelize placebo distribution bootstrapping.
