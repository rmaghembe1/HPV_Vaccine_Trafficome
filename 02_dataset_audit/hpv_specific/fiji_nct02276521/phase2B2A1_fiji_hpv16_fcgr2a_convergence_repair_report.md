# Phase 2B2A1 HPV16–FcγR2A convergence repair

## Decision

**READY_FOR_PHASE2B2D_INTEGRATION_AND_BIOLOGICAL_SYNTHESIS**

The original HPV16–FcγR2A L-BFGS model returned finite coefficients but did not satisfy the optimizer convergence criterion. It also produced singular random-effects and Hessian warnings.

The model was therefore refitted independently using multiple optimizers. Only models with explicit optimizer convergence, finite fixed effects, finite fixed-effect covariance and an acceptable covariance eigenstructure were eligible for selection.

- Selected optimizer: powell
- Selected log likelihood: -388.39307934746193
- Acceptable optimizers: 2
- Remaining nonconverged models: 0
- Repaired contrast rows: 1012

All Phase 2B2A multiplicity adjustments were recomputed after replacing the eleven HPV16–FcγR2A contrasts. The repaired tables are authoritative for subsequent integration and biological synthesis.
