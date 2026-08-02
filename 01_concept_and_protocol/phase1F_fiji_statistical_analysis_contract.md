# Phase 1F Fiji statistical analysis contract

## Decision

**READY_FOR_PHASE2_MODELING**

## Biological estimands

The Fiji cohort contains two distinct immunological trajectories that must not be combined under a uniform booster-response label:

1. **Primary induction:** unvaccinated baseline to the day-28 primary 2vHPV response among participants with zero previous 4vHPV doses.
2. **Persistence and recall:** six-year 4vHPV immunity to day-28 heterologous 2vHPV recall among participants with one, two or three previous 4vHPV doses.

## Core modeling rules

- Previous dose number is categorical in primary models.
- Participant identity is retained as a random intercept for repeated-measure models.
- Models are fitted separately by antibody feature or prespecified functional family; raw assay scales are not pooled indiscriminately.
- HPV16 and HPV18 are vaccine-target antigens.
- HPV31, HPV33, HPV45, HPV52 and HPV58 define the cross-reactive antigen layer.
- BPV is retained as a heterologous control and is not treated as an HPV vaccine-response antigen.
- ADCP and neutralization are analyzed only for HPV16 and HPV18.

## Transformation rules

- All observed assay measurements are positive; therefore primary continuous models use `log2(raw value)` without adding an arbitrary pseudocount.
- Neutralizing antibody is analyzed as a discrete dilution titer on the log2 scale.
- Effect sizes are reported as log2 differences and geometric-mean ratios.
- Original-scale summaries remain available for biological interpretation.

## Assay-floor policy

| Floor fraction | Severity | Primary treatment |
|---:|---|---|
| <5% | None | Standard log2 continuous model |
| 5%–<15% | Low | Continuous model plus floor summary |
| 15%–<30% | Moderate | Continuous model plus binary above-floor sensitivity analysis |
| ≥30% | High | Two-part sensitivity analysis: probability above floor and conditional response magnitude |

Observed distribution counts:

- none: 105
- low: 26
- moderate: 21
- high: 32

## Multiplicity control

Benjamini–Hochberg false-discovery-rate correction is applied within prespecified biological families rather than across every numerical test in the project. Families include antibody abundance, IgG subclasses, Fc-receptor binding, ADCP, neutralization, persistence, primary induction, recall, dose interactions and cross-reactive breadth.

## Reporting requirements

Each reported comparison must include the number of participants, effect estimate, 95% confidence interval, raw P value, within-family adjusted P value, floor severity and biological context. Statistical significance alone will not substitute for effect magnitude or mechanistic coherence.

## Excluded Fiji-specific covariates

Age, weight, BMI, sex and ethnicity are not present in the open workbook and must not be imputed or inferred. These questions will be addressed through additional HPV cohorts, data requests or mechanistic comparator datasets.
