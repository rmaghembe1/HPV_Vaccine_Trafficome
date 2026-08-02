# Phase 1D Fiji statistical-readiness audit

## Decision

**READY_FOR_PHASE2_DESCRIPTIVE_AND_MIXED_EFFECTS_ANALYSIS**

## Dataset structure

- Unique participants: 80
- Numeric feature observations: 14720
- Complete paired feature observations: 7360
- Duplicate analytical keys: 0
- Nonnumeric measurements: 0

## Previous 4vHPV dose distribution

| Previous doses | Participants |
|---:|---:|
| 0 | 20 |
| 1 | 20 |
| 2 | 21 |
| 3 | 19 |

## Assay-behaviour findings

- Target-feature transformation rules: 92
- Visit-feature distributions with possible lower-bound accumulation: 79
- Discrete or titer-like distributions: 4

## Transformation framework

- Neutralizing-antibody measurements are treated as discrete titers and analyzed on the log2 scale.
- Strongly skewed positive antibody and Fc-receptor measurements use target-feature-specific log2 transformations.
- The pseudocount is half the smallest positive value for the relevant antigen-feature combination.
- Repeated minimum values are flagged as possible assay-floor accumulation.

## Primary paired response

`log2(Visit 2 + pseudocount) - log2(Visit 1 + pseudocount)`

This represents the participant-specific transition from long-term persistence six years after previous 4vHPV vaccination to day-28 recall following the 2vHPV booster.

## Modeling implication

Initial Fiji models should include participant pairing, previous 4vHPV dose number, visit, antigen target or antigen class, and antibody-functional layer. Age, BMI, sex and ethnicity remain unavailable in the open workbook.
