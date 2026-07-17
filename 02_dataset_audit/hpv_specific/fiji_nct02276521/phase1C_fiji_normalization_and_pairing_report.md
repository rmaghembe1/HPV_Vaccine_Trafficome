# Phase 1C Fiji normalization and pairing report

## Structural decision

The previous Phase 1B2 lexical classifier produced a false-negative decision because it did not recognize the exact workbook headers `ID` and `Dosage`, and because visit and antigen target are encoded in worksheet names. The Phase 1B2 report is superseded by the corrected decision table generated in this phase.

## Normalized dataset

- Unique participants: 80
- Participant–antigen–visit rows: 1280
- Nonmissing feature-level observations: 14720
- Antigen targets: BPV, HPV16, HPV18, HPV31, HPV33, HPV45, HPV52, HPV58
- Assay features: IgG, IgM, IgA1, IgA2, IgG1, IgG2, IgG3, IgG4, FcgR2A, FcgR2B, FcgR3A, ADCP, nAb
- Duplicate participant–antigen–visit keys: 0
- Participants with inconsistent dosage values: 0

## Previous 4vHPV dose distribution

| Previous doses | Participants |
|---:|---:|
| 0 | 20 |
| 1 | 20 |
| 2 | 21 |
| 3 | 19 |

## Pairing by antigen

| Antigen | Visit 1 | Visit 2 | Paired | V1 only | V2 only |
|---|---:|---:|---:|---:|---:|
| BPV | 80 | 80 | 80 | 0 | 0 |
| HPV16 | 80 | 80 | 80 | 0 | 0 |
| HPV18 | 80 | 80 | 80 | 0 | 0 |
| HPV31 | 80 | 80 | 80 | 0 | 0 |
| HPV33 | 80 | 80 | 80 | 0 | 0 |
| HPV45 | 80 | 80 | 80 | 0 | 0 |
| HPV52 | 80 | 80 | 80 | 0 | 0 |
| HPV58 | 80 | 80 | 80 | 0 | 0 |

## Metadata feasibility

- Participant ID: directly available.
- Previous 4vHPV dose number: directly available.
- Visit and booster timing: derivable from worksheet suffix and glossary.
- Antigen target: derivable from worksheet name.
- Age, weight, BMI, sex and ethnicity: absent from the open workbook.

## Analysis decision

The Fiji cohort is suitable for participant-paired analysis of long-term antibody persistence and day-28 booster recall by previous 4vHPV dose number, antigen target and antibody functional layer. It is not independently suitable for age, BMI, sex or ethnicity modeling without external participant metadata linkage.

## Primary analytical contrasts

1. Visit 2 versus Visit 1 within participants.
2. Interaction between visit and previous 4vHPV dose number.
3. HPV16/18 vaccine-target responses versus HPV31/33/45/52/58 cross-reactive responses.
4. HPV-specific responses versus BPV heterologous control.
5. Antibody quantity versus Fc-receptor engagement, ADCP and neutralization.
