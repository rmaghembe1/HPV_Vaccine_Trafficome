# HPV Vaccine Trafficome Project

## Project identity

This is an independent HPV-vaccine computational systems-immunology repository. The current release is centered on the Fiji HPV systems-serology study associated with ClinicalTrials.gov identifier `NCT02276521`.

It is separate from the earlier general cross-vaccine Vaccine Trafficome project and from that project's manuscript, journal correspondence, figures, and submission history.

## Current analytical scope

The Fiji analysis includes 80 participants, 7,360 participant-antigen-feature observations, and 92 antigen-feature analysis rules. It evaluates primary bivalent HPV vaccine induction, long-term immunity after quadrivalent HPV vaccination, heterologous bivalent HPV vaccine recall, HPV16/18 vaccine-target responses, cross-reactive HPV31/33/45/52/58 responses, antibody abundance, subclass organization, Fc-gamma-receptor binding, neutralization, antibody-dependent cellular phagocytosis, previous-dose effects, persistence, primary-versus-recall contrasts, and multivariate immune-state architecture.

## Evidence boundary

The Fiji resource is a systems-serology dataset. It measures downstream antibody magnitude, breadth, subclass organization, Fc-receptor engagement, and functional activity. It does **not** directly measure intracellular endocytosis, endosomal routing, lysosomal processing, or antigen-presentation kinetics. Trafficome-related interpretation is therefore mechanistic and hypothesis-generating rather than a claim of direct intracellular-trafficking measurement.

## Principal findings

- Primary immunization generated strong HPV16/18 IgG, IgG1, IgA1, neutralizing-antibody, and phagocytic responses.
- Heterologous recall produced broader systems-serology remodeling, especially in subclass and Fc-receptor features.
- Neutralization and phagocytosis remained functionally coupled after dose adjustment.
- Cross-reactive serological breadth extended beyond HPV16/18.
- Two reproducible continuous immune-state axes captured recall breadth and HPV16/18 abundance-effector organization.
- A stable raw two-cluster solution was predominantly aligned with primary-versus-recall context and is not interpreted as an intrinsic biological subtype.
- Previous-dose effects were comparatively sparse.
- Bovine papillomavirus control behavior was not uniformly inert and was evaluated through calibration and sensitivity analyses.

## Data limitations

The deposited workbook does not provide participant-level BMI, age, sex, or ethnicity variables suitable for the covariate analyses considered here. Those attributes are not inferred or reconstructed.

## Repository organization

- `02_dataset_audit/`: metadata, quality-control, and analytical decisions
- `03_data_raw/`: local source data excluded from Git
- `05_gene_sets_and_modules/`: mechanistic module definitions
- `06_scripts/`: reproducible analysis and figure-generation code
- `08_results/`: processed tables, model outputs, and figure source data
- `09_figures/`: publication-quality PNG, TIFF, and editable SVG figures
- `10_reproducibility/`: logs and reproducibility records

## Reproducibility checkpoint

The verified HPV-only analytical lineage ends at `4c0fee0b52387e1d92efb5ec9d669399a7ceb92c`. The local source workbook is excluded from Git and has verified MD5 `e42173e1d8297cd64420fd9682c42674`.

## Public data

The source workbook is publicly available from Zenodo record `14848069` with DOI `10.5281/zenodo.14848069`. The deposited filename is `NCOMMS-24-64334A HPV collated antibody feature data.xlsx`.

## Licensing

This repository uses scope-based dual licensing:

- software, scripts, and software documentation are licensed under the
  **MIT License**; see `LICENSE`;
- original documentation, figures, figure legends, and derived analytical
  tables are licensed under **Creative Commons Attribution 4.0 International
  (CC BY 4.0)**; see `LICENSE-CONTENT.md`;
- source datasets, deposited workbooks, third-party publications, and other
  third-party material retain their original terms and are not relicensed by
  this repository.

Copyright (c) 2026 Reuben S. Maghembe.
