# Phase 2C5B Fiji figure QA and manuscript integration

## Decision

**PHASE2C5B_REPAIR_REQUIRED**

## Automated file QA

- Figure files inspected: 6
- Figure files passing automated QA: 5
- Raster checks: successful opening, minimum pixel dimensions, approximately 600-dpi metadata, image variation and non-empty content occupancy.
- SVG checks: valid XML, retained text elements, viewBox present and no embedded raster images.

## Visual review file

- Contact sheet: `09_figures/hpv_specific/fiji_nct02276521/phase2C5/phase2C5B_fiji_figure_QA_contact_sheet.png`

The contact sheet contains the main and supplementary figures at reduced scale for manual inspection of panel balance, text size, legends, axis labeling and clipping.

## Manuscript integration

- Figure legends: `10_manuscript/hpv_specific/fiji_nct02276521/phase2C5B_fiji_multivariate_figure_legends.md`
- Legend registry: `08_results/tables/phase2C5B_fiji_figure_legend_registry.tsv`

The main legend presents the continuous two-axis immune-state architecture, while the supplementary legend documents loading structure, feature-family robustness and the context-dominated nature of the clustering solutions.

## Scientific interpretation

The publication-facing interpretation remains that PC1 represents cross-reactive recall breadth and PC2 represents vaccine-type HPV16/18 effector organization. Raw k=2 clustering is treated as an experimental-context partition, not as an intrinsic immune subtype. The raw-only recall-dose association remains a qualified secondary result because it was not retained after BPV calibration.
