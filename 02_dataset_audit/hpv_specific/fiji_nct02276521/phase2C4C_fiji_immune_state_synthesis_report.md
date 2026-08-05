# Phase 2C4C Fiji immune-state synthesis

## Decision

**READY_FOR_PHASE2C4_COMMIT_AND_PHASE2C5_FIGURE_CONSTRUCTION**

## Core multivariate architecture

Fiji systems-serology responses were organized along two principal and reproducible immune-state axes. PC1 represented a cross-reactive recall-breadth programme dominated by HPV31/33/45/52/58 IgG, IgG-subclass and Fc-receptor features. PC2 represented a vaccine-type HPV16/18 antibody-abundance and effector programme, including ADCP and neutralization in the raw matrix.

Both axes differed between primary induction and heterologous recall in the raw and BPV-calibrated representations. PC1 showed the strongest bootstrap and feature-family-omission stability. PC2 was reproducible but showed greater bootstrap and family-removal sensitivity.

## Cross-representation interpretation

- PC1 (cross_reactive_recall_breadth_axis): score Spearman rho=0.898; shared-feature loading Pearson r=0.948; high_cross_representation_concordance.
- PC2 (vaccine_type_effector_axis): score Spearman rho=0.749; shared-feature loading Pearson r=0.826; moderate_cross_representation_concordance.

## Discrete clustering versus continuous structure

The raw matrix supported a stable k=2 partition (silhouette=0.213; mean subsample adjusted Rand index=0.998). However, 77/80 participants (96.25%) aligned with the known primary-versus-recall context. The raw clusters therefore mainly recapitulated the experimental immunization contrast rather than defining independent intrinsic immune-response subtypes.

The best BPV-calibrated k=2 solution remained strongly context-associated but did not meet the prespecified silhouette threshold (silhouette=0.169; required at least 0.20). No calibrated k value from two through six met all stability criteria. The preferred biological model is therefore a continuous multiaxial recall landscape rather than stable discrete immune states.

## Previous-dose-group effect

Only raw PC2 differed among one-, two- and three-dose recall groups after FDR correction. The corresponding BPV-calibrated PC2 test was not significant. This schedule association remains a qualified secondary observation and should not be interpreted as evidence for discrete dose-defined immune states.

## Biological boundary

The multivariate axes summarize downstream antibody quantity, subclass, Fc-receptor, phagocytic and neutralizing organization. They do not directly measure intracellular antigen routing, endosomal processing, HLA-II loading, germinal-centre dynamics or memory B-cell lineage evolution.
