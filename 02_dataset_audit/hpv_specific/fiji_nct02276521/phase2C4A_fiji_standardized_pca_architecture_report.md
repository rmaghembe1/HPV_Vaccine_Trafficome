# Phase 2C4A Fiji standardized PCA architecture

## Decision

**READY_FOR_PHASE2C4B_PCA_STABILITY_AND_CLUSTERING**

- Participants: 80
- Raw features: 92
- BPV-calibrated features: 77
- Features were standardized to zero mean and unit variance separately within each representation.
- Raw components required for 80% variance: 21
- BPV-calibrated components required for 80% variance: 16
- Primary-versus-recall FDR-significant PCs among PC1–PC10: 4
- Recall-dose FDR-significant PCs among PC1–PC10: 1

The PCA models are unsupervised and were fit without immunization-context or dose-group labels. Group labels were introduced only after PCA to describe score centroids and test PC-score differences.

Loadings for PC1–PC10 were annotated by antigen and systems-serology feature family. PCA component signs are mathematically arbitrary; biological interpretation should therefore focus on relative loading architecture, absolute contribution and internally consistent score direction within the fitted model.

Phase 2C4B should evaluate leave-one-feature-family-out stability, bootstrap component reproducibility, centroid separation and unsupervised clustering across plausible cluster numbers. Raw and BPV-calibrated representations must remain separate throughout that analysis.
