# Phase 2C4B Fiji PCA stability and clustering

## Decision

**READY_FOR_PHASE2C4C_IMMUNE_STATE_SYNTHESIS**

## PCA stability design

- Leave-one-feature-family-out analyses: 8
- Stability rows across PC1–PC5: 40
- Participant-bootstrap replicates per representation: 300
- Bootstrap component-level rows: 3000

Components were matched to the full-data reference PCA using maximum absolute loading correlation and their signs were aligned before score comparison. This avoids treating arbitrary PCA sign changes or component reordering as biological instability.

## Clustering design

- Candidate cluster counts: k=2 through k=6
- Clustering space: representation-specific PCA scores required to explain at least 80% of variance
- Participant-subsampling stability replicates per k: 100
- A stable candidate requires mean subsample adjusted Rand index of at least 0.75, silhouette score of at least 0.20 and a minimum cluster size of at least eight participants.

A selected k is reported only when these prespecified criteria are met. Failure to identify a stable partition should be interpreted as evidence for continuous immune-state structure rather than as an analytical failure.

## Phase 2C4A biological reference

PC1 represents a cross-reactive recall-breadth axis dominated by HPV31/45/52/58 IgG, IgG-subclass and Fc-receptor features opposed to vaccine-type HPV16 features. PC2 represents a vaccine-type HPV16/18 abundance and effector axis. The raw PC2 previous-dose association was not retained after BPV calibration and therefore remains a qualified secondary schedule observation.
