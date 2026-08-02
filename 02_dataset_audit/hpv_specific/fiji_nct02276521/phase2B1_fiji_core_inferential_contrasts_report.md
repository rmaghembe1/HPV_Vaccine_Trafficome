# Phase 2B1 Fiji core inferential contrasts

## Decision

**READY_FOR_PHASE2B2_MIXED_MODEL_AND_FLOOR_SENSITIVITY**

## Analytical implementation

Because the Fiji dataset contains exactly two complete measurements per participant, participant-paired log2 change removes each participant-specific intercept. The primary induction and recall tests are therefore implemented on paired log2 changes. This is the direct two-time-point analogue of modeling a visit effect with a participant-specific intercept.

## Tests completed

- Within-trajectory primary, recall and BPV-control tests: 368
- Global persistence and recall dose tests: 184
- Pairwise dose contrasts: 552
- Primary-versus-recall contrasts: 81

## Inferential methods

- Within-participant changes: two-sided one-sample t tests of mean log2 change against zero.
- Three-group dose heterogeneity: Welch heteroscedastic one-way analysis.
- Pairwise dose comparisons: Welch two-group contrasts.
- Primary-versus-recall comparisons: Welch contrasts between the dose-0 primary group and pooled previously vaccinated participants.
- Multiplicity control: Benjamini–Hochberg adjustment within the model and biological outcome families locked in Phase 1F.

## FDR-significant results

- Within-trajectory tests: 238
- Global dose tests: 9
- Pairwise dose contrasts: 14
- Primary-versus-recall contrasts: 52

## Interpretation boundary

These results establish statistical evidence for systems-serology changes but do not directly measure intracellular trafficking, antigen processing or APC signaling. Those mechanisms will be addressed through integration with HPV cellular datasets and mechanistic transcriptomic comparators.

## Next phase

Phase 2B2 will fit long-format participant-random-intercept models as confirmation and implement the prespecified above-floor and two-part sensitivity analyses for moderate- and high-floor outcomes.
