# Phase 2A Fiji descriptive immune landscape

## Decision

**READY_FOR_PHASE2B_INFERENTIAL_MODELING**

## Analytical scope

This phase reconstructs the systems-serology landscape before inferential mixed-effects modeling. It summarizes original visit states, participant-paired log2 changes, geometric-mean ratios, primary-versus-recall effect differences and BPV-calibrated HPV-specific changes.

No hypothesis-test P values or false-discovery-rate decisions are generated in this phase. Confidence intervals describe effect-size precision and do not replace the prespecified Phase 2B models.

## Validated structure

- Unique participants: 80
- Long feature observations: 14720
- Complete paired feature records: 7360
- Visit-level descriptive rows: 736
- Dose–antigen–feature paired-effect rows: 368
- Primary-versus-recall descriptive contrasts: 92
- BPV-calibrated effect rows: 308

## Authoritative effect scale

All continuous assay values are analyzed as `log2(raw positive value)`. The paired response is:

`log2(Visit 2) - log2(Visit 1)`

The corresponding geometric-mean ratio is `2^(mean log2 change)`. Ratios above one indicate higher Visit 2 responses; ratios below one indicate lower Visit 2 responses.

## Biological trajectories

- Dose 0: unvaccinated baseline to primary 2vHPV induction.
- Doses 1–3: six-year 4vHPV persistence to heterologous 2vHPV recall.
- Cross-reactive HPV31/33/45/52/58 responses are retained separately from HPV16/18 vaccine-target responses.
- BPV-calibrated effects estimate HPV-specific changes after subtracting contemporaneous heterologous-control movement on the log2 scale.

## Generated matrices

- Primary-induction effect matrix for dose 0.
- Six-year persistence matrix for previous dose groups 1–3.
- Heterologous recall matrix for previous dose groups 1–3.
- Primary-versus-recall descriptive contrast table.
- BPV-calibrated HPV-specific change table.

## Assay-floor interpretation

Every effect carries the maximum floor severity observed across its Visit 1 and Visit 2 distributions. Moderate-floor results require an above-floor sensitivity model, and high-floor results require the two-part modeling strategy locked in Phase 1F.

## Next phase

Phase 2B will fit the prespecified inferential models for primary induction, six-year persistence, heterologous recall, primary-versus-recall contrasts, cross-reactive breadth, BPV controls and antibody functional coupling. Benjamini–Hochberg adjustment will be applied within the biological families locked in Phase 1F.
