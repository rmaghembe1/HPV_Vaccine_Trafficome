# Phase 1B2 supersession notice

The initial Phase 1B2 lexical classifier did not recognize:

- `ID` as the participant identifier;
- `Dosage` as the participant-level previous-dose variable;
- visit encoded by the `_v1` and `_v2` worksheet suffixes;
- antigen target encoded by worksheet prefixes.

The original Phase 1B2 script and outputs are retained only as provenance.

Authoritative replacements are:

- `08_results/tables/phase1B2_fiji_metadata_decision_corrected.tsv`
- `02_dataset_audit/hpv_specific/fiji_nct02276521/phase1C_fiji_normalization_and_pairing_report.md`
- `02_dataset_audit/hpv_specific/fiji_nct02276521/phase1E_fiji_immunization_context_correction.md`
