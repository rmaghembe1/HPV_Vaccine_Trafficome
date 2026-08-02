# Phase 1B2 metadata structure decision

## Decision summary

| Variable | Decision | Structured columns | Worksheets |
|---|---|---:|---|
| participant_id | TEXT_OR_GLOSSARY_ONLY | 0 | Data glossary |
| age | NOT_DETECTED | 0 | — |
| weight | NOT_DETECTED | 0 | — |
| bmi | NOT_DETECTED | 0 | — |
| sex | NOT_DETECTED | 0 | — |
| ethnicity | NOT_DETECTED | 0 | — |
| dose | TEXT_OR_GLOSSARY_ONLY | 0 | Data glossary |
| visit_time | TEXT_OR_GLOSSARY_ONLY | 0 | Data glossary |
| vaccine | TEXT_OR_GLOSSARY_ONLY | 0 | Data glossary |

## Decision counts

- NOT_DETECTED: 5
- TEXT_OR_GLOSSARY_ONLY: 4

## Interpretation

`PARTICIPANT_LEVEL_CANDIDATE` means that the workbook appears to contain a structured column with repeated values below a candidate header. It still requires confirmation that the field is linked to a stable participant identifier and is not an assay or group label.

Variables reported in the publication but absent from the workbook should be classified as requiring external linkage or author-provided participant metadata.
