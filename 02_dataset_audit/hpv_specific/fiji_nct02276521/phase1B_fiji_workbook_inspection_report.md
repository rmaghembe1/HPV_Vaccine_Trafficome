# Phase 1B Fiji workbook inspection

## File integrity

- Workbook: `/mnt/d/HPV_Vaccine_Trafficome_Project/03_data_raw/hpv_specific/fiji_nct02276521/NCOMMS-24-64334A_HPV_collated_antibody_feature_data.xlsx`
- Expected MD5: `e42173e1d8297cd64420fd9682c42674`
- Observed MD5: `e42173e1d8297cd64420fd9682c42674`
- Integrity decision: **PASS**

## Workbook structure

- Number of worksheets: 17
- Total worksheet rows: 1390
- Total detected keyword hits: 204

## Worksheet inventory

| Worksheet | Rows | Columns | Candidate header row |
|---|---:|---:|---:|
| Data glossary | 5 | 2 | 3 |
| HPV16_v1 | 81 | 15 | 1 |
| HPV16_v2 | 92 | 15 | 1 |
| HPV18_v1 | 84 | 15 | 1 |
| HPV18_v2 | 84 | 15 | 1 |
| HPV31_v1 | 94 | 13 | 1 |
| HPV31_v2 | 81 | 13 | 1 |
| HPV33_v1 | 92 | 13 | 1 |
| HPV33_v2 | 86 | 13 | 1 |
| HPV45_v1 | 92 | 13 | 1 |
| HPV45_v2 | 81 | 13 | 1 |
| HPV52_v1 | 92 | 13 | 1 |
| HPV52_v2 | 86 | 13 | 1 |
| HPV58_v1 | 92 | 13 | 1 |
| HPV58_v2 | 86 | 13 | 1 |
| BPV_v1 | 81 | 13 | 1 |
| BPV_v2 | 81 | 13 | 1 |

## Candidate variable categories

| Category | Cell hits | Worksheets | Preliminary status |
|---|---:|---|---|
| participant_identifier | 1 | Data glossary | Candidate identifier field detected |
| age | 0 | — | Not detected |
| weight | 0 | — | Not detected |
| bmi_adiposity | 0 | — | Not detected |
| sex_gender | 0 | — | Not detected |
| ethnicity | 0 | — | Not detected |
| dose_regimen | 18 | BPV_v1, BPV_v2, Data glossary, HPV16_v1, HPV16_v2, HPV18_v1, HPV18_v2, HPV31_v1, HPV31_v2, HPV33_v1, HPV33_v2, HPV45_v1, HPV45_v2, HPV52_v1, HPV52_v2, HPV58_v1, HPV58_v2 | Detected; participant-level availability requires structural confirmation |
| time_visit | 2 | Data glossary | Detected; participant-level availability requires structural confirmation |
| vaccine_type | 3 | Data glossary | Detected; participant-level availability requires structural confirmation |
| hpv_genotype | 0 | — | Not detected |
| antibody_isotype_subclass | 128 | BPV_v1, BPV_v2, HPV16_v1, HPV16_v2, HPV18_v1, HPV18_v2, HPV31_v1, HPV31_v2, HPV33_v1, HPV33_v2, HPV45_v1, HPV45_v2, HPV52_v1, HPV52_v2, HPV58_v1, HPV58_v2 | Detected; participant-level availability requires structural confirmation |
| fc_receptor | 48 | BPV_v1, BPV_v2, HPV16_v1, HPV16_v2, HPV18_v1, HPV18_v2, HPV31_v1, HPV31_v2, HPV33_v1, HPV33_v2, HPV45_v1, HPV45_v2, HPV52_v1, HPV52_v2, HPV58_v1, HPV58_v2 | Detected; participant-level availability requires structural confirmation |
| adcp | 4 | HPV16_v1, HPV16_v2, HPV18_v1, HPV18_v2 | Detected; participant-level availability requires structural confirmation |

## Critical feasibility decisions

- **Age:** no matching workbook field detected by the first-pass lexical audit.
- **Weight:** no matching workbook field detected by the first-pass lexical audit.
- **BMI/adiposity:** no matching workbook field detected by the first-pass lexical audit.
- **Sex/gender:** no matching workbook field detected by the first-pass lexical audit.
- **Ethnicity:** no matching workbook field detected by the first-pass lexical audit.
- **Dose/regimen:** candidate workbook content detected; inspect the candidate header and hit tables before classifying it as participant-level metadata.
- **Time/visit:** candidate workbook content detected; inspect the candidate header and hit tables before classifying it as participant-level metadata.

## Generated outputs

- `/mnt/d/HPV_Vaccine_Trafficome_Project/08_results/tables/phase1B_fiji_workbook_sheet_inventory.tsv`
- `/mnt/d/HPV_Vaccine_Trafficome_Project/08_results/tables/phase1B_fiji_candidate_headers.tsv`
- `/mnt/d/HPV_Vaccine_Trafficome_Project/08_results/tables/phase1B_fiji_candidate_metadata_hits.tsv`
- `/mnt/d/HPV_Vaccine_Trafficome_Project/02_dataset_audit/hpv_specific/fiji_nct02276521/sheet_previews`

## Interpretation rule

A keyword hit is not automatically a usable covariate. A variable will be classified as participant-level only after confirming that it is linked to a stable participant identifier, has sufficient nonmissing values, and is not merely part of a worksheet title, legend, assay label or explanatory note.
