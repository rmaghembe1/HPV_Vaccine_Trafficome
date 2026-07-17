#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/d/HPV_Vaccine_Trafficome_Project"

if [ "$(pwd)" != "$ROOT" ]; then
    echo "ERROR: Run from $ROOT"
    echo "Current directory: $(pwd)"
    exit 1
fi

DATA_DIR="$ROOT/03_data_raw/hpv_specific/fiji_nct02276521"
CODE_DIR="$DATA_DIR/source_code"
META_DIR="$ROOT/02_dataset_audit/hpv_specific/fiji_nct02276521"
LOG_DIR="$ROOT/10_reproducibility/logs"

mkdir -p "$DATA_DIR" "$CODE_DIR" "$META_DIR" "$LOG_DIR"

WORKBOOK="$DATA_DIR/NCOMMS-24-64334A_HPV_collated_antibody_feature_data.xlsx"

download_file() {
    local destination="$1"
    shift

    if [ -s "$destination" ]; then
        echo "Already present: $destination"
        return 0
    fi

    local temporary="${destination}.part"
    rm -f "$temporary"

    for url in "$@"; do
        echo "Downloading:"
        echo "  $url"

        if curl \
            --fail \
            --location \
            --retry 8 \
            --retry-delay 5 \
            --retry-all-errors \
            --connect-timeout 30 \
            --max-time 600 \
            --user-agent "Mozilla/5.0 HPV-Vaccine-Trafficome/1.0" \
            "$url" \
            --output "$temporary"
        then
            mv "$temporary" "$destination"
            echo "Saved: $destination"
            return 0
        fi

        rm -f "$temporary"
        echo "Download route failed; trying fallback."
    done

    echo "ERROR: All download routes failed for $destination"
    exit 1
}

echo "===== FIJI DATASET ACQUISITION ====="

download_file \
    "$WORKBOOK" \
    "https://zenodo.org/records/14848069/files/NCOMMS-24-64334A%20HPV%20collated%20antibody%20feature%20data.xlsx?download=1" \
    "https://zenodo.org/api/records/14848069/files/NCOMMS-24-64334A%20HPV%20collated%20antibody%20feature%20data.xlsx/content"

download_file \
    "$CODE_DIR/countmember.m" \
    "https://zenodo.org/records/14848092/files/countmember.m?download=1" \
    "https://zenodo.org/api/records/14848092/files/countmember.m/content"

download_file \
    "$CODE_DIR/Figure_3_polar_plot_code.R" \
    "https://zenodo.org/records/14848092/files/Figure%203%20polar%20plot%20code.R?download=1" \
    "https://zenodo.org/api/records/14848092/files/Figure%203%20polar%20plot%20code.R/content"

download_file \
    "$CODE_DIR/Figure_4_Supp_Fig_9_polar_plot_code.R" \
    "https://zenodo.org/records/14848092/files/Figure%204-Supp%20Fig%209%20polar%20plot%20code.R?download=1" \
    "https://zenodo.org/api/records/14848092/files/Figure%204-Supp%20Fig%209%20polar%20plot%20code.R/content"

download_file \
    "$CODE_DIR/lassotest.m" \
    "https://zenodo.org/records/14848092/files/lassotest.m?download=1" \
    "https://zenodo.org/api/records/14848092/files/lassotest.m/content"

cat > "$META_DIR/source_record.md" <<'EOF'
# Fiji HPV vaccine systems-serology resource

## Study

Systems serology analysis shows IgG1 and IgG3 memory responses six
years after one dose of quadrivalent HPV vaccine.

Clinical trial: NCT02276521

Article DOI: 10.1038/s41467-025-57443-z

## Data record

Zenodo record: 14848069  
DOI: 10.5281/zenodo.14848069  
License: CC BY 4.0

Deposited workbook:

- NCOMMS-24-64334A HPV collated antibody feature data.xlsx
- Expected MD5: e42173e1d8297cd64420fd9682c42674

## Code record

Zenodo record: 14848092  
DOI: 10.5281/zenodo.14848092  
License: CC BY 4.0

Deposited source files:

- countmember.m
- Figure 3 polar plot code.R
- Figure 4-Supp Fig 9 polar plot code.R
- lassotest.m

## Intended project role

This is the first directly reanalyzable HPV-vaccine dataset in the
HPV Vaccine Trafficome Project. It will be used to reconstruct
dose-history-associated antibody persistence, recall, Fc-receptor
engagement, cross-reactive breadth and antibody-dependent cellular
phagocytosis.
EOF

cat > "$META_DIR/expected_md5.tsv" <<'EOF'
record_id	expected_md5	local_file
14848069	e42173e1d8297cd64420fd9682c42674	03_data_raw/hpv_specific/fiji_nct02276521/NCOMMS-24-64334A_HPV_collated_antibody_feature_data.xlsx
14848092	72d29ed3ef23f6f9961f0d58445fc786	03_data_raw/hpv_specific/fiji_nct02276521/source_code/countmember.m
14848092	4f86c67429b16a69656e7d0f43e9f96b	03_data_raw/hpv_specific/fiji_nct02276521/source_code/Figure_3_polar_plot_code.R
14848092	67d17c68469a3479a3881265ab157f1e	03_data_raw/hpv_specific/fiji_nct02276521/source_code/Figure_4_Supp_Fig_9_polar_plot_code.R
14848092	fc52df9ca5963ce00db37dc50e43cbc8	03_data_raw/hpv_specific/fiji_nct02276521/source_code/lassotest.m
EOF

echo
echo "===== FILE INTEGRITY ====="

printf '%s  %s\n' \
    "e42173e1d8297cd64420fd9682c42674" \
    "$WORKBOOK" | md5sum -c -

printf '%s  %s\n' \
    "72d29ed3ef23f6f9961f0d58445fc786" \
    "$CODE_DIR/countmember.m" | md5sum -c -

printf '%s  %s\n' \
    "4f86c67429b16a69656e7d0f43e9f96b" \
    "$CODE_DIR/Figure_3_polar_plot_code.R" | md5sum -c -

printf '%s  %s\n' \
    "67d17c68469a3479a3881265ab157f1e" \
    "$CODE_DIR/Figure_4_Supp_Fig_9_polar_plot_code.R" | md5sum -c -

printf '%s  %s\n' \
    "fc52df9ca5963ce00db37dc50e43cbc8" \
    "$CODE_DIR/lassotest.m" | md5sum -c -

echo
echo "===== ACQUIRED FILES ====="
find "$DATA_DIR" -maxdepth 2 -type f \
    -printf '%P\t%s bytes\n' \
    | sort

echo
echo "Fiji systems-serology acquisition completed."
