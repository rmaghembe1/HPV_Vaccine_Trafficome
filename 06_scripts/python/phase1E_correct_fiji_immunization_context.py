#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError as exc:
    sys.exit(
        "ERROR: pandas is required.\n"
        "Install it with:\n"
        "  python -m pip install --user pandas\n"
        f"Original error: {exc}"
    )


ROOT = Path("/mnt/d/HPV_Vaccine_Trafficome_Project")

PROCESSED_DIR = ROOT / "07_data_processed" / "fiji_nct02276521"
TABLE_DIR = ROOT / "08_results" / "tables"
AUDIT_DIR = (
    ROOT
    / "02_dataset_audit"
    / "hpv_specific"
    / "fiji_nct02276521"
)

LONG_INPUT = (
    PROCESSED_DIR
    / "phase1C_fiji_participant_antigen_visit_feature_long.tsv"
)

PAIRED_INPUT = (
    PROCESSED_DIR
    / "phase1D_fiji_paired_feature_changes.tsv"
)

LONG_OUTPUT = (
    PROCESSED_DIR
    / "phase1E_fiji_long_immunization_context_corrected.tsv"
)

PAIRED_OUTPUT = (
    PROCESSED_DIR
    / "phase1E_fiji_paired_immunization_context_corrected.tsv"
)


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        sep="\t",
        index=False,
        na_rep="",
    )


def normalize_dose(series: pd.Series) -> pd.Series:
    dose = pd.to_numeric(
        series,
        errors="coerce",
    )

    if dose.isna().any():
        bad = (
            series[dose.isna()]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(
            "Unresolved dose values: "
            + ", ".join(bad)
        )

    if not dose.isin([0, 1, 2, 3]).all():
        bad = sorted(
            dose[~dose.isin([0, 1, 2, 3])]
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(
            "Unexpected dose values: "
            + ", ".join(map(str, bad))
        )

    return dose.astype(int)


def visit_context(dose: int, visit: str) -> str:
    if dose == 0 and visit == "v1":
        return "unvaccinated_pre_2vHPV_baseline"

    if dose == 0 and visit == "v2":
        return "primary_2vHPV_response_day28"

    if dose in {1, 2, 3} and visit == "v1":
        return "six_year_4vHPV_persistence"

    if dose in {1, 2, 3} and visit == "v2":
        return "heterologous_2vHPV_recall_day28"

    raise ValueError(
        f"Unsupported dose/visit combination: "
        f"dose={dose}, visit={visit}"
    )


def transition_context(dose: int) -> str:
    if dose == 0:
        return (
            "unvaccinated_baseline_to_"
            "primary_2vHPV_response"
        )

    return (
        "six_year_4vHPV_persistence_to_"
        "heterologous_2vHPV_recall"
    )


def prior_exposure(dose: int) -> str:
    if dose == 0:
        return "hpv_vaccine_naive"

    return "previously_4vHPV_vaccinated"


def main() -> None:
    for required_path in [
        LONG_INPUT,
        PAIRED_INPUT,
    ]:
        if not required_path.exists():
            sys.exit(
                f"ERROR: Required input missing: "
                f"{required_path}"
            )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    long_df = pd.read_csv(
        LONG_INPUT,
        sep="\t",
        dtype={
            "participant_id": "string",
            "previous_4vHPV_doses": "string",
            "visit": "string",
        },
    )

    paired_df = pd.read_csv(
        PAIRED_INPUT,
        sep="\t",
        dtype={
            "participant_id": "string",
            "previous_4vHPV_doses": "string",
        },
    )

    required_long = {
        "participant_id",
        "previous_4vHPV_doses",
        "visit",
        "antigen_target",
        "feature",
        "value",
    }

    required_paired = {
        "participant_id",
        "previous_4vHPV_doses",
        "antigen_target",
        "feature",
        "v1",
        "v2",
        "paired_complete",
    }

    missing_long = required_long - set(long_df.columns)
    missing_paired = required_paired - set(paired_df.columns)

    if missing_long:
        sys.exit(
            "ERROR: Long table missing columns: "
            + ", ".join(sorted(missing_long))
        )

    if missing_paired:
        sys.exit(
            "ERROR: Paired table missing columns: "
            + ", ".join(sorted(missing_paired))
        )

    long_df["previous_4vHPV_doses"] = normalize_dose(
        long_df["previous_4vHPV_doses"]
    )

    paired_df["previous_4vHPV_doses"] = normalize_dose(
        paired_df["previous_4vHPV_doses"]
    )

    long_df["visit"] = (
        long_df["visit"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    unexpected_visits = (
        set(long_df["visit"].unique())
        - {"v1", "v2"}
    )

    if unexpected_visits:
        sys.exit(
            "ERROR: Unexpected visits: "
            + ", ".join(sorted(unexpected_visits))
        )

    long_df["prior_4vHPV_exposure_status"] = [
        prior_exposure(int(dose))
        for dose in long_df["previous_4vHPV_doses"]
    ]

    long_df["immunization_context"] = [
        visit_context(
            int(dose),
            str(visit),
        )
        for dose, visit in zip(
            long_df["previous_4vHPV_doses"],
            long_df["visit"],
        )
    ]

    long_df["response_type"] = long_df[
        "immunization_context"
    ].map(
        {
            "unvaccinated_pre_2vHPV_baseline": (
                "unvaccinated_baseline"
            ),
            "primary_2vHPV_response_day28": (
                "primary_immunization"
            ),
            "six_year_4vHPV_persistence": (
                "long_term_persistence"
            ),
            "heterologous_2vHPV_recall_day28": (
                "memory_recall"
            ),
        }
    )

    paired_df["prior_4vHPV_exposure_status"] = [
        prior_exposure(int(dose))
        for dose in paired_df[
            "previous_4vHPV_doses"
        ]
    ]

    paired_df["response_transition"] = [
        transition_context(int(dose))
        for dose in paired_df[
            "previous_4vHPV_doses"
        ]
    ]

    paired_df["v1_biological_context"] = [
        (
            "unvaccinated_pre_2vHPV_baseline"
            if int(dose) == 0
            else "six_year_4vHPV_persistence"
        )
        for dose in paired_df[
            "previous_4vHPV_doses"
        ]
    ]

    paired_df["v2_biological_context"] = [
        (
            "primary_2vHPV_response_day28"
            if int(dose) == 0
            else "heterologous_2vHPV_recall_day28"
        )
        for dose in paired_df[
            "previous_4vHPV_doses"
        ]
    ]

    write_tsv(
        long_df,
        LONG_OUTPUT,
    )

    write_tsv(
        paired_df,
        PAIRED_OUTPUT,
    )

    context_summary = (
        long_df.groupby(
            [
                "previous_4vHPV_doses",
                "visit",
                "prior_4vHPV_exposure_status",
                "immunization_context",
                "response_type",
            ],
            dropna=False,
        )
        .agg(
            participants=(
                "participant_id",
                "nunique",
            ),
            antigen_targets=(
                "antigen_target",
                "nunique",
            ),
            assay_features=(
                "feature",
                "nunique",
            ),
            feature_observations=(
                "value",
                "size",
            ),
        )
        .reset_index()
    )

    transition_summary = (
        paired_df.groupby(
            [
                "previous_4vHPV_doses",
                "prior_4vHPV_exposure_status",
                "response_transition",
                "v1_biological_context",
                "v2_biological_context",
            ],
            dropna=False,
        )
        .agg(
            participants=(
                "participant_id",
                "nunique",
            ),
            paired_feature_records=(
                "paired_complete",
                "size",
            ),
            complete_paired_feature_records=(
                "paired_complete",
                lambda values: int(
                    (values == "yes").sum()
                ),
            ),
        )
        .reset_index()
    )

    write_tsv(
        context_summary,
        TABLE_DIR
        / "phase1E_fiji_immunization_context_summary.tsv",
    )

    write_tsv(
        transition_summary,
        TABLE_DIR
        / "phase1E_fiji_response_transition_summary.tsv",
    )

    expected_contexts = {
        "unvaccinated_pre_2vHPV_baseline",
        "primary_2vHPV_response_day28",
        "six_year_4vHPV_persistence",
        "heterologous_2vHPV_recall_day28",
    }

    observed_contexts = set(
        long_df["immunization_context"].unique()
    )

    missing_contexts = (
        expected_contexts - observed_contexts
    )

    unexpected_contexts = (
        observed_contexts - expected_contexts
    )

    decision = (
        "PASS"
        if not missing_contexts
        and not unexpected_contexts
        else "FAIL"
    )

    decision_table = pd.DataFrame(
        [
            {
                "decision": decision,
                "unique_participants": (
                    long_df[
                        "participant_id"
                    ].nunique()
                ),
                "long_feature_observations": (
                    len(long_df)
                ),
                "paired_feature_records": (
                    len(paired_df)
                ),
                "observed_contexts": ";".join(
                    sorted(observed_contexts)
                ),
                "missing_contexts": ";".join(
                    sorted(missing_contexts)
                ),
                "unexpected_contexts": ";".join(
                    sorted(unexpected_contexts)
                ),
            }
        ]
    )

    write_tsv(
        decision_table,
        TABLE_DIR
        / "phase1E_fiji_immunization_context_decision.tsv",
    )

    report_path = (
        AUDIT_DIR
        / "phase1E_fiji_immunization_context_correction.md"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as report:
        report.write(
            "# Phase 1E Fiji immunization-context correction\n\n"
        )

        report.write("## Decision\n\n")
        report.write(f"**{decision}**\n\n")

        report.write("## Biological correction\n\n")
        report.write(
            "Participants with zero previous 4vHPV doses "
            "were HPV-vaccine-naive at Visit 1. Their Visit 2 "
            "measurements therefore represent a primary response "
            "to 2vHPV, not a booster-memory or recall response.\n\n"
        )

        report.write(
            "Participants with one, two or three previous 4vHPV "
            "doses represent a distinct memory trajectory. Visit 1 "
            "captures immunity approximately six years after their "
            "last 4vHPV dose, whereas Visit 2 captures heterologous "
            "recall 28 days after 2vHPV administration.\n\n"
        )

        report.write("## Authoritative contexts\n\n")
        report.write(
            "| Previous 4vHPV doses | Visit | Interpretation |\n"
        )
        report.write("|---:|---|---|\n")
        report.write(
            "| 0 | v1 | Unvaccinated pre-2vHPV baseline |\n"
        )
        report.write(
            "| 0 | v2 | Primary 2vHPV response at day 28 |\n"
        )
        report.write(
            "| 1–3 | v1 | Six-year persistence after 4vHPV |\n"
        )
        report.write(
            "| 1–3 | v2 | Heterologous 2vHPV recall at day 28 |\n"
        )

        report.write("\n## Statistical consequence\n\n")
        report.write(
            "The visit effect must be interpreted conditionally "
            "on previous 4vHPV exposure. The visit-by-dose "
            "interaction distinguishes primary immune induction "
            "from memory persistence and recall and is therefore "
            "biologically more informative than a uniform Visit 2 "
            "booster effect.\n\n"
        )

        report.write("## Supersession\n\n")
        report.write(
            "This report supersedes earlier uniform descriptions "
            "of all Visit 1 observations as persistence and all "
            "Visit 2 observations as recall. Numerical results from "
            "Phases 1C and 1D remain valid.\n"
        )

    print("===== PHASE 1E COMPLETE =====")
    print(f"Decision: {decision}")
    print(
        "Unique participants: "
        f"{long_df['participant_id'].nunique()}"
    )
    print(
        "Long feature observations: "
        f"{len(long_df)}"
    )
    print(
        "Paired feature records: "
        f"{len(paired_df)}"
    )
    print(
        "Contexts: "
        + ", ".join(sorted(observed_contexts))
    )
    print(f"Report: {report_path}")

    if decision != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
