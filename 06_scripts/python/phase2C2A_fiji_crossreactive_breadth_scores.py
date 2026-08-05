#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/mnt/d/HPV_Vaccine_Trafficome_Project")

PROCESSED = (
    ROOT
    / "07_data_processed"
    / "fiji_nct02276521"
)

TABLES = ROOT / "08_results" / "tables"

REPORTS = (
    ROOT
    / "02_dataset_audit"
    / "hpv_specific"
    / "fiji_nct02276521"
)

PAIRED_INPUT = (
    PROCESSED
    / "phase2A_fiji_paired_effects_analysis_ready.tsv"
)

CALIBRATED_INPUT = (
    PROCESSED
    / "phase2C1_fiji_bpv_calibrated_effects_long.tsv"
)

METADATA_INPUT = (
    PROCESSED
    / "phase2C1_fiji_participant_metadata.tsv"
)

C1_DECISION_INPUT = (
    TABLES
    / "phase2C1_fiji_multivariate_readiness_decision.tsv"
)

EXPECTED_C1_DECISION = (
    "READY_FOR_PHASE2C2_BREADTH_AND_PHASE2C3_FUNCTIONAL_COUPLING"
)

CROSS_REACTIVE_ANTIGENS = [
    "HPV31",
    "HPV33",
    "HPV45",
    "HPV52",
    "HPV58",
]

SHARED_FEATURES = [
    "IgG",
    "IgM",
    "IgA1",
    "IgA2",
    "IgG1",
    "IgG2",
    "IgG3",
    "IgG4",
    "FcgR2A",
    "FcgR2B",
    "FcgR3A",
]

AXES = {
    "binding_antibody_abundance": [
        "IgG",
        "IgM",
        "IgA1",
        "IgA2",
    ],
    "igg_subclass_architecture": [
        "IgG1",
        "IgG2",
        "IgG3",
        "IgG4",
    ],
    "fc_receptor_communication": [
        "FcgR2A",
        "FcgR2B",
        "FcgR3A",
    ],
    "global_shared_serology": SHARED_FEATURES,
}


def require_columns(
    frame: pd.DataFrame,
    required: set[str],
    label: str,
) -> None:
    missing = required - set(frame.columns)

    if missing:
        sys.exit(
            f"ERROR: {label} is missing columns: "
            + ", ".join(sorted(missing))
        )


def write_tsv(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        path,
        sep="\t",
        index=False,
        na_rep="",
    )


def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    for path in [
        PAIRED_INPUT,
        CALIBRATED_INPUT,
        METADATA_INPUT,
        C1_DECISION_INPUT,
    ]:
        if not path.exists():
            sys.exit(
                f"ERROR: Required input is missing: {path}"
            )

    decision = pd.read_csv(
        C1_DECISION_INPUT,
        sep="\t",
    )

    observed = str(
        decision.loc[0, "decision"]
    )

    if observed != EXPECTED_C1_DECISION:
        sys.exit(
            "ERROR: Phase 2C1 decision is "
            f"{observed}; expected "
            f"{EXPECTED_C1_DECISION}."
        )

    paired = pd.read_csv(
        PAIRED_INPUT,
        sep="\t",
        dtype={
            "participant_id": "string",
        },
    )

    calibrated = pd.read_csv(
        CALIBRATED_INPUT,
        sep="\t",
        dtype={
            "participant_id": "string",
        },
    )

    metadata = pd.read_csv(
        METADATA_INPUT,
        sep="\t",
        dtype={
            "participant_id": "string",
        },
    )

    require_columns(
        paired,
        {
            "participant_id",
            "previous_4vHPV_doses",
            "antigen_target",
            "feature",
            "log2_change_authoritative",
        },
        "Paired-effect table",
    )

    require_columns(
        calibrated,
        {
            "participant_id",
            "previous_4vHPV_doses",
            "antigen_target",
            "feature",
            "bpv_calibrated_log2_change",
        },
        "BPV-calibrated long table",
    )

    require_columns(
        metadata,
        {
            "participant_id",
            "previous_4vHPV_doses",
            "analysis_context",
        },
        "Participant metadata",
    )

    return paired, calibrated, metadata


def construct_long_scores(
    paired: pd.DataFrame,
    calibrated: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    raw = paired[
        paired["antigen_target"].isin(
            CROSS_REACTIVE_ANTIGENS
        )
        & paired["feature"].isin(
            SHARED_FEATURES
        )
    ][
        [
            "participant_id",
            "previous_4vHPV_doses",
            "antigen_target",
            "feature",
            "log2_change_authoritative",
        ]
    ].rename(
        columns={
            "log2_change_authoritative": "response_score",
        }
    )

    raw["matrix_type"] = "raw_log2_change"

    bpv = calibrated[
        calibrated["antigen_target"].isin(
            CROSS_REACTIVE_ANTIGENS
        )
        & calibrated["feature"].isin(
            SHARED_FEATURES
        )
    ][
        [
            "participant_id",
            "previous_4vHPV_doses",
            "antigen_target",
            "feature",
            "bpv_calibrated_log2_change",
        ]
    ].rename(
        columns={
            "bpv_calibrated_log2_change": "response_score",
        }
    )

    bpv["matrix_type"] = (
        "bpv_calibrated_log2_change"
    )

    combined = pd.concat(
        [
            raw,
            bpv,
        ],
        ignore_index=True,
    )

    combined[
        "previous_4vHPV_doses"
    ] = pd.to_numeric(
        combined[
            "previous_4vHPV_doses"
        ],
        errors="raise",
    ).astype(int)

    metadata_small = metadata[
        [
            "participant_id",
            "previous_4vHPV_doses",
            "analysis_context",
        ]
    ].drop_duplicates()

    metadata_small[
        "previous_4vHPV_doses"
    ] = pd.to_numeric(
        metadata_small[
            "previous_4vHPV_doses"
        ],
        errors="raise",
    ).astype(int)

    combined = combined.merge(
        metadata_small,
        on=[
            "participant_id",
            "previous_4vHPV_doses",
        ],
        how="left",
        validate="many_to_one",
    )

    duplicates = combined.duplicated(
        subset=[
            "matrix_type",
            "participant_id",
            "antigen_target",
            "feature",
        ],
        keep=False,
    )

    if duplicates.any():
        sys.exit(
            "ERROR: Duplicate participant-antigen-feature "
            "records were detected."
        )

    return combined.sort_values(
        [
            "matrix_type",
            "previous_4vHPV_doses",
            "participant_id",
            "antigen_target",
            "feature",
        ]
    ).reset_index(
        drop=True
    )


def construct_antigen_axis_scores(
    long_scores: pd.DataFrame,
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []

    for axis, features in AXES.items():
        subset = long_scores[
            long_scores["feature"].isin(
                features
            )
        ]

        axis_scores = (
            subset.groupby(
                [
                    "matrix_type",
                    "participant_id",
                    "previous_4vHPV_doses",
                    "analysis_context",
                    "antigen_target",
                ],
                observed=True,
            )
            .agg(
                antigen_axis_score=(
                    "response_score",
                    "mean",
                ),
                antigen_axis_median=(
                    "response_score",
                    "median",
                ),
                contributing_features=(
                    "feature",
                    "nunique",
                ),
            )
            .reset_index()
        )

        axis_scores["mechanistic_axis"] = axis

        outputs.append(axis_scores)

    return pd.concat(
        outputs,
        ignore_index=True,
    ).sort_values(
        [
            "matrix_type",
            "mechanistic_axis",
            "previous_4vHPV_doses",
            "participant_id",
            "antigen_target",
        ]
    ).reset_index(
        drop=True
    )


def construct_breadth_scores(
    antigen_axis: pd.DataFrame,
) -> pd.DataFrame:
    breadth = (
        antigen_axis.groupby(
            [
                "matrix_type",
                "participant_id",
                "previous_4vHPV_doses",
                "analysis_context",
                "mechanistic_axis",
            ],
            observed=True,
        )
        .agg(
            evaluated_antigens=(
                "antigen_target",
                "nunique",
            ),
            mean_cross_reactive_score=(
                "antigen_axis_score",
                "mean",
            ),
            median_cross_reactive_score=(
                "antigen_axis_score",
                "median",
            ),
            minimum_cross_reactive_score=(
                "antigen_axis_score",
                "min",
            ),
            maximum_cross_reactive_score=(
                "antigen_axis_score",
                "max",
            ),
            cross_antigen_standard_deviation=(
                "antigen_axis_score",
                "std",
            ),
            positive_antigen_count=(
                "antigen_axis_score",
                lambda values: int(
                    (
                        pd.Series(values)
                        > 0
                    ).sum()
                ),
            ),
            twofold_antigen_count=(
                "antigen_axis_score",
                lambda values: int(
                    (
                        pd.Series(values)
                        > 1
                    ).sum()
                ),
            ),
        )
        .reset_index()
    )

    breadth[
        "positive_antigen_fraction"
    ] = (
        breadth[
            "positive_antigen_count"
        ]
        / breadth[
            "evaluated_antigens"
        ]
    )

    breadth[
        "twofold_antigen_fraction"
    ] = (
        breadth[
            "twofold_antigen_count"
        ]
        / breadth[
            "evaluated_antigens"
        ]
    )

    return breadth.sort_values(
        [
            "matrix_type",
            "mechanistic_axis",
            "previous_4vHPV_doses",
            "participant_id",
        ]
    ).reset_index(
        drop=True
    )


def expand_analysis_strata(
    breadth: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for record in breadth.to_dict(
        orient="records"
    ):
        dose = int(
            record[
                "previous_4vHPV_doses"
            ]
        )

        if dose == 0:
            primary = dict(record)
            primary[
                "analysis_stratum"
            ] = "primary_dose0"

            rows.append(primary)
        else:
            recall = dict(record)
            recall[
                "analysis_stratum"
            ] = "recall_all_doses"

            dose_specific = dict(record)
            dose_specific[
                "analysis_stratum"
            ] = f"recall_dose{dose}"

            rows.extend(
                [
                    recall,
                    dose_specific,
                ]
            )

    return pd.DataFrame(rows)


def summarize_breadth(
    expanded: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "mean_cross_reactive_score",
        "median_cross_reactive_score",
        "minimum_cross_reactive_score",
        "positive_antigen_count",
        "twofold_antigen_count",
        "positive_antigen_fraction",
        "twofold_antigen_fraction",
    ]

    rows: list[dict[str, object]] = []

    grouped = expanded.groupby(
        [
            "matrix_type",
            "mechanistic_axis",
            "analysis_stratum",
        ],
        observed=True,
    )

    for keys, group in grouped:
        (
            matrix_type,
            mechanistic_axis,
            analysis_stratum,
        ) = keys

        for metric in metrics:
            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            ).dropna()

            rows.append(
                {
                    "matrix_type": matrix_type,
                    "mechanistic_axis": (
                        mechanistic_axis
                    ),
                    "analysis_stratum": (
                        analysis_stratum
                    ),
                    "metric": metric,
                    "participants": len(values),
                    "mean": float(
                        values.mean()
                    ),
                    "standard_deviation": float(
                        values.std(
                            ddof=1
                        )
                    ),
                    "median": float(
                        values.median()
                    ),
                    "first_quartile": float(
                        values.quantile(
                            0.25
                        )
                    ),
                    "third_quartile": float(
                        values.quantile(
                            0.75
                        )
                    ),
                    "minimum": float(
                        values.min()
                    ),
                    "maximum": float(
                        values.max()
                    ),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    paired, calibrated, metadata = load_inputs()

    long_scores = construct_long_scores(
        paired,
        calibrated,
        metadata,
    )

    antigen_axis = (
        construct_antigen_axis_scores(
            long_scores
        )
    )

    breadth = construct_breadth_scores(
        antigen_axis
    )

    expanded = expand_analysis_strata(
        breadth
    )

    summary = summarize_breadth(
        expanded
    )

    failures: list[str] = []

    expected_counts = {
        "long_scores": 8800,
        "antigen_axis": 3200,
        "breadth": 640,
        "expanded": 1120,
        "summary": 280,
    }

    observed_counts = {
        "long_scores": len(
            long_scores
        ),
        "antigen_axis": len(
            antigen_axis
        ),
        "breadth": len(
            breadth
        ),
        "expanded": len(
            expanded
        ),
        "summary": len(
            summary
        ),
    }

    for label, expected in expected_counts.items():
        observed = observed_counts[label]

        if observed != expected:
            failures.append(
                f"{label}: expected {expected}, "
                f"observed {observed}."
            )

    if long_scores[
        "response_score"
    ].isna().any():
        failures.append(
            "Long-score table contains missing responses."
        )

    if antigen_axis[
        "antigen_axis_score"
    ].isna().any():
        failures.append(
            "Antigen-axis table contains missing scores."
        )

    if not (
        breadth[
            "evaluated_antigens"
        ]
        == 5
    ).all():
        failures.append(
            "Some breadth scores do not contain five "
            "cross-reactive HPV types."
        )

    if not breadth[
        "positive_antigen_count"
    ].between(
        0,
        5,
    ).all():
        failures.append(
            "Positive-antigen counts are outside 0-5."
        )

    if not breadth[
        "twofold_antigen_count"
    ].between(
        0,
        5,
    ).all():
        failures.append(
            "Twofold-antigen counts are outside 0-5."
        )

    decision_value = (
        "READY_FOR_PHASE2C2B_CROSSREACTIVE_BREADTH_INFERENCE"
        if not failures
        else "PHASE2C2A_REPAIR_REQUIRED"
    )

    long_output = (
        PROCESSED
        / "phase2C2A_fiji_crossreactive_long_scores.tsv"
    )

    antigen_axis_output = (
        PROCESSED
        / "phase2C2A_fiji_crossreactive_antigen_axis_scores.tsv"
    )

    breadth_output = (
        PROCESSED
        / "phase2C2A_fiji_crossreactive_breadth_scores.tsv"
    )

    summary_output = (
        TABLES
        / "phase2C2A_fiji_crossreactive_breadth_summary.tsv"
    )

    decision_output = (
        TABLES
        / "phase2C2A_fiji_crossreactive_breadth_decision.tsv"
    )

    write_tsv(
        long_scores,
        long_output,
    )

    write_tsv(
        antigen_axis,
        antigen_axis_output,
    )

    write_tsv(
        breadth,
        breadth_output,
    )

    write_tsv(
        summary,
        summary_output,
    )

    decision = pd.DataFrame(
        [
            {
                "decision": decision_value,
                "cross_reactive_antigens": 5,
                "shared_features": 11,
                "mechanistic_axes": 4,
                "long_score_rows": len(
                    long_scores
                ),
                "antigen_axis_rows": len(
                    antigen_axis
                ),
                "participant_breadth_rows": len(
                    breadth
                ),
                "expanded_stratum_rows": len(
                    expanded
                ),
                "descriptive_summary_rows": len(
                    summary
                ),
                "missing_response_scores": int(
                    long_scores[
                        "response_score"
                    ].isna().sum()
                ),
                "validation_failures": "; ".join(
                    failures
                ),
            }
        ]
    )

    write_tsv(
        decision,
        decision_output,
    )

    report_path = (
        REPORTS
        / "phase2C2A_fiji_crossreactive_breadth_report.md"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as report:
        report.write(
            "# Phase 2C2A Fiji cross-reactive breadth\n\n"
        )

        report.write(
            "## Decision\n\n"
        )

        report.write(
            f"**{decision_value}**\n\n"
        )

        report.write(
            "- Cross-reactive HPV types: "
            "HPV31, HPV33, HPV45, HPV52 and HPV58\n"
        )

        report.write(
            "- Shared systems-serology features: 11\n"
        )

        report.write(
            "- Mechanistic axes: binding-antibody abundance, "
            "IgG-subclass architecture, Fc-receptor communication "
            "and global shared serology\n"
        )

        report.write(
            f"- Participant-level breadth rows: "
            f"{len(breadth)}\n\n"
        )

        report.write(
            "Breadth is represented by the mean, median, minimum, "
            "positive-antigen count and twofold-antigen count across "
            "the five cross-reactive HPV types. Raw and matched "
            "BPV-calibrated scores are retained separately. The "
            "BPV-calibrated scores provide the preferred basis for "
            "HPV-associated cross-reactive interpretation.\n"
        )

    print(
        "===== PHASE 2C2A COMPLETE ====="
    )

    print(
        f"Decision: {decision_value}"
    )

    print(
        f"Long response rows: {len(long_scores)}"
    )

    print(
        f"Antigen-axis rows: {len(antigen_axis)}"
    )

    print(
        f"Participant breadth rows: {len(breadth)}"
    )

    print(
        f"Descriptive summary rows: {len(summary)}"
    )

    print(
        f"Report: {report_path}"
    )

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
