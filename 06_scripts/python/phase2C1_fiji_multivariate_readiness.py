#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    sys.exit(
        "ERROR: numpy and pandas are required.\n"
        f"Original error: {exc}"
    )


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

EVIDENCE_INPUT = (
    TABLES
    / "phase2B2D1_fiji_integrated_evidence_registry.tsv"
)

SYNTHESIS_DECISION = (
    TABLES
    / "phase2B2D2_fiji_biological_synthesis_decision.tsv"
)

EXPECTED_SYNTHESIS_DECISION = (
    "READY_FOR_PHASE2B2D_COMMIT_AND_PHASE2C_MULTIVARIATE_ANALYSIS"
)

SEVERITY_ORDER = {
    "none": 0,
    "low": 1,
    "moderate": 2,
    "high": 3,
}


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


def require_columns(
    frame: pd.DataFrame,
    columns: set[str],
    label: str,
) -> None:
    missing = columns - set(
        frame.columns
    )

    if missing:
        sys.exit(
            f"ERROR: {label} is missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )


def maximum_severity(
    values: pd.Series,
) -> str:
    valid = [
        str(value)
        for value in values
        if pd.notna(value)
        and str(value) in SEVERITY_ORDER
    ]

    if not valid:
        return "unresolved"

    return max(
        valid,
        key=lambda value: SEVERITY_ORDER[value],
    )


def sorted_dose_string(
    values: pd.Series,
) -> str:
    doses = sorted(
        {
            int(value)
            for value in values
            if pd.notna(value)
        }
    )

    return ",".join(
        str(value)
        for value in doses
    )


def construct_metadata(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    participant_checks = (
        paired.groupby(
            "participant_id",
            observed=True,
        )
        .agg(
            dose_count=(
                "previous_4vHPV_doses",
                "nunique",
            ),
            trajectory_count=(
                "trajectory",
                "nunique",
            ),
            prior_status_count=(
                "prior_4vHPV_exposure_status",
                "nunique",
            ),
        )
        .reset_index()
    )

    invalid = participant_checks[
        (
            participant_checks["dose_count"] != 1
        )
        | (
            participant_checks["trajectory_count"] != 1
        )
        | (
            participant_checks["prior_status_count"] != 1
        )
    ]

    if not invalid.empty:
        sys.exit(
            "ERROR: Participant metadata is not unique."
        )

    metadata = (
        paired[
            [
                "participant_id",
                "previous_4vHPV_doses",
                "prior_4vHPV_exposure_status",
                "trajectory",
                "v1_biological_context",
                "v2_biological_context",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    metadata[
        "previous_4vHPV_doses"
    ] = pd.to_numeric(
        metadata[
            "previous_4vHPV_doses"
        ],
        errors="raise",
    ).astype(int)

    metadata[
        "analysis_context"
    ] = np.where(
        metadata[
            "previous_4vHPV_doses"
        ]
        == 0,
        "primary_2vHPV_induction",
        "heterologous_2vHPV_recall",
    )

    metadata = metadata.sort_values(
        [
            "previous_4vHPV_doses",
            "participant_id",
        ]
    ).reset_index(
        drop=True
    )

    return metadata


def construct_raw_matrix(
    paired: pd.DataFrame,
    participant_order: list[str],
) -> pd.DataFrame:
    raw_long = paired.copy()

    raw_long[
        "variable_id"
    ] = (
        raw_long[
            "antigen_target"
        ].astype(str)
        + "__"
        + raw_long[
            "feature"
        ].astype(str)
    )

    duplicate_count = int(
        raw_long.duplicated(
            subset=[
                "participant_id",
                "variable_id",
            ],
            keep=False,
        ).sum()
    )

    if duplicate_count:
        sys.exit(
            "ERROR: Raw matrix contains duplicate "
            "participant-variable records."
        )

    matrix = raw_long.pivot(
        index="participant_id",
        columns="variable_id",
        values="log2_change_authoritative",
    )

    matrix = matrix.reindex(
        participant_order
    )

    matrix = matrix.reindex(
        sorted(matrix.columns),
        axis=1,
    )

    matrix.columns.name = None

    return matrix.reset_index()


def construct_bpv_calibrated_matrix(
    paired: pd.DataFrame,
    participant_order: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bpv = paired[
        paired[
            "antigen_target"
        ]
        == "BPV"
    ][
        [
            "participant_id",
            "previous_4vHPV_doses",
            "feature",
            "log2_change_authoritative",
        ]
    ].rename(
        columns={
            "log2_change_authoritative": (
                "bpv_log2_change"
            ),
        }
    )

    bpv_features = sorted(
        bpv[
            "feature"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    hpv = paired[
        (
            paired[
                "antigen_target"
            ]
            != "BPV"
        )
        & (
            paired[
                "feature"
            ].isin(
                bpv_features
            )
        )
    ].copy()

    calibrated_long = hpv.merge(
        bpv,
        on=[
            "participant_id",
            "previous_4vHPV_doses",
            "feature",
        ],
        how="inner",
        validate="many_to_one",
    )

    calibrated_long[
        "bpv_calibrated_log2_change"
    ] = (
        calibrated_long[
            "log2_change_authoritative"
        ]
        - calibrated_long[
            "bpv_log2_change"
        ]
    )

    calibrated_long[
        "variable_id"
    ] = (
        calibrated_long[
            "antigen_target"
        ].astype(str)
        + "__"
        + calibrated_long[
            "feature"
        ].astype(str)
    )

    duplicate_count = int(
        calibrated_long.duplicated(
            subset=[
                "participant_id",
                "variable_id",
            ],
            keep=False,
        ).sum()
    )

    if duplicate_count:
        sys.exit(
            "ERROR: BPV-calibrated matrix contains "
            "duplicate participant-variable records."
        )

    matrix = calibrated_long.pivot(
        index="participant_id",
        columns="variable_id",
        values="bpv_calibrated_log2_change",
    )

    matrix = matrix.reindex(
        participant_order
    )

    matrix = matrix.reindex(
        sorted(matrix.columns),
        axis=1,
    )

    matrix.columns.name = None

    return (
        matrix.reset_index(),
        calibrated_long,
    )


def matrix_audit_row(
    matrix_name: str,
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    participant_mask: pd.Series,
    analysis_scope: str,
) -> dict[str, object]:
    participants = metadata.loc[
        participant_mask,
        "participant_id",
    ].tolist()

    subset = (
        matrix.set_index(
            "participant_id"
        )
        .reindex(
            participants
        )
    )

    numeric = subset.apply(
        pd.to_numeric,
        errors="coerce",
    )

    missing_values = int(
        numeric.isna().sum().sum()
    )

    total_values = int(
        numeric.shape[0]
        * numeric.shape[1]
    )

    variance = numeric.var(
        axis=0,
        ddof=1,
    )

    zero_variance_variables = int(
        (
            variance.fillna(0)
            <= 1e-15
        ).sum()
    )

    participants_count = int(
        numeric.shape[0]
    )

    variables_count = int(
        numeric.shape[1]
    )

    maximum_pca_components = max(
        min(
            participants_count - 1,
            variables_count,
        ),
        0,
    )

    if participants_count < 10:
        readiness = (
            "insufficient_sample_size"
        )
    elif missing_values:
        readiness = (
            "requires_missing_data_resolution"
        )
    elif zero_variance_variables:
        readiness = (
            "ready_after_zero_variance_filter"
        )
    else:
        readiness = (
            "ready_for_descriptive_multivariate_analysis"
        )

    return {
        "matrix_name": matrix_name,
        "analysis_scope": analysis_scope,
        "participants": participants_count,
        "variables": variables_count,
        "participant_to_variable_ratio": (
            participants_count
            / variables_count
            if variables_count
            else np.nan
        ),
        "variable_to_participant_ratio": (
            variables_count
            / participants_count
            if participants_count
            else np.nan
        ),
        "total_matrix_values": total_values,
        "missing_values": missing_values,
        "missing_fraction": (
            missing_values
            / total_values
            if total_values
            else np.nan
        ),
        "zero_variance_variables": (
            zero_variance_variables
        ),
        "maximum_pca_components": (
            maximum_pca_components
        ),
        "readiness": readiness,
    }


def build_feature_manifest(
    paired: pd.DataFrame,
    evidence: pd.DataFrame,
    raw_matrix: pd.DataFrame,
    calibrated_matrix: pd.DataFrame,
) -> pd.DataFrame:
    feature_metadata = (
        paired.groupby(
            [
                "antigen_target",
                "feature",
            ],
            observed=True,
        )
        .agg(
            antigen_class=(
                "antigen_class",
                "first",
            ),
            assay_family=(
                "assay_family",
                "first",
            ),
            outcome_family=(
                "outcome_family",
                "first",
            ),
            maximum_paired_floor_severity=(
                "paired_floor_severity",
                maximum_severity,
            ),
        )
        .reset_index()
    )

    evidence_summary = (
        evidence.groupby(
            [
                "antigen_target",
                "feature",
            ],
            observed=True,
        )
        .agg(
            evaluated_trajectories=(
                "evidence_id",
                "size",
            ),
            raw_fdr_significant_trajectories=(
                "raw_fdr_significant",
                lambda values: int(
                    pd.Series(values)
                    .astype(str)
                    .str.lower()
                    .eq("true")
                    .sum()
                ),
            ),
            claim_ready_trajectories=(
                "claim_readiness",
                lambda values: int(
                    (
                        pd.Series(values)
                        == "claim_ready"
                    ).sum()
                ),
            ),
            qualified_trajectories=(
                "claim_readiness",
                lambda values: int(
                    (
                        pd.Series(values)
                        == "qualified_claim_only"
                    ).sum()
                ),
            ),
            primary_claim_ready=(
                "claim_readiness",
                lambda values: "",
            ),
        )
        .drop(
            columns=[
                "primary_claim_ready",
            ]
        )
        .reset_index()
    )

    primary_claim = (
        evidence[
            (
                evidence[
                    "analysis_context"
                ]
                == "primary_2vHPV_induction"
            )
            & (
                evidence[
                    "claim_readiness"
                ]
                == "claim_ready"
            )
        ][
            [
                "antigen_target",
                "feature",
            ]
        ]
        .drop_duplicates()
        .assign(
            primary_claim_ready="yes"
        )
    )

    recall_claim = (
        evidence[
            (
                evidence[
                    "analysis_context"
                ]
                == "heterologous_2vHPV_recall"
            )
            & (
                evidence[
                    "claim_readiness"
                ]
                == "claim_ready"
            )
        ]
        .groupby(
            [
                "antigen_target",
                "feature",
            ],
            observed=True,
        )[
            "previous_4vHPV_doses"
        ]
        .apply(
            sorted_dose_string
        )
        .rename(
            "recall_claim_ready_doses"
        )
        .reset_index()
    )

    base = feature_metadata.merge(
        evidence_summary,
        on=[
            "antigen_target",
            "feature",
        ],
        how="left",
        validate="one_to_one",
    )

    base = base.merge(
        primary_claim,
        on=[
            "antigen_target",
            "feature",
        ],
        how="left",
        validate="one_to_one",
    )

    base = base.merge(
        recall_claim,
        on=[
            "antigen_target",
            "feature",
        ],
        how="left",
        validate="one_to_one",
    )

    base[
        "primary_claim_ready"
    ] = base[
        "primary_claim_ready"
    ].fillna(
        "no"
    )

    base[
        "recall_claim_ready_doses"
    ] = base[
        "recall_claim_ready_doses"
    ].fillna(
        ""
    )

    manifest_rows: list[
        dict[str, object]
    ] = []

    matrix_specs = [
        (
            "raw_log2_change",
            raw_matrix,
        ),
        (
            "bpv_calibrated_log2_change",
            calibrated_matrix,
        ),
    ]

    for matrix_type, matrix in matrix_specs:
        matrix_values = matrix.set_index(
            "participant_id"
        )

        for variable_id in matrix_values.columns:
            antigen_target, feature = (
                str(variable_id)
                .split(
                    "__",
                    1,
                )
            )

            metadata_row = base[
                (
                    base[
                        "antigen_target"
                    ]
                    == antigen_target
                )
                & (
                    base[
                        "feature"
                    ]
                    == feature
                )
            ]

            if metadata_row.empty:
                sys.exit(
                    "ERROR: Feature metadata is absent for "
                    f"{variable_id}."
                )

            metadata = metadata_row.iloc[0]

            values = pd.to_numeric(
                matrix_values[
                    variable_id
                ],
                errors="coerce",
            )

            manifest_rows.append(
                {
                    "matrix_type": matrix_type,
                    "variable_id": variable_id,
                    "antigen_target": antigen_target,
                    "feature": feature,
                    "antigen_class": metadata[
                        "antigen_class"
                    ],
                    "assay_family": metadata[
                        "assay_family"
                    ],
                    "outcome_family": metadata[
                        "outcome_family"
                    ],
                    "maximum_paired_floor_severity": (
                        metadata[
                            "maximum_paired_floor_severity"
                        ]
                    ),
                    "available_participants": int(
                        values.notna().sum()
                    ),
                    "missing_participants": int(
                        values.isna().sum()
                    ),
                    "mean": float(
                        values.mean()
                    ),
                    "standard_deviation": float(
                        values.std(
                            ddof=1
                        )
                    ),
                    "variance": float(
                        values.var(
                            ddof=1
                        )
                    ),
                    "evaluated_trajectories": metadata[
                        "evaluated_trajectories"
                    ],
                    "raw_fdr_significant_trajectories": (
                        metadata[
                            "raw_fdr_significant_trajectories"
                        ]
                    ),
                    "claim_ready_trajectories": metadata[
                        "claim_ready_trajectories"
                    ],
                    "qualified_trajectories": metadata[
                        "qualified_trajectories"
                    ],
                    "primary_claim_ready": metadata[
                        "primary_claim_ready"
                    ],
                    "recall_claim_ready_doses": metadata[
                        "recall_claim_ready_doses"
                    ],
                }
            )

    return pd.DataFrame(
        manifest_rows
    ).sort_values(
        [
            "matrix_type",
            "antigen_target",
            "feature",
        ]
    ).reset_index(
        drop=True
    )


def main() -> None:
    for path in [
        PAIRED_INPUT,
        EVIDENCE_INPUT,
        SYNTHESIS_DECISION,
    ]:
        if not path.exists():
            sys.exit(
                f"ERROR: Required input missing: {path}"
            )

    synthesis_decision = pd.read_csv(
        SYNTHESIS_DECISION,
        sep="\t",
    )

    observed_decision = str(
        synthesis_decision.loc[
            0,
            "decision",
        ]
    )

    if observed_decision != EXPECTED_SYNTHESIS_DECISION:
        sys.exit(
            "ERROR: Phase 2B2D2 decision is "
            f"{observed_decision}; expected "
            f"{EXPECTED_SYNTHESIS_DECISION}."
        )

    paired = pd.read_csv(
        PAIRED_INPUT,
        sep="\t",
        dtype={
            "participant_id": "string",
            "antigen_target": "string",
            "feature": "string",
        },
    )

    evidence = pd.read_csv(
        EVIDENCE_INPUT,
        sep="\t",
    )

    require_columns(
        paired,
        {
            "participant_id",
            "previous_4vHPV_doses",
            "antigen_target",
            "antigen_class",
            "feature",
            "assay_family",
            "outcome_family",
            "log2_change_authoritative",
            "prior_4vHPV_exposure_status",
            "trajectory",
            "v1_biological_context",
            "v2_biological_context",
            "paired_floor_severity",
        },
        "Paired-effect input",
    )

    require_columns(
        evidence,
        {
            "evidence_id",
            "analysis_context",
            "antigen_target",
            "feature",
            "previous_4vHPV_doses",
            "raw_fdr_significant",
            "claim_readiness",
        },
        "Integrated evidence registry",
    )

    paired[
        "previous_4vHPV_doses"
    ] = pd.to_numeric(
        paired[
            "previous_4vHPV_doses"
        ],
        errors="raise",
    ).astype(int)

    evidence[
        "previous_4vHPV_doses"
    ] = pd.to_numeric(
        evidence[
            "previous_4vHPV_doses"
        ],
        errors="coerce",
    ).astype("Int64")

    metadata = construct_metadata(
        paired
    )

    participant_order = metadata[
        "participant_id"
    ].astype(str).tolist()

    raw_matrix = construct_raw_matrix(
        paired,
        participant_order,
    )

    (
        calibrated_matrix,
        calibrated_long,
    ) = construct_bpv_calibrated_matrix(
        paired,
        participant_order,
    )

    feature_manifest = build_feature_manifest(
        paired,
        evidence,
        raw_matrix,
        calibrated_matrix,
    )

    context_masks = {
        "all_participants": (
            metadata[
                "participant_id"
            ].notna()
        ),
        "primary_dose0": (
            metadata[
                "previous_4vHPV_doses"
            ]
            == 0
        ),
        "recall_all_doses": (
            metadata[
                "previous_4vHPV_doses"
            ]
            > 0
        ),
        "recall_dose1": (
            metadata[
                "previous_4vHPV_doses"
            ]
            == 1
        ),
        "recall_dose2": (
            metadata[
                "previous_4vHPV_doses"
            ]
            == 2
        ),
        "recall_dose3": (
            metadata[
                "previous_4vHPV_doses"
            ]
            == 3
        ),
    }

    readiness_rows: list[
        dict[str, object]
    ] = []

    for matrix_name, matrix in [
        (
            "raw_log2_change",
            raw_matrix,
        ),
        (
            "bpv_calibrated_log2_change",
            calibrated_matrix,
        ),
    ]:
        for scope, mask in context_masks.items():
            readiness_rows.append(
                matrix_audit_row(
                    matrix_name=matrix_name,
                    matrix=matrix,
                    metadata=metadata,
                    participant_mask=mask,
                    analysis_scope=scope,
                )
            )

    readiness = pd.DataFrame(
        readiness_rows
    )

    expected_dose_counts = {
        0: 20,
        1: 20,
        2: 21,
        3: 19,
    }

    observed_dose_counts = (
        metadata[
            "previous_4vHPV_doses"
        ]
        .value_counts()
        .to_dict()
    )

    failures: list[str] = []

    if len(paired) != 7360:
        failures.append(
            f"Expected 7360 paired rows, observed {len(paired)}."
        )

    if len(metadata) != 80:
        failures.append(
            f"Expected 80 participants, observed {len(metadata)}."
        )

    for dose, expected in expected_dose_counts.items():
        observed = int(
            observed_dose_counts.get(
                dose,
                0,
            )
        )

        if observed != expected:
            failures.append(
                f"Dose {dose}: expected {expected} participants, "
                f"observed {observed}."
            )

    raw_shape = (
        len(raw_matrix),
        len(raw_matrix.columns) - 1,
    )

    calibrated_shape = (
        len(calibrated_matrix),
        len(calibrated_matrix.columns) - 1,
    )

    if raw_shape != (80, 92):
        failures.append(
            "Raw matrix shape is "
            f"{raw_shape}; expected (80, 92)."
        )

    if calibrated_shape != (80, 77):
        failures.append(
            "BPV-calibrated matrix shape is "
            f"{calibrated_shape}; expected (80, 77)."
        )

    if raw_matrix.drop(
        columns=[
            "participant_id",
        ]
    ).isna().any().any():
        failures.append(
            "Raw matrix contains missing values."
        )

    if calibrated_matrix.drop(
        columns=[
            "participant_id",
        ]
    ).isna().any().any():
        failures.append(
            "BPV-calibrated matrix contains missing values."
        )

    if len(calibrated_long) != 6160:
        failures.append(
            "Expected 6160 calibrated long records, "
            f"observed {len(calibrated_long)}."
        )

    if len(feature_manifest) != 169:
        failures.append(
            "Expected 169 feature-manifest rows, "
            f"observed {len(feature_manifest)}."
        )

    if len(readiness) != 12:
        failures.append(
            "Expected 12 readiness-audit rows, "
            f"observed {len(readiness)}."
        )

    decision_value = (
        "READY_FOR_PHASE2C2_BREADTH_AND_PHASE2C3_FUNCTIONAL_COUPLING"
        if not failures
        else "PHASE2C1_REPAIR_REQUIRED"
    )

    metadata_output = (
        PROCESSED
        / "phase2C1_fiji_participant_metadata.tsv"
    )

    raw_output = (
        PROCESSED
        / "phase2C1_fiji_raw_log2_change_matrix.tsv"
    )

    calibrated_output = (
        PROCESSED
        / "phase2C1_fiji_bpv_calibrated_log2_change_matrix.tsv"
    )

    calibrated_long_output = (
        PROCESSED
        / "phase2C1_fiji_bpv_calibrated_effects_long.tsv"
    )

    manifest_output = (
        TABLES
        / "phase2C1_fiji_multivariate_feature_manifest.tsv"
    )

    readiness_output = (
        TABLES
        / "phase2C1_fiji_multivariate_readiness_audit.tsv"
    )

    decision_output = (
        TABLES
        / "phase2C1_fiji_multivariate_readiness_decision.tsv"
    )

    write_tsv(
        metadata,
        metadata_output,
    )

    write_tsv(
        raw_matrix,
        raw_output,
    )

    write_tsv(
        calibrated_matrix,
        calibrated_output,
    )

    write_tsv(
        calibrated_long,
        calibrated_long_output,
    )

    write_tsv(
        feature_manifest,
        manifest_output,
    )

    write_tsv(
        readiness,
        readiness_output,
    )

    decision = pd.DataFrame(
        [
            {
                "decision": decision_value,
                "paired_effect_rows": len(paired),
                "participants": len(metadata),
                "raw_matrix_rows": raw_shape[0],
                "raw_matrix_variables": raw_shape[1],
                "bpv_calibrated_matrix_rows": (
                    calibrated_shape[0]
                ),
                "bpv_calibrated_matrix_variables": (
                    calibrated_shape[1]
                ),
                "bpv_calibrated_long_rows": len(
                    calibrated_long
                ),
                "feature_manifest_rows": len(
                    feature_manifest
                ),
                "readiness_audit_rows": len(
                    readiness
                ),
                "raw_missing_values": int(
                    raw_matrix.drop(
                        columns=[
                            "participant_id",
                        ]
                    ).isna().sum().sum()
                ),
                "bpv_calibrated_missing_values": int(
                    calibrated_matrix.drop(
                        columns=[
                            "participant_id",
                        ]
                    ).isna().sum().sum()
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
        / "phase2C1_fiji_multivariate_readiness_report.md"
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as report:
        report.write(
            "# Phase 2C1 Fiji multivariate readiness\n\n"
        )

        report.write("## Decision\n\n")
        report.write(f"**{decision_value}**\n\n")

        report.write(
            f"- Participants: {len(metadata)}\n"
        )
        report.write(
            f"- Raw participant matrix: "
            f"{raw_shape[0]} × {raw_shape[1]}\n"
        )
        report.write(
            f"- BPV-calibrated participant matrix: "
            f"{calibrated_shape[0]} × "
            f"{calibrated_shape[1]}\n"
        )
        report.write(
            f"- BPV-calibrated long records: "
            f"{len(calibrated_long)}\n"
        )
        report.write(
            f"- Feature-manifest rows: "
            f"{len(feature_manifest)}\n"
        )
        report.write(
            f"- Readiness-audit strata: "
            f"{len(readiness)}\n\n"
        )

        report.write(
            "The raw matrix contains participant-level paired log2 "
            "changes for all 92 antigen-feature combinations. The "
            "BPV-calibrated matrix contains the 77 HPV variables that "
            "share an assay feature with the heterologous BPV control. "
            "Primary and recall groups should be analyzed separately "
            "where biological context is central. Dose-specific recall "
            "analyses have limited sample sizes and should emphasize "
            "descriptive structure, stability and effect concordance "
            "rather than high-dimensional prediction.\n"
        )

    print("===== PHASE 2C1 COMPLETE =====")
    print(f"Decision: {decision_value}")
    print(
        f"Participants: {len(metadata)}"
    )
    print(
        "Raw matrix: "
        f"{raw_shape[0]} x {raw_shape[1]}"
    )
    print(
        "BPV-calibrated matrix: "
        f"{calibrated_shape[0]} x "
        f"{calibrated_shape[1]}"
    )
    print(
        "Calibrated long records: "
        f"{len(calibrated_long)}"
    )
    print(
        "Feature manifest rows: "
        f"{len(feature_manifest)}"
    )
    print(
        "Readiness audit rows: "
        f"{len(readiness)}"
    )
    print(f"Report: {report_path}")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
