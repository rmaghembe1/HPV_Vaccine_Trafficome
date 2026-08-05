#!/usr/bin/env python3

from __future__ import annotations

import math
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
TABLES = ROOT / "08_results" / "tables"
REPORTS = (
    ROOT
    / "02_dataset_audit"
    / "hpv_specific"
    / "fiji_nct02276521"
)

RAW_WITHIN = (
    TABLES
    / "phase2B1_fiji_within_trajectory_tests.tsv"
)

REPAIRED_CONFIRMATION = (
    TABLES
    / "phase2B2A1_fiji_within_effect_confirmation_repaired.tsv"
)

FLOOR_REGISTRY = (
    TABLES
    / "phase2B2B_fiji_floor_robustness_registry.tsv"
)

BPV_REGISTRY = (
    TABLES
    / "phase2B2C_fiji_raw_vs_bpv_calibrated_registry.tsv"
)

BOUNDARY_REGISTRY = (
    TABLES
    / "phase2B2A2_hpv16_fcgr2a_boundary_sensitivity_contrasts.tsv"
)

DECISION_FILES = {
    "phase2B2B": (
        TABLES
        / "phase2B2B_fiji_floor_sensitivity_decision.tsv"
    ),
    "phase2B2C": (
        TABLES
        / "phase2B2C_fiji_bpv_calibrated_decision.tsv"
    ),
    "phase2B2A1": (
        TABLES
        / "phase2B2A1_fiji_hpv16_fcgr2a_refit_decision.tsv"
    ),
    "phase2B2A2": (
        TABLES
        / "phase2B2A2_hpv16_fcgr2a_boundary_sensitivity_decision.tsv"
    ),
}

EXPECTED_DECISIONS = {
    "phase2B2B": (
        "READY_FOR_PHASE2B2C_BPV_CALIBRATED_INFERENCE"
    ),
    "phase2B2C": (
        "READY_FOR_PHASE2B2_INTEGRATION_AND_BIOLOGICAL_SYNTHESIS"
    ),
    "phase2B2A1": (
        "READY_FOR_PHASE2B2D_INTEGRATION_AND_BIOLOGICAL_SYNTHESIS"
    ),
    "phase2B2A2": (
        "READY_FOR_PHASE2B2D_INTEGRATION_AND_BIOLOGICAL_SYNTHESIS"
    ),
}

GRADE_ORDER = {
    "A1_robust_bpv_calibrated": 1,
    "A2_robust_without_bpv_calibration": 2,
    "B1_method_sensitive": 3,
    "B2_attenuated_after_bpv_calibration": 4,
    "B3_direction_changed_after_bpv_calibration": 5,
    "B4_not_supported_after_floor_sensitivity": 6,
    "B5_mixed_model_direction_not_confirmed": 7,
    "C1_emerges_after_bpv_calibration": 8,
    "C2_direction_changed_without_raw_fdr": 9,
    "D1_not_fdr_significant": 10,
    "CTRL_heterologous_bpv_control": 11,
}

GRADE_LABELS = {
    "A1_robust_bpv_calibrated": (
        "Robust and supported after BPV calibration"
    ),
    "A2_robust_without_bpv_calibration": (
        "Robust but BPV calibration unavailable"
    ),
    "B1_method_sensitive": (
        "Direction and magnitude confirmed; nominal inference method-sensitive"
    ),
    "B2_attenuated_after_bpv_calibration": (
        "Raw effect attenuated after BPV calibration"
    ),
    "B3_direction_changed_after_bpv_calibration": (
        "Effect direction changed after BPV calibration"
    ),
    "B4_not_supported_after_floor_sensitivity": (
        "Raw effect not supported after floor sensitivity"
    ),
    "B5_mixed_model_direction_not_confirmed": (
        "Mixed-model direction not confirmed"
    ),
    "C1_emerges_after_bpv_calibration": (
        "Effect emerges only after BPV calibration"
    ),
    "C2_direction_changed_without_raw_fdr": (
        "Direction changed after calibration without raw FDR support"
    ),
    "D1_not_fdr_significant": (
        "Not FDR-significant in the primary paired analysis"
    ),
    "CTRL_heterologous_bpv_control": (
        "Heterologous BPV control trajectory"
    ),
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
    required: set[str],
    label: str,
) -> None:
    missing = required - set(frame.columns)

    if missing:
        sys.exit(
            f"ERROR: {label} is missing columns: "
            + ", ".join(sorted(missing))
        )


def normalize_dose(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    output = frame.copy()

    output["previous_4vHPV_doses"] = pd.to_numeric(
        output["previous_4vHPV_doses"],
        errors="coerce",
    ).astype("Int64")

    return output


def evidence_id(
    frame: pd.DataFrame,
) -> pd.Series:
    return (
        frame["antigen_target"].astype(str)
        + "|"
        + frame["feature"].astype(str)
        + "|dose"
        + frame["previous_4vHPV_doses"].astype(str)
    )


def read_decision(
    label: str,
    path: Path,
    expected: str,
) -> None:
    if not path.exists():
        sys.exit(
            f"ERROR: Missing {label} decision file: {path}"
        )

    frame = pd.read_csv(
        path,
        sep="\t",
    )

    if "decision" not in frame.columns or frame.empty:
        sys.exit(
            f"ERROR: Invalid {label} decision file."
        )

    observed = str(
        frame.loc[0, "decision"]
    )

    if observed != expected:
        sys.exit(
            f"ERROR: {label} decision is {observed}; "
            f"expected {expected}."
        )


def derive_floor_status(
    row: pd.Series,
) -> str:
    severity = str(
        row.get(
            "maximum_floor_severity",
            "",
        )
    )

    classification = str(
        row.get(
            "floor_robustness_classification",
            "",
        )
    )

    if severity in {
        "none",
        "low",
    }:
        return "not_required"

    if classification == "sensitivity_supported":
        return "supported"

    if classification == (
        "not_supported_after_floor_sensitivity"
    ):
        return "not_supported"

    if classification == (
        "continuous_result_not_fdr_significant"
    ):
        return "not_applicable_raw_not_fdr_significant"

    return "unresolved"


def derive_bpv_status(
    row: pd.Series,
) -> str:
    if str(row["antigen_target"]) == "BPV":
        return "heterologous_control"

    classification = row.get(
        "calibration_classification",
        np.nan,
    )

    if pd.isna(classification):
        return "not_available_for_unshared_assay"

    return str(classification)


def derive_method_status(
    row: pd.Series,
) -> str:
    agreement = row.get(
        "nominal_significance_agreement",
        np.nan,
    )

    if pd.isna(agreement):
        return "not_applicable"

    if str(agreement) == "yes":
        return "boundary_solution_confirmed"

    return "nominal_significance_method_sensitive"


def assign_grade(
    row: pd.Series,
) -> str:
    antigen = str(
        row["antigen_target"]
    )

    raw_significant = bool(
        row["raw_fdr_significant"]
    )

    mixed_direction = str(
        row["mixed_direction_agreement"]
    )

    floor_status = str(
        row["floor_status"]
    )

    bpv_status = str(
        row["bpv_status"]
    )

    method_status = str(
        row["method_status"]
    )

    if antigen == "BPV":
        return "CTRL_heterologous_bpv_control"

    if not raw_significant:
        if bpv_status == "emerges_after_bpv_calibration":
            return "C1_emerges_after_bpv_calibration"

        if bpv_status == (
            "direction_changed_after_bpv_calibration"
        ):
            return "C2_direction_changed_without_raw_fdr"

        return "D1_not_fdr_significant"

    if mixed_direction != "yes":
        return "B5_mixed_model_direction_not_confirmed"

    if floor_status == "not_supported":
        return "B4_not_supported_after_floor_sensitivity"

    if bpv_status == (
        "direction_changed_after_bpv_calibration"
    ):
        return "B3_direction_changed_after_bpv_calibration"

    if bpv_status == (
        "attenuated_after_bpv_calibration"
    ):
        return "B2_attenuated_after_bpv_calibration"

    if method_status == (
        "nominal_significance_method_sensitive"
    ):
        return "B1_method_sensitive"

    if bpv_status == "bpv_calibrated_supported":
        return "A1_robust_bpv_calibrated"

    if bpv_status == "not_available_for_unshared_assay":
        return "A2_robust_without_bpv_calibration"

    return "A2_robust_without_bpv_calibration"


def claim_readiness(
    grade: str,
) -> str:
    if grade.startswith("A"):
        return "claim_ready"

    if grade.startswith("B"):
        return "qualified_claim_only"

    if grade.startswith("C"):
        return "exploratory_only"

    if grade.startswith("CTRL"):
        return "control_context_only"

    return "not_claim_ready"


def interpretation_guardrail(
    grade: str,
) -> str:
    mapping = {
        "A1_robust_bpv_calibrated": (
            "May support a primary HPV-associated systems-serology claim."
        ),
        "A2_robust_without_bpv_calibration": (
            "May support a primary claim, but BPV calibration was unavailable "
            "for this assay feature."
        ),
        "B1_method_sensitive": (
            "Report direction and magnitude; describe nominal significance "
            "as method-sensitive."
        ),
        "B2_attenuated_after_bpv_calibration": (
            "Do not describe the raw response as independently HPV-specific."
        ),
        "B3_direction_changed_after_bpv_calibration": (
            "Do not interpret the raw direction without explicit BPV context."
        ),
        "B4_not_supported_after_floor_sensitivity": (
            "Do not promote without explicit assay-floor qualification."
        ),
        "B5_mixed_model_direction_not_confirmed": (
            "Do not promote until model-direction disagreement is resolved."
        ),
        "C1_emerges_after_bpv_calibration": (
            "Treat as exploratory and calibration-dependent."
        ),
        "C2_direction_changed_without_raw_fdr": (
            "Treat as exploratory heterologous-control-sensitive behavior."
        ),
        "D1_not_fdr_significant": (
            "Do not present as a statistically supported trajectory."
        ),
        "CTRL_heterologous_bpv_control": (
            "Use only to characterize heterologous-control movement."
        ),
    }

    return mapping[grade]


def priority_score(
    row: pd.Series,
) -> float:
    q_value = pd.to_numeric(
        pd.Series(
            [row["raw_q_value"]]
        ),
        errors="coerce",
    ).iloc[0]

    effect = pd.to_numeric(
        pd.Series(
            [row["raw_effect_log2"]]
        ),
        errors="coerce",
    ).iloc[0]

    q_component = 0.0

    if np.isfinite(q_value):
        q_component = min(
            -math.log10(
                max(
                    float(q_value),
                    1e-300,
                )
            ),
            50.0,
        )

    effect_component = 0.0

    if np.isfinite(effect):
        effect_component = min(
            abs(float(effect)),
            10.0,
        )

    grade_bonus = {
        "A1_robust_bpv_calibrated": 8.0,
        "A2_robust_without_bpv_calibration": 6.0,
        "B1_method_sensitive": 3.0,
        "B2_attenuated_after_bpv_calibration": 1.0,
        "B3_direction_changed_after_bpv_calibration": 0.0,
        "B4_not_supported_after_floor_sensitivity": 0.0,
        "B5_mixed_model_direction_not_confirmed": 0.0,
        "C1_emerges_after_bpv_calibration": 1.0,
        "C2_direction_changed_without_raw_fdr": 0.0,
        "D1_not_fdr_significant": 0.0,
        "CTRL_heterologous_bpv_control": 0.0,
    }[str(row["evidence_grade"])]

    return (
        q_component
        + effect_component
        + grade_bonus
    )


def main() -> None:
    for label, path in DECISION_FILES.items():
        read_decision(
            label,
            path,
            EXPECTED_DECISIONS[label],
        )

    required_files = [
        RAW_WITHIN,
        REPAIRED_CONFIRMATION,
        FLOOR_REGISTRY,
        BPV_REGISTRY,
        BOUNDARY_REGISTRY,
    ]

    for path in required_files:
        if not path.exists():
            sys.exit(
                f"ERROR: Required input missing: {path}"
            )

    raw = normalize_dose(
        pd.read_csv(
            RAW_WITHIN,
            sep="\t",
        )
    )

    confirmation = normalize_dose(
        pd.read_csv(
            REPAIRED_CONFIRMATION,
            sep="\t",
        )
    )

    floor = normalize_dose(
        pd.read_csv(
            FLOOR_REGISTRY,
            sep="\t",
        )
    )

    bpv = normalize_dose(
        pd.read_csv(
            BPV_REGISTRY,
            sep="\t",
        )
    )

    boundary = pd.read_csv(
        BOUNDARY_REGISTRY,
        sep="\t",
    )

    require_columns(
        raw,
        {
            "model_family",
            "previous_4vHPV_doses",
            "antigen_target",
            "feature",
            "mean_log2_change",
            "geometric_mean_ratio",
            "maximum_floor_severity",
            "p_value",
            "q_value",
        },
        "Phase 2B1 within table",
    )

    require_columns(
        confirmation,
        {
            "previous_4vHPV_doses",
            "antigen_target",
            "antigen_class",
            "feature",
            "assay_family",
            "outcome_family",
            "fit_method",
            "estimate_log2",
            "standard_error",
            "p_value",
            "q_value",
            "effect_difference",
            "effect_direction_agreement",
        },
        "Repaired mixed-model confirmation",
    )

    require_columns(
        floor,
        {
            "sensitivity_id",
            "previous_4vHPV_doses",
            "antigen_target",
            "feature",
            "robustness_classification",
            "detection_effect",
            "detection_q_value",
            "conditional_effect",
            "conditional_q_value",
        },
        "Floor robustness registry",
    )

    require_columns(
        bpv,
        {
            "previous_4vHPV_doses",
            "antigen_target",
            "feature",
            "raw_effect",
            "raw_q_value",
            "calibrated_effect",
            "calibrated_q_value",
            "calibration_classification",
        },
        "BPV calibration registry",
    )

    require_columns(
        boundary,
        {
            "contrast_label",
            "estimate_log2",
            "p_value",
            "cluster_robust_estimate_log2",
            "cluster_robust_standard_error",
            "cluster_robust_p_value",
            "direction_agreement",
            "nominal_significance_agreement",
        },
        "Boundary-sensitivity registry",
    )

    keys = [
        "antigen_target",
        "feature",
        "previous_4vHPV_doses",
    ]

    raw["evidence_id"] = evidence_id(
        raw
    )

    confirmation_selected = confirmation[
        [
            *keys,
            "antigen_class",
            "assay_family",
            "outcome_family",
            "fit_method",
            "estimate_log2",
            "standard_error",
            "p_value",
            "q_value",
            "effect_difference",
            "effect_direction_agreement",
        ]
    ].rename(
        columns={
            "fit_method": "mixed_fit_method",
            "estimate_log2": "mixed_effect_log2",
            "standard_error": "mixed_standard_error",
            "p_value": "mixed_p_value",
            "q_value": "mixed_q_value",
            "effect_difference": "mixed_raw_effect_difference",
            "effect_direction_agreement": (
                "mixed_direction_agreement"
            ),
        }
    )

    registry = raw.merge(
        confirmation_selected,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    # Coalesce metadata fields duplicated across the Phase 2B1 and
    # repaired mixed-model tables.
    for metadata_field in [
        "antigen_class",
        "assay_family",
        "outcome_family",
    ]:
        left_column = f"{metadata_field}_x"
        right_column = f"{metadata_field}_y"

        if (
            left_column in registry.columns
            and right_column in registry.columns
        ):
            registry[metadata_field] = registry[
                left_column
            ].combine_first(
                registry[right_column]
            )

            registry = registry.drop(
                columns=[
                    left_column,
                    right_column,
                ]
            )

        elif left_column in registry.columns:
            registry = registry.rename(
                columns={
                    left_column: metadata_field,
                }
            )

        elif right_column in registry.columns:
            registry = registry.rename(
                columns={
                    right_column: metadata_field,
                }
            )

    require_columns(
        registry,
        {
            "antigen_class",
            "assay_family",
            "outcome_family",
        },
        "Integrated registry metadata",
    )

    floor_within = floor[
        floor["sensitivity_id"]
        .astype(str)
        .str.startswith("within|")
    ][
        [
            *keys,
            "robustness_classification",
            "detection_effect",
            "detection_q_value",
            "conditional_effect",
            "conditional_q_value",
        ]
    ].rename(
        columns={
            "robustness_classification": (
                "floor_robustness_classification"
            ),
            "detection_effect": "floor_detection_effect",
            "detection_q_value": "floor_detection_q_value",
            "conditional_effect": "floor_conditional_effect",
            "conditional_q_value": "floor_conditional_q_value",
        }
    )

    registry = registry.merge(
        floor_within,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    bpv_selected = bpv[
        [
            *keys,
            "calibrated_effect",
            "calibrated_q_value",
            "calibration_classification",
        ]
    ].rename(
        columns={
            "calibrated_effect": "bpv_calibrated_effect_log2",
            "calibrated_q_value": "bpv_calibrated_q_value",
        }
    )

    registry = registry.merge(
        bpv_selected,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    dose_map = {
        "dose0_v2_minus_v1": 0,
        "dose1_v2_minus_v1": 1,
        "dose2_v2_minus_v1": 2,
        "dose3_v2_minus_v1": 3,
    }

    boundary_within = boundary[
        boundary["contrast_label"].isin(
            dose_map
        )
    ].copy()

    boundary_within[
        "antigen_target"
    ] = "HPV16"

    boundary_within[
        "feature"
    ] = "FcgR2A"

    boundary_within[
        "previous_4vHPV_doses"
    ] = boundary_within[
        "contrast_label"
    ].map(
        dose_map
    ).astype("Int64")

    boundary_selected = boundary_within[
        [
            *keys,
            "cluster_robust_estimate_log2",
            "cluster_robust_standard_error",
            "cluster_robust_p_value",
            "direction_agreement",
            "nominal_significance_agreement",
        ]
    ].rename(
        columns={
            "direction_agreement": (
                "boundary_direction_agreement"
            ),
        }
    )

    registry = registry.merge(
        boundary_selected,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    registry = registry.rename(
        columns={
            "mean_log2_change": "raw_effect_log2",
            "geometric_mean_ratio": "raw_geometric_mean_ratio",
            "p_value": "raw_p_value",
            "q_value": "raw_q_value",
        }
    )

    registry["raw_fdr_significant"] = (
        pd.to_numeric(
            registry["raw_q_value"],
            errors="coerce",
        )
        < 0.05
    )

    registry["effect_direction"] = np.select(
        [
            registry["raw_effect_log2"] > 0,
            registry["raw_effect_log2"] < 0,
        ],
        [
            "increased",
            "decreased",
        ],
        default="no_change",
    )

    registry["analysis_context"] = np.where(
        registry["antigen_target"] == "BPV",
        "heterologous_bpv_control",
        np.where(
            registry["previous_4vHPV_doses"] == 0,
            "primary_2vHPV_induction",
            "heterologous_2vHPV_recall",
        ),
    )

    registry["antigen_group"] = np.select(
        [
            registry["antigen_target"].isin(
                ["HPV16", "HPV18"]
            ),
            registry["antigen_target"].isin(
                [
                    "HPV31",
                    "HPV33",
                    "HPV45",
                    "HPV52",
                    "HPV58",
                ]
            ),
            registry["antigen_target"] == "BPV",
        ],
        [
            "vaccine_type",
            "cross_reactive_type",
            "heterologous_control",
        ],
        default="unresolved",
    )

    registry["floor_status"] = registry.apply(
        derive_floor_status,
        axis=1,
    )

    registry["bpv_status"] = registry.apply(
        derive_bpv_status,
        axis=1,
    )

    registry["method_status"] = registry.apply(
        derive_method_status,
        axis=1,
    )

    registry["evidence_grade"] = registry.apply(
        assign_grade,
        axis=1,
    )

    registry["evidence_grade_label"] = registry[
        "evidence_grade"
    ].map(
        GRADE_LABELS
    )

    registry["claim_readiness"] = registry[
        "evidence_grade"
    ].map(
        claim_readiness
    )

    registry["interpretation_guardrail"] = registry[
        "evidence_grade"
    ].map(
        interpretation_guardrail
    )

    registry["evidence_priority_score"] = registry.apply(
        priority_score,
        axis=1,
    )

    registry["grade_order"] = registry[
        "evidence_grade"
    ].map(
        GRADE_ORDER
    )

    registry = registry.sort_values(
        [
            "grade_order",
            "raw_q_value",
            "evidence_priority_score",
        ],
        ascending=[
            True,
            True,
            False,
        ],
        na_position="last",
    ).reset_index(
        drop=True
    )

    failures: list[str] = []

    if len(registry) != 368:
        failures.append(
            f"Expected 368 integrated rows, observed {len(registry)}."
        )

    if registry["evidence_id"].nunique() != 368:
        failures.append(
            "Integrated evidence identifiers are not unique."
        )

    if registry[
        "mixed_effect_log2"
    ].isna().any():
        failures.append(
            "Some rows lack repaired mixed-model confirmation."
        )

    if len(floor_within) != 148:
        failures.append(
            f"Expected 148 within-trajectory floor rows, "
            f"observed {len(floor_within)}."
        )

    if len(bpv_selected) != 308:
        failures.append(
            f"Expected 308 BPV-calibrated rows, "
            f"observed {len(bpv_selected)}."
        )

    if len(boundary_selected) != 4:
        failures.append(
            f"Expected four HPV16-FcgR2A boundary rows, "
            f"observed {len(boundary_selected)}."
        )

    direction_failures = int(
        (
            registry[
                "mixed_direction_agreement"
            ]
            .astype(str)
            != "yes"
        ).sum()
    )

    if direction_failures:
        failures.append(
            f"{direction_failures} mixed-model directions "
            "do not agree with Phase 2B1."
        )

    maximum_effect_difference = float(
        pd.to_numeric(
            registry[
                "mixed_raw_effect_difference"
            ],
            errors="coerce",
        )
        .abs()
        .max()
    )

    if maximum_effect_difference > 1e-8:
        failures.append(
            "Mixed and paired effect estimates differ by more "
            "than 1e-8 log2 units."
        )

    if registry[
        "evidence_grade"
    ].isna().any():
        failures.append(
            "Some rows lack an evidence grade."
        )

    grade_summary = (
        registry.groupby(
            [
                "evidence_grade",
                "evidence_grade_label",
                "claim_readiness",
            ],
            dropna=False,
        )
        .agg(
            effects=(
                "evidence_id",
                "size",
            ),
            antigens=(
                "antigen_target",
                "nunique",
            ),
            features=(
                "feature",
                "nunique",
            ),
            median_absolute_log2_effect=(
                "raw_effect_log2",
                lambda values: float(
                    np.nanmedian(
                        np.abs(
                            pd.to_numeric(
                                values,
                                errors="coerce",
                            )
                        )
                    )
                ),
            ),
            minimum_raw_q_value=(
                "raw_q_value",
                "min",
            ),
        )
        .reset_index()
    )

    grade_summary["grade_order"] = grade_summary[
        "evidence_grade"
    ].map(
        GRADE_ORDER
    )

    grade_summary = grade_summary.sort_values(
        "grade_order"
    ).drop(
        columns="grade_order"
    )

    claim_ready = registry[
        registry["claim_readiness"]
        == "claim_ready"
    ].copy()

    qualified = registry[
        registry["claim_readiness"]
        == "qualified_claim_only"
    ].copy()

    decision_value = (
        "READY_FOR_PHASE2B2D2_BIOLOGICAL_SYNTHESIS"
        if not failures
        else "PHASE2B2D1_REPAIR_REQUIRED"
    )

    decision = pd.DataFrame(
        [
            {
                "decision": decision_value,
                "integrated_effect_rows": len(registry),
                "unique_effect_ids": (
                    registry["evidence_id"].nunique()
                ),
                "claim_ready_effects": len(claim_ready),
                "qualified_effects": len(qualified),
                "floor_rows_integrated": len(floor_within),
                "bpv_calibrated_rows_integrated": len(bpv_selected),
                "boundary_rows_integrated": len(boundary_selected),
                "mixed_direction_disagreements": direction_failures,
                "maximum_mixed_raw_effect_difference": (
                    maximum_effect_difference
                ),
                "validation_failures": "; ".join(failures),
            }
        ]
    )

    registry_output = (
        TABLES
        / "phase2B2D1_fiji_integrated_evidence_registry.tsv"
    )

    grade_output = (
        TABLES
        / "phase2B2D1_fiji_evidence_grade_summary.tsv"
    )

    claim_output = (
        TABLES
        / "phase2B2D1_fiji_claim_ready_effects.tsv"
    )

    qualified_output = (
        TABLES
        / "phase2B2D1_fiji_qualified_effects.tsv"
    )

    decision_output = (
        TABLES
        / "phase2B2D1_fiji_integrated_evidence_decision.tsv"
    )

    write_tsv(
        registry,
        registry_output,
    )

    write_tsv(
        grade_summary,
        grade_output,
    )

    write_tsv(
        claim_ready,
        claim_output,
    )

    write_tsv(
        qualified,
        qualified_output,
    )

    write_tsv(
        decision,
        decision_output,
    )

    report_path = (
        REPORTS
        / "phase2B2D1_fiji_integrated_evidence_grading_report.md"
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    grade_counts = (
        registry[
            "evidence_grade"
        ]
        .value_counts()
        .to_dict()
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as report:
        report.write(
            "# Phase 2B2D1 Fiji integrated evidence grading\n\n"
        )

        report.write("## Decision\n\n")
        report.write(f"**{decision_value}**\n\n")

        report.write(
            f"- Integrated trajectory effects: {len(registry)}\n"
        )
        report.write(
            f"- Claim-ready effects: {len(claim_ready)}\n"
        )
        report.write(
            f"- Qualified effects: {len(qualified)}\n"
        )
        report.write(
            f"- Floor-sensitive rows incorporated: "
            f"{len(floor_within)}\n"
        )
        report.write(
            f"- BPV-calibrated rows incorporated: "
            f"{len(bpv_selected)}\n"
        )
        report.write(
            f"- HPV16–FcγR2A boundary rows incorporated: "
            f"{len(boundary_selected)}\n\n"
        )

        report.write("## Evidence-grade counts\n\n")

        for grade in sorted(
            grade_counts,
            key=lambda value: GRADE_ORDER[value],
        ):
            report.write(
                f"- `{grade}`: {grade_counts[grade]}\n"
            )

        report.write("\n## Interpretation framework\n\n")

        report.write(
            "The primary paired Phase 2B1 analysis remains the main "
            "trajectory inference. Repaired mixed models confirm effect "
            "direction and dose-aware structure. Moderate/high-floor "
            "features require their two-part sensitivity evidence. "
            "BPV calibration distinguishes HPV-associated movement from "
            "heterologous-control or assay-wide movement. HPV16–FcγR2A "
            "recall contrasts retain confirmed direction and magnitude, "
            "but nominal significance is explicitly method-sensitive.\n"
        )

    print("===== PHASE 2B2D1 COMPLETE =====")
    print(f"Decision: {decision_value}")
    print(f"Integrated effects: {len(registry)}")
    print(f"Claim-ready effects: {len(claim_ready)}")
    print(f"Qualified effects: {len(qualified)}")
    print(f"Floor rows integrated: {len(floor_within)}")
    print(f"BPV rows integrated: {len(bpv_selected)}")
    print(f"Boundary rows integrated: {len(boundary_selected)}")
    print(f"Report: {report_path}")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
