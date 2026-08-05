#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/mnt/d/HPV_Vaccine_Trafficome_Project")

TABLES = ROOT / "08_results" / "tables"

REPORTS = (
    ROOT
    / "02_dataset_audit"
    / "hpv_specific"
    / "fiji_nct02276521"
)

CALIBRATION_INPUT = (
    TABLES
    / "phase2C3A_fiji_raw_vs_bpv_coupling_registry.tsv"
)

CONTEXT_INPUT = (
    TABLES
    / "phase2C3B_fiji_primary_vs_recall_correlation_differences.tsv"
)

PARTIAL_INPUT = (
    TABLES
    / "phase2C3B_fiji_recall_partial_coupling_tests.tsv"
)

HETEROGENEITY_INPUT = (
    TABLES
    / "phase2C3B_fiji_recall_dose_correlation_heterogeneity.tsv"
)

C3B_DECISION_INPUT = (
    TABLES
    / "phase2C3B_fiji_functional_coupling_inference_decision.tsv"
)

EXPECTED_C3B_DECISION = (
    "READY_FOR_PHASE2C3C_FUNCTIONAL_COUPLING_SYNTHESIS"
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


def require_columns(
    frame: pd.DataFrame,
    required: set[str],
    label: str,
) -> None:
    missing = required - set(
        frame.columns
    )

    if missing:
        sys.exit(
            f"ERROR: {label} is missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )


def relationship_label(
    antigen: str,
    predictor: str,
    outcome: str,
) -> str:
    if predictor == "ADCP" and outcome == "nAb":
        return (
            f"{antigen} ADCP–neutralization coordination"
        )

    return (
        f"{antigen} {predictor}–{outcome} coupling"
    )


def build_context_emergent(
    context: pd.DataFrame,
) -> pd.DataFrame:
    supported = context[
        pd.to_numeric(
            context["bh_q_value"],
            errors="coerce",
        )
        < 0.05
    ].copy()

    supported[
        "relationship"
    ] = supported.apply(
        lambda row: relationship_label(
            str(
                row["antigen_target"]
            ),
            str(
                row["predictor_feature"]
            ),
            str(
                row["functional_outcome"]
            ),
        ),
        axis=1,
    )

    supported[
        "evidence_grade"
    ] = (
        "A_context_difference_supported"
    )

    supported[
        "biological_interpretation"
    ] = supported.apply(
        lambda row: (
            f"{row['relationship']} was stronger during "
            f"heterologous recall than primary induction "
            f"(primary rho={row['primary_spearman_rho']:.3f}; "
            f"recall rho={row['recall_spearman_rho']:.3f}; "
            f"BH q={row['bh_q_value']:.3g})."
        ),
        axis=1,
    )

    selected = [
        "analysis_layer",
        "predictor_representation",
        "antigen_target",
        "predictor_feature",
        "functional_outcome",
        "relationship",
        "primary_participants",
        "recall_participants",
        "primary_spearman_rho",
        "recall_spearman_rho",
        "rho_difference_recall_minus_primary",
        "p_value",
        "bh_q_value",
        "context_pattern",
        "pair_maximum_floor_severity",
        "floor_sensitive_pair",
        "evidence_grade",
        "biological_interpretation",
    ]

    return supported[
        selected
    ].sort_values(
        "bh_q_value"
    ).reset_index(
        drop=True
    )


def build_claim_ready_predictors(
    calibration: pd.DataFrame,
    partial: pd.DataFrame,
    context: pd.DataFrame,
) -> pd.DataFrame:
    stable = calibration[
        (
            calibration[
                "analysis_stratum"
            ]
            == "recall_all_doses"
        )
        & (
            calibration[
                "bpv_calibration_status"
            ]
            == "supported_raw_and_bpv_calibrated"
        )
    ].copy()

    partial_bpv = partial[
        (
            partial[
                "analysis_layer"
            ]
            == "predictor_function"
        )
        & (
            partial[
                "predictor_representation"
            ]
            == "bpv_calibrated_predictor"
        )
        & (
            pd.to_numeric(
                partial["bh_q_value"],
                errors="coerce",
            )
            < 0.05
        )
    ][
        [
            "antigen_target",
            "predictor_feature",
            "functional_outcome",
            "participants",
            "zero_order_spearman_rho",
            "partial_spearman_rho",
            "rho_change_after_dose_adjustment",
            "p_value",
            "bh_q_value",
            "pair_maximum_floor_severity",
            "floor_sensitive_pair",
        ]
    ].rename(
        columns={
            "participants": (
                "partial_participants"
            ),
            "p_value": (
                "partial_p_value"
            ),
            "bh_q_value": (
                "partial_q_value"
            ),
            "pair_maximum_floor_severity": (
                "partial_pair_floor_severity"
            ),
            "floor_sensitive_pair": (
                "partial_floor_sensitive_pair"
            ),
        }
    )

    claim_ready = stable.merge(
        partial_bpv,
        on=[
            "antigen_target",
            "predictor_feature",
            "functional_outcome",
        ],
        how="inner",
        validate="one_to_one",
    )

    raw_context = context[
        (
            context[
                "analysis_layer"
            ]
            == "predictor_function"
        )
        & (
            context[
                "predictor_representation"
            ]
            == "raw_predictor"
        )
    ][
        [
            "antigen_target",
            "predictor_feature",
            "functional_outcome",
            "primary_spearman_rho",
            "recall_spearman_rho",
            "rho_difference_recall_minus_primary",
            "bh_q_value",
        ]
    ].rename(
        columns={
            "bh_q_value": (
                "raw_context_difference_q_value"
            ),
        }
    )

    claim_ready = claim_ready.merge(
        raw_context,
        on=[
            "antigen_target",
            "predictor_feature",
            "functional_outcome",
        ],
        how="left",
        validate="one_to_one",
    )

    claim_ready[
        "raw_context_difference_supported"
    ] = (
        claim_ready[
            "raw_context_difference_q_value"
        ]
        < 0.05
    )

    claim_ready[
        "relationship"
    ] = claim_ready.apply(
        lambda row: relationship_label(
            str(
                row["antigen_target"]
            ),
            str(
                row["predictor_feature"]
            ),
            str(
                row["functional_outcome"]
            ),
        ),
        axis=1,
    )

    claim_ready[
        "evidence_grade"
    ] = (
        "A1_bpv_stable_and_dose_adjusted"
    )

    claim_ready[
        "claim_readiness"
    ] = "claim_ready"

    claim_ready[
        "biological_interpretation"
    ] = claim_ready.apply(
        lambda row: (
            f"{row['relationship']} remained supported after "
            f"matched BPV calibration and adjustment for previous-dose "
            f"group (BPV-calibrated zero-order rho="
            f"{row['bpv_calibrated_spearman_rho']:.3f}; "
            f"partial rho={row['partial_spearman_rho']:.3f}; "
            f"partial BH q={row['partial_q_value']:.3g})."
        ),
        axis=1,
    )

    selected = [
        "antigen_target",
        "predictor_feature",
        "functional_outcome",
        "relationship",
        "raw_spearman_rho",
        "bpv_calibrated_spearman_rho",
        "raw_q_value",
        "bpv_calibrated_q_value",
        "partial_participants",
        "partial_spearman_rho",
        "rho_change_after_dose_adjustment",
        "partial_q_value",
        "bpv_calibrated_pair_floor_severity",
        "partial_floor_sensitive_pair",
        "raw_context_difference_supported",
        "raw_context_difference_q_value",
        "evidence_grade",
        "claim_readiness",
        "biological_interpretation",
    ]

    return claim_ready[
        selected
    ].sort_values(
        [
            "partial_q_value",
            "bpv_calibrated_q_value",
        ]
    ).reset_index(
        drop=True
    )


def build_claim_ready_functions(
    partial: pd.DataFrame,
    context: pd.DataFrame,
) -> pd.DataFrame:
    functions = partial[
        (
            partial[
                "analysis_layer"
            ]
            == "function_function"
        )
        & (
            pd.to_numeric(
                partial["bh_q_value"],
                errors="coerce",
            )
            < 0.05
        )
    ].copy()

    function_context = context[
        context[
            "analysis_layer"
        ]
        == "function_function"
    ][
        [
            "antigen_target",
            "primary_spearman_rho",
            "recall_spearman_rho",
            "rho_difference_recall_minus_primary",
            "bh_q_value",
            "context_pattern",
        ]
    ].rename(
        columns={
            "bh_q_value": (
                "context_difference_q_value"
            ),
        }
    )

    functions = functions.merge(
        function_context,
        on="antigen_target",
        how="left",
        validate="one_to_one",
    )

    functions[
        "context_difference_supported"
    ] = (
        functions[
            "context_difference_q_value"
        ]
        < 0.05
    )

    functions[
        "relationship"
    ] = functions.apply(
        lambda row: relationship_label(
            str(
                row["antigen_target"]
            ),
            "ADCP",
            "nAb",
        ),
        axis=1,
    )

    functions[
        "evidence_grade"
    ] = (
        "A2_function_function_dose_adjusted"
    )

    functions[
        "claim_readiness"
    ] = (
        "claim_ready_with_floor_qualification"
    )

    functions[
        "biological_interpretation"
    ] = functions.apply(
        lambda row: (
            f"{row['relationship']} remained associated after adjustment "
            f"for previous-dose group (zero-order rho="
            f"{row['zero_order_spearman_rho']:.3f}; partial rho="
            f"{row['partial_spearman_rho']:.3f}; BH q="
            f"{row['bh_q_value']:.3g})."
        ),
        axis=1,
    )

    selected = [
        "antigen_target",
        "predictor_feature",
        "functional_outcome",
        "relationship",
        "participants",
        "zero_order_spearman_rho",
        "partial_spearman_rho",
        "rho_change_after_dose_adjustment",
        "p_value",
        "bh_q_value",
        "primary_spearman_rho",
        "recall_spearman_rho",
        "rho_difference_recall_minus_primary",
        "context_difference_q_value",
        "context_difference_supported",
        "pair_maximum_floor_severity",
        "floor_sensitive_pair",
        "evidence_grade",
        "claim_readiness",
        "biological_interpretation",
    ]

    return functions[
        selected
    ].sort_values(
        "bh_q_value"
    ).reset_index(
        drop=True
    )


def build_qualified(
    calibration: pd.DataFrame,
    partial: pd.DataFrame,
    heterogeneity: pd.DataFrame,
    claim_predictors: pd.DataFrame,
) -> pd.DataFrame:
    raw_partial = partial[
        (
            partial[
                "analysis_layer"
            ]
            == "predictor_function"
        )
        & (
            partial[
                "predictor_representation"
            ]
            == "raw_predictor"
        )
        & (
            pd.to_numeric(
                partial["bh_q_value"],
                errors="coerce",
            )
            < 0.05
        )
    ].copy()

    claim_keys = claim_predictors[
        [
            "antigen_target",
            "predictor_feature",
            "functional_outcome",
        ]
    ].drop_duplicates()

    raw_partial = raw_partial.merge(
        claim_keys.assign(
            bpv_stable_claim_ready=True
        ),
        on=[
            "antigen_target",
            "predictor_feature",
            "functional_outcome",
        ],
        how="left",
        validate="one_to_one",
    )

    raw_only = raw_partial[
        raw_partial[
            "bpv_stable_claim_ready"
        ].isna()
    ].copy()

    raw_only[
        "qualification_type"
    ] = (
        "raw_partial_supported_but_not_bpv_stable"
    )

    raw_only[
        "evidence_grade"
    ] = (
        "B1_calibration_sensitive_partial_association"
    )

    raw_only[
        "interpretation_guardrail"
    ] = (
        "The recall association remained significant after adjustment "
        "for previous-dose group in the raw HPV representation, but did "
        "not meet the combined BPV-stability criterion."
    )

    stable = calibration[
        (
            calibration[
                "analysis_stratum"
            ]
            == "recall_all_doses"
        )
        & (
            calibration[
                "bpv_calibration_status"
            ]
            == "supported_raw_and_bpv_calibrated"
        )
    ].copy()

    partial_bpv_keys = partial[
        (
            partial[
                "analysis_layer"
            ]
            == "predictor_function"
        )
        & (
            partial[
                "predictor_representation"
            ]
            == "bpv_calibrated_predictor"
        )
        & (
            pd.to_numeric(
                partial["bh_q_value"],
                errors="coerce",
            )
            < 0.05
        )
    ][
        [
            "antigen_target",
            "predictor_feature",
            "functional_outcome",
        ]
    ].drop_duplicates()

    stable = stable.merge(
        partial_bpv_keys.assign(
            partial_supported=True
        ),
        on=[
            "antigen_target",
            "predictor_feature",
            "functional_outcome",
        ],
        how="left",
        validate="one_to_one",
    )

    stable_not_partial = stable[
        stable[
            "partial_supported"
        ].isna()
    ].copy()

    stable_not_partial[
        "qualification_type"
    ] = (
        "bpv_stable_zero_order_not_partial_supported"
    )

    stable_not_partial[
        "evidence_grade"
    ] = (
        "B2_bpv_stable_zero_order_only"
    )

    stable_not_partial[
        "interpretation_guardrail"
    ] = (
        "The zero-order relationship was supported in raw and "
        "BPV-calibrated analyses, but it did not remain FDR significant "
        "after adjustment for previous-dose group."
    )

    heterogeneous = heterogeneity[
        pd.to_numeric(
            heterogeneity["bh_q_value"],
            errors="coerce",
        )
        < 0.05
    ].copy()

    heterogeneous[
        "qualification_type"
    ] = (
        "recall_dose_correlation_heterogeneity"
    )

    heterogeneous[
        "evidence_grade"
    ] = (
        "C1_dose_heterogeneous_coupling"
    )

    heterogeneous[
        "interpretation_guardrail"
    ] = (
        "The coupling relationship differed among previous-dose groups "
        "and should not be summarized as one homogeneous recall effect."
    )

    common_columns = [
        "antigen_target",
        "predictor_feature",
        "functional_outcome",
        "qualification_type",
        "evidence_grade",
        "interpretation_guardrail",
    ]

    raw_only_output = raw_only.copy()

    raw_only_output[
        "effect_value"
    ] = raw_only_output[
        "partial_spearman_rho"
    ]

    raw_only_output[
        "q_value"
    ] = raw_only_output[
        "bh_q_value"
    ]

    stable_output = stable_not_partial.copy()

    stable_output[
        "effect_value"
    ] = stable_output[
        "bpv_calibrated_spearman_rho"
    ]

    stable_output[
        "q_value"
    ] = stable_output[
        "bpv_calibrated_q_value"
    ]

    heterogeneous_output = heterogeneous.copy()

    heterogeneous_output[
        "effect_value"
    ] = heterogeneous[
        "i_squared_percent"
    ]

    heterogeneous_output[
        "q_value"
    ] = heterogeneous[
        "bh_q_value"
    ]

    selected = (
        common_columns
        + [
            "effect_value",
            "q_value",
        ]
    )

    qualified = pd.concat(
        [
            raw_only_output[
                selected
            ],
            stable_output[
                selected
            ],
            heterogeneous_output[
                selected
            ],
        ],
        ignore_index=True,
    )

    qualified[
        "relationship"
    ] = qualified.apply(
        lambda row: relationship_label(
            str(
                row["antigen_target"]
            ),
            str(
                row["predictor_feature"]
            ),
            str(
                row["functional_outcome"]
            ),
        ),
        axis=1,
    )

    return qualified[
        [
            "antigen_target",
            "predictor_feature",
            "functional_outcome",
            "relationship",
            "qualification_type",
            "effect_value",
            "q_value",
            "evidence_grade",
            "interpretation_guardrail",
        ]
    ].sort_values(
        [
            "evidence_grade",
            "q_value",
        ]
    ).reset_index(
        drop=True
    )


def build_summary(
    context_emergent: pd.DataFrame,
    claim_predictors: pd.DataFrame,
    claim_functions: pd.DataFrame,
    qualified: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "evidence_category": (
                "context_difference_supported"
            ),
            "findings": len(
                context_emergent
            ),
        },
        {
            "evidence_category": (
                "bpv_stable_and_dose_adjusted_predictor_function"
            ),
            "findings": len(
                claim_predictors
            ),
        },
        {
            "evidence_category": (
                "dose_adjusted_function_function"
            ),
            "findings": len(
                claim_functions
            ),
        },
    ]

    for grade, group in qualified.groupby(
        "evidence_grade",
        observed=True,
    ):
        rows.append(
            {
                "evidence_category": grade,
                "findings": len(
                    group
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    for path in [
        CALIBRATION_INPUT,
        CONTEXT_INPUT,
        PARTIAL_INPUT,
        HETEROGENEITY_INPUT,
        C3B_DECISION_INPUT,
    ]:
        if not path.exists():
            sys.exit(
                f"ERROR: Required input missing: {path}"
            )

    decision = pd.read_csv(
        C3B_DECISION_INPUT,
        sep="\t",
    )

    observed_decision = str(
        decision.loc[
            0,
            "decision",
        ]
    )

    if observed_decision != EXPECTED_C3B_DECISION:
        sys.exit(
            "ERROR: Phase 2C3B decision is "
            f"{observed_decision}; expected "
            f"{EXPECTED_C3B_DECISION}."
        )

    calibration = pd.read_csv(
        CALIBRATION_INPUT,
        sep="\t",
    )

    context = pd.read_csv(
        CONTEXT_INPUT,
        sep="\t",
    )

    partial = pd.read_csv(
        PARTIAL_INPUT,
        sep="\t",
    )

    heterogeneity = pd.read_csv(
        HETEROGENEITY_INPUT,
        sep="\t",
    )

    require_columns(
        calibration,
        {
            "analysis_stratum",
            "antigen_target",
            "predictor_feature",
            "functional_outcome",
            "raw_spearman_rho",
            "bpv_calibrated_spearman_rho",
            "raw_q_value",
            "bpv_calibrated_q_value",
            "bpv_calibration_status",
            "bpv_calibrated_pair_floor_severity",
        },
        "Calibration registry",
    )

    require_columns(
        context,
        {
            "analysis_layer",
            "predictor_representation",
            "antigen_target",
            "predictor_feature",
            "functional_outcome",
            "primary_spearman_rho",
            "recall_spearman_rho",
            "rho_difference_recall_minus_primary",
            "bh_q_value",
            "context_pattern",
            "pair_maximum_floor_severity",
            "floor_sensitive_pair",
        },
        "Context-difference table",
    )

    require_columns(
        partial,
        {
            "analysis_layer",
            "predictor_representation",
            "antigen_target",
            "predictor_feature",
            "functional_outcome",
            "participants",
            "zero_order_spearman_rho",
            "partial_spearman_rho",
            "rho_change_after_dose_adjustment",
            "p_value",
            "bh_q_value",
            "pair_maximum_floor_severity",
            "floor_sensitive_pair",
        },
        "Recall partial-coupling table",
    )

    require_columns(
        heterogeneity,
        {
            "analysis_layer",
            "predictor_representation",
            "antigen_target",
            "predictor_feature",
            "functional_outcome",
            "i_squared_percent",
            "bh_q_value",
        },
        "Dose-heterogeneity table",
    )

    context_emergent = build_context_emergent(
        context
    )

    claim_predictors = (
        build_claim_ready_predictors(
            calibration,
            partial,
            context,
        )
    )

    claim_functions = (
        build_claim_ready_functions(
            partial,
            context,
        )
    )

    qualified = build_qualified(
        calibration,
        partial,
        heterogeneity,
        claim_predictors,
    )

    summary = build_summary(
        context_emergent,
        claim_predictors,
        claim_functions,
        qualified,
    )

    failures: list[str] = []

    raw_partial_significant = int(
        (
            (
                partial[
                    "analysis_layer"
                ]
                == "predictor_function"
            )
            & (
                partial[
                    "predictor_representation"
                ]
                == "raw_predictor"
            )
            & (
                pd.to_numeric(
                    partial["bh_q_value"],
                    errors="coerce",
                )
                < 0.05
            )
        ).sum()
    )

    bpv_partial_significant = int(
        (
            (
                partial[
                    "analysis_layer"
                ]
                == "predictor_function"
            )
            & (
                partial[
                    "predictor_representation"
                ]
                == "bpv_calibrated_predictor"
            )
            & (
                pd.to_numeric(
                    partial["bh_q_value"],
                    errors="coerce",
                )
                < 0.05
            )
        ).sum()
    )

    function_partial_significant = int(
        (
            (
                partial[
                    "analysis_layer"
                ]
                == "function_function"
            )
            & (
                pd.to_numeric(
                    partial["bh_q_value"],
                    errors="coerce",
                )
                < 0.05
            )
        ).sum()
    )

    heterogeneity_significant = int(
        (
            pd.to_numeric(
                heterogeneity["bh_q_value"],
                errors="coerce",
            )
            < 0.05
        ).sum()
    )

    expected = {
        "context_emergent": (
            len(context_emergent),
            5,
        ),
        "claim_predictors": (
            len(claim_predictors),
            6,
        ),
        "claim_functions": (
            len(claim_functions),
            2,
        ),
        "raw_partial_significant": (
            raw_partial_significant,
            17,
        ),
        "bpv_partial_significant": (
            bpv_partial_significant,
            6,
        ),
        "function_partial_significant": (
            function_partial_significant,
            2,
        ),
        "heterogeneity_significant": (
            heterogeneity_significant,
            1,
        ),
        "qualified": (
            len(qualified),
            13,
        ),
    }

    for label, values in expected.items():
        observed, required = values

        if observed != required:
            failures.append(
                f"{label}: expected {required}, observed {observed}."
            )

    qualified_counts = (
        qualified[
            "evidence_grade"
        ]
        .value_counts()
        .to_dict()
    )

    expected_qualified = {
        "B1_calibration_sensitive_partial_association": 11,
        "B2_bpv_stable_zero_order_only": 1,
        "C1_dose_heterogeneous_coupling": 1,
    }

    for grade, required in expected_qualified.items():
        observed = int(
            qualified_counts.get(
                grade,
                0,
            )
        )

        if observed != required:
            failures.append(
                f"{grade}: expected {required}, observed {observed}."
            )

    decision_value = (
        "READY_FOR_PHASE2C3_COMMIT_AND_PHASE2C4_IMMUNE_STATE_STRUCTURE"
        if not failures
        else "PHASE2C3C_REPAIR_REQUIRED"
    )

    context_output = (
        TABLES
        / "phase2C3C_fiji_context_emergent_functional_coupling.tsv"
    )

    predictor_output = (
        TABLES
        / "phase2C3C_fiji_claim_ready_predictor_function_coupling.tsv"
    )

    function_output = (
        TABLES
        / "phase2C3C_fiji_claim_ready_function_function_coupling.tsv"
    )

    qualified_output = (
        TABLES
        / "phase2C3C_fiji_qualified_functional_coupling.tsv"
    )

    summary_output = (
        TABLES
        / "phase2C3C_fiji_functional_coupling_synthesis_summary.tsv"
    )

    decision_output = (
        TABLES
        / "phase2C3C_fiji_functional_coupling_synthesis_decision.tsv"
    )

    write_tsv(
        context_emergent,
        context_output,
    )

    write_tsv(
        claim_predictors,
        predictor_output,
    )

    write_tsv(
        claim_functions,
        function_output,
    )

    write_tsv(
        qualified,
        qualified_output,
    )

    write_tsv(
        summary,
        summary_output,
    )

    decision_frame = pd.DataFrame(
        [
            {
                "decision": decision_value,
                "context_difference_findings": len(
                    context_emergent
                ),
                "claim_ready_predictor_function_findings": len(
                    claim_predictors
                ),
                "claim_ready_function_function_findings": len(
                    claim_functions
                ),
                "raw_partial_predictor_findings": (
                    raw_partial_significant
                ),
                "bpv_calibrated_partial_predictor_findings": (
                    bpv_partial_significant
                ),
                "qualified_findings": len(
                    qualified
                ),
                "dose_heterogeneity_findings": (
                    heterogeneity_significant
                ),
                "validation_failures": "; ".join(
                    failures
                ),
            }
        ]
    )

    write_tsv(
        decision_frame,
        decision_output,
    )

    report_path = (
        REPORTS
        / "phase2C3C_fiji_functional_coupling_synthesis_report.md"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as report:
        report.write(
            "# Phase 2C3C Fiji functional-coupling synthesis\n\n"
        )

        report.write("## Decision\n\n")
        report.write(f"**{decision_value}**\n\n")

        report.write(
            "## Principal biological result\n\n"
        )

        report.write(
            "Heterologous recall generated a coordinated functional "
            "antibody state in which antibody abundance, IgG-subclass "
            "architecture, Fc-receptor engagement, phagocytic activity "
            "and neutralization became more tightly coupled than during "
            "primary induction.\n\n"
        )

        report.write(
            "The clearest state transition occurred for HPV16 "
            "ADCP–neutralization coordination, which was essentially "
            "absent during primary induction but strong during recall. "
            "This coupling remained nearly unchanged after adjustment "
            "for previous-dose group. HPV18 ADCP–neutralization coupling "
            "also remained significant after dose adjustment, although "
            "its primary-versus-recall correlation difference did not "
            "reach FDR significance.\n\n"
        )

        report.write(
            "## BPV-stable predictor–function relationships\n\n"
        )

        for sentence in claim_predictors[
            "biological_interpretation"
        ]:
            report.write(
                f"- {sentence}\n"
            )

        report.write(
            "\n## Function–function coordination\n\n"
        )

        for sentence in claim_functions[
            "biological_interpretation"
        ]:
            report.write(
                f"- {sentence}\n"
            )

        report.write(
            "\n## Primary-versus-recall restructuring\n\n"
        )

        report.write(
            "Five correlation differences survived FDR correction. "
            "All involved HPV16 neutralization, including IgG, IgG1, "
            "IgG3 and FcγR2B relationships and the direct "
            "ADCP–neutralization relationship. The predictor differences "
            "were evident in the raw HPV representation; BPV calibration "
            "showed that only a subset represents stable HPV-associated "
            "predictor–function coupling.\n\n"
        )

        report.write(
            "## Previous-dose-group effects\n\n"
        )

        report.write(
            "Twenty-five recall relationships remained significant after "
            "adjustment for previous-dose group, while only one relationship "
            "showed significant correlation heterogeneity among the one-, "
            "two- and three-dose groups. The heterogeneous relationship was "
            "HPV16 FcγR2A–neutralization coupling and was also moderately "
            "floor-sensitive. It should therefore remain a qualified "
            "schedule-specific observation rather than a general recall "
            "relationship.\n\n"
        )

        report.write(
            "## Assay interpretation\n\n"
        )

        report.write(
            "The strongest BPV-stable ADCP relationships—HPV16 IgG3–ADCP "
            "and HPV18 IgG/IgG1–ADCP—were not materially affected by assay "
            "flooring. Neutralization relationships require explicit "
            "qualification because HPV16 neutralization was moderately "
            "floor-sensitive and HPV18 neutralization was highly "
            "floor-sensitive.\n\n"
        )

        report.write(
            "## Mechanistic boundary\n\n"
        )

        report.write(
            "These data demonstrate downstream coordination among antibody "
            "quantity, subclass architecture, Fc-receptor communication, "
            "phagocytic activity and neutralization. They are compatible "
            "with memory B-cell maturation and coordinated Fc/neutralizing "
            "effector programming, but they do not directly measure VLP "
            "uptake, intracellular trafficking, antigen processing, HLA-II "
            "loading or Tfh–B-cell interactions.\n"
        )

    print(
        "===== PHASE 2C3C COMPLETE ====="
    )

    print(
        f"Decision: {decision_value}"
    )

    print(
        "Context-difference findings: "
        f"{len(context_emergent)}"
    )

    print(
        "Claim-ready predictor-function findings: "
        f"{len(claim_predictors)}"
    )

    print(
        "Claim-ready function-function findings: "
        f"{len(claim_functions)}"
    )

    print(
        "Qualified findings: "
        f"{len(qualified)}"
    )

    print(
        "Dose-heterogeneity findings: "
        f"{heterogeneity_significant}"
    )

    print(
        f"Report: {report_path}"
    )

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
