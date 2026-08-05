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

REGISTRY_INPUT = (
    TABLES
    / "phase2B2D1_fiji_integrated_evidence_registry.tsv"
)

D1_DECISION = (
    TABLES
    / "phase2B2D1_fiji_integrated_evidence_decision.tsv"
)

EXPECTED_D1_DECISION = (
    "READY_FOR_PHASE2B2D2_BIOLOGICAL_SYNTHESIS"
)

CLAIM_READY_GRADES = {
    "A1_robust_bpv_calibrated",
    "A2_robust_without_bpv_calibration",
}

QUALIFIED_GRADES = {
    "B1_method_sensitive",
    "B2_attenuated_after_bpv_calibration",
    "B3_direction_changed_after_bpv_calibration",
    "B4_not_supported_after_floor_sensitivity",
    "B5_mixed_model_direction_not_confirmed",
}

EXPLORATORY_GRADES = {
    "C1_emerges_after_bpv_calibration",
    "C2_direction_changed_without_raw_fdr",
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

CONTEXT_LABELS = {
    "primary_2vHPV_induction": (
        "Primary 2vHPV induction"
    ),
    "heterologous_2vHPV_recall": (
        "Heterologous 2vHPV recall"
    ),
    "heterologous_bpv_control": (
        "Heterologous BPV control"
    ),
}

ANTIGEN_GROUP_LABELS = {
    "vaccine_type": (
        "Vaccine-type HPV16/18"
    ),
    "cross_reactive_type": (
        "Cross-reactive HPV31/33/45/52/58"
    ),
    "heterologous_control": (
        "Heterologous BPV control"
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


def numeric_series(
    values: pd.Series,
) -> pd.Series:
    return pd.to_numeric(
        values,
        errors="coerce",
    )


def safe_median_absolute(
    values: pd.Series,
) -> float:
    numeric = numeric_series(
        values
    ).dropna()

    if numeric.empty:
        return np.nan

    return float(
        np.median(
            np.abs(
                numeric.to_numpy(
                    dtype=float
                )
            )
        )
    )


def safe_maximum_absolute(
    values: pd.Series,
) -> float:
    numeric = numeric_series(
        values
    ).dropna()

    if numeric.empty:
        return np.nan

    return float(
        np.max(
            np.abs(
                numeric.to_numpy(
                    dtype=float
                )
            )
        )
    )


def comma_join(
    values: pd.Series,
) -> str:
    cleaned = sorted(
        {
            str(value)
            for value in values
            if pd.notna(value)
            and str(value)
        }
    )

    return ", ".join(
        cleaned
    )


def mechanistic_axis(
    feature: str,
    outcome_family: str,
) -> str:
    feature = str(feature)
    outcome_family = str(
        outcome_family
    )

    if feature == "nAb":
        return "Neutralizing-antibody function"

    if feature == "ADCP":
        return "Phagocyte-directed Fc effector function"

    if feature in {
        "FcgR2A",
        "FcgR2B",
        "FcgR3A",
    }:
        return "Fc-receptor communication"

    if feature in {
        "IgG1",
        "IgG2",
        "IgG3",
        "IgG4",
    }:
        return "IgG subclass architecture"

    if outcome_family == "Neutralization":
        return "Neutralizing-antibody function"

    return "Binding-antibody abundance"


def evidence_tier(
    grade: str,
) -> str:
    if grade in CLAIM_READY_GRADES:
        return "claim_ready"

    if grade in QUALIFIED_GRADES:
        return "qualified"

    if grade in EXPLORATORY_GRADES:
        return "exploratory"

    if grade == "CTRL_heterologous_bpv_control":
        return "control_context"

    return "not_supported"


def effect_sentence(
    row: pd.Series,
) -> str:
    context = CONTEXT_LABELS.get(
        str(
            row["analysis_context"]
        ),
        str(
            row["analysis_context"]
        ),
    )

    antigen = str(
        row["antigen_target"]
    )

    feature = str(
        row["feature"]
    )

    effect = float(
        row["raw_effect_log2"]
    )

    ratio = float(
        row[
            "raw_geometric_mean_ratio"
        ]
    )

    q_value = float(
        row["raw_q_value"]
    )

    direction = (
        "increased"
        if effect > 0
        else "decreased"
    )

    floor_status = str(
        row["floor_status"]
    )

    bpv_status = str(
        row["bpv_status"]
    )

    return (
        f"{context}: {antigen} {feature} {direction} "
        f"by {effect:.2f} log2 units "
        f"({ratio:.2f}-fold; q={q_value:.3g}); "
        f"floor={floor_status}; "
        f"BPV calibration={bpv_status}."
    )


def prepare_registry(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    output = frame.copy()

    output[
        "previous_4vHPV_doses"
    ] = pd.to_numeric(
        output[
            "previous_4vHPV_doses"
        ],
        errors="coerce",
    ).astype(
        "Int64"
    )

    output[
        "raw_effect_log2"
    ] = numeric_series(
        output[
            "raw_effect_log2"
        ]
    )

    output[
        "raw_geometric_mean_ratio"
    ] = numeric_series(
        output[
            "raw_geometric_mean_ratio"
        ]
    )

    output[
        "raw_q_value"
    ] = numeric_series(
        output[
            "raw_q_value"
        ]
    )

    output[
        "evidence_priority_score"
    ] = numeric_series(
        output[
            "evidence_priority_score"
        ]
    )

    output[
        "mechanistic_axis"
    ] = [
        mechanistic_axis(
            feature,
            outcome_family,
        )
        for feature, outcome_family in zip(
            output["feature"],
            output["outcome_family"],
        )
    ]

    output[
        "evidence_tier"
    ] = output[
        "evidence_grade"
    ].map(
        evidence_tier
    )

    output[
        "effect_direction"
    ] = np.select(
        [
            output[
                "raw_effect_log2"
            ]
            > 0,
            output[
                "raw_effect_log2"
            ]
            < 0,
        ],
        [
            "increased",
            "decreased",
        ],
        default="no_change",
    )

    return output


def build_context_summary(
    registry: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[
        dict[str, object]
    ] = []

    for context, group in registry.groupby(
        "analysis_context",
        observed=True,
        dropna=False,
    ):
        claim_ready = group[
            group[
                "evidence_tier"
            ]
            == "claim_ready"
        ]

        qualified = group[
            group[
                "evidence_tier"
            ]
            == "qualified"
        ]

        exploratory = group[
            group[
                "evidence_tier"
            ]
            == "exploratory"
        ]

        rows.append(
            {
                "analysis_context": context,
                "context_label": CONTEXT_LABELS.get(
                    str(context),
                    str(context),
                ),
                "total_effects": len(
                    group
                ),
                "raw_fdr_significant_effects": int(
                    (
                        group[
                            "raw_q_value"
                        ]
                        < 0.05
                    ).sum()
                ),
                "claim_ready_effects": len(
                    claim_ready
                ),
                "qualified_effects": len(
                    qualified
                ),
                "exploratory_effects": len(
                    exploratory
                ),
                "positive_claim_ready_effects": int(
                    (
                        claim_ready[
                            "raw_effect_log2"
                        ]
                        > 0
                    ).sum()
                ),
                "negative_claim_ready_effects": int(
                    (
                        claim_ready[
                            "raw_effect_log2"
                        ]
                        < 0
                    ).sum()
                ),
                "claim_ready_antigens": (
                    claim_ready[
                        "antigen_target"
                    ].nunique()
                ),
                "claim_ready_features": (
                    claim_ready[
                        "feature"
                    ].nunique()
                ),
                "claim_ready_mechanistic_axes": (
                    claim_ready[
                        "mechanistic_axis"
                    ].nunique()
                ),
                "median_absolute_claim_ready_log2_effect": (
                    safe_median_absolute(
                        claim_ready[
                            "raw_effect_log2"
                        ]
                    )
                ),
                "maximum_absolute_claim_ready_log2_effect": (
                    safe_maximum_absolute(
                        claim_ready[
                            "raw_effect_log2"
                        ]
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        "analysis_context"
    ).reset_index(
        drop=True
    )


def build_antigen_summary(
    registry: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[
        dict[str, object]
    ] = []

    group_columns = [
        "analysis_context",
        "antigen_group",
        "antigen_target",
    ]

    for keys, group in registry.groupby(
        group_columns,
        observed=True,
        dropna=False,
    ):
        (
            context,
            antigen_group,
            antigen_target,
        ) = keys

        claim_ready = group[
            group[
                "evidence_tier"
            ]
            == "claim_ready"
        ]

        qualified = group[
            group[
                "evidence_tier"
            ]
            == "qualified"
        ]

        top_feature = ""
        top_axis = ""
        top_score = np.nan
        top_effect = np.nan
        top_q_value = np.nan

        if not claim_ready.empty:
            top = claim_ready.sort_values(
                [
                    "evidence_priority_score",
                    "raw_q_value",
                ],
                ascending=[
                    False,
                    True,
                ],
            ).iloc[0]

            top_feature = str(
                top["feature"]
            )

            top_axis = str(
                top[
                    "mechanistic_axis"
                ]
            )

            top_score = float(
                top[
                    "evidence_priority_score"
                ]
            )

            top_effect = float(
                top[
                    "raw_effect_log2"
                ]
            )

            top_q_value = float(
                top[
                    "raw_q_value"
                ]
            )

        rows.append(
            {
                "analysis_context": context,
                "context_label": CONTEXT_LABELS.get(
                    str(context),
                    str(context),
                ),
                "antigen_group": antigen_group,
                "antigen_group_label": ANTIGEN_GROUP_LABELS.get(
                    str(
                        antigen_group
                    ),
                    str(
                        antigen_group
                    ),
                ),
                "antigen_target": antigen_target,
                "total_effects": len(
                    group
                ),
                "raw_fdr_significant_effects": int(
                    (
                        group[
                            "raw_q_value"
                        ]
                        < 0.05
                    ).sum()
                ),
                "claim_ready_effects": len(
                    claim_ready
                ),
                "qualified_effects": len(
                    qualified
                ),
                "claim_ready_features": comma_join(
                    claim_ready[
                        "feature"
                    ]
                ),
                "claim_ready_axes": comma_join(
                    claim_ready[
                        "mechanistic_axis"
                    ]
                ),
                "median_absolute_claim_ready_log2_effect": (
                    safe_median_absolute(
                        claim_ready[
                            "raw_effect_log2"
                        ]
                    )
                ),
                "maximum_absolute_claim_ready_log2_effect": (
                    safe_maximum_absolute(
                        claim_ready[
                            "raw_effect_log2"
                        ]
                    )
                ),
                "top_claim_ready_feature": top_feature,
                "top_claim_ready_axis": top_axis,
                "top_claim_ready_effect_log2": top_effect,
                "top_claim_ready_q_value": top_q_value,
                "top_claim_ready_priority_score": top_score,
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        [
            "analysis_context",
            "antigen_group",
            "antigen_target",
        ]
    ).reset_index(
        drop=True
    )


def build_axis_summary(
    registry: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[
        dict[str, object]
    ] = []

    group_columns = [
        "analysis_context",
        "antigen_group",
        "mechanistic_axis",
    ]

    for keys, group in registry.groupby(
        group_columns,
        observed=True,
        dropna=False,
    ):
        (
            context,
            antigen_group,
            axis,
        ) = keys

        claim_ready = group[
            group[
                "evidence_tier"
            ]
            == "claim_ready"
        ]

        qualified = group[
            group[
                "evidence_tier"
            ]
            == "qualified"
        ]

        rows.append(
            {
                "analysis_context": context,
                "context_label": CONTEXT_LABELS.get(
                    str(context),
                    str(context),
                ),
                "antigen_group": antigen_group,
                "antigen_group_label": ANTIGEN_GROUP_LABELS.get(
                    str(
                        antigen_group
                    ),
                    str(
                        antigen_group
                    ),
                ),
                "mechanistic_axis": axis,
                "total_effects": len(
                    group
                ),
                "raw_fdr_significant_effects": int(
                    (
                        group[
                            "raw_q_value"
                        ]
                        < 0.05
                    ).sum()
                ),
                "claim_ready_effects": len(
                    claim_ready
                ),
                "qualified_effects": len(
                    qualified
                ),
                "bpv_calibrated_supported_effects": int(
                    (
                        group[
                            "evidence_grade"
                        ]
                        == "A1_robust_bpv_calibrated"
                    ).sum()
                ),
                "robust_without_bpv_calibration": int(
                    (
                        group[
                            "evidence_grade"
                        ]
                        == "A2_robust_without_bpv_calibration"
                    ).sum()
                ),
                "attenuated_after_bpv_calibration": int(
                    (
                        group[
                            "evidence_grade"
                        ]
                        == "B2_attenuated_after_bpv_calibration"
                    ).sum()
                ),
                "direction_changed_after_bpv_calibration": int(
                    (
                        group[
                            "evidence_grade"
                        ]
                        == "B3_direction_changed_after_bpv_calibration"
                    ).sum()
                ),
                "not_supported_after_floor_sensitivity": int(
                    (
                        group[
                            "evidence_grade"
                        ]
                        == "B4_not_supported_after_floor_sensitivity"
                    ).sum()
                ),
                "claim_ready_antigens": comma_join(
                    claim_ready[
                        "antigen_target"
                    ]
                ),
                "claim_ready_features": comma_join(
                    claim_ready[
                        "feature"
                    ]
                ),
                "median_absolute_claim_ready_log2_effect": (
                    safe_median_absolute(
                        claim_ready[
                            "raw_effect_log2"
                        ]
                    )
                ),
                "maximum_absolute_claim_ready_log2_effect": (
                    safe_maximum_absolute(
                        claim_ready[
                            "raw_effect_log2"
                        ]
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        [
            "analysis_context",
            "antigen_group",
            "mechanistic_axis",
        ]
    ).reset_index(
        drop=True
    )


def build_headline_effects(
    registry: pd.DataFrame,
) -> pd.DataFrame:
    claim_ready = registry[
        registry[
            "evidence_tier"
        ]
        == "claim_ready"
    ].copy()

    claim_ready = claim_ready.sort_values(
        [
            "analysis_context",
            "antigen_group",
            "mechanistic_axis",
            "evidence_priority_score",
            "raw_q_value",
        ],
        ascending=[
            True,
            True,
            True,
            False,
            True,
        ],
    )

    headline = (
        claim_ready.groupby(
            [
                "analysis_context",
                "antigen_group",
                "mechanistic_axis",
            ],
            observed=True,
            dropna=False,
            group_keys=False,
        )
        .head(3)
        .copy()
    )

    headline[
        "biological_summary"
    ] = headline.apply(
        effect_sentence,
        axis=1,
    )

    selected_columns = [
        "analysis_context",
        "antigen_group",
        "antigen_target",
        "previous_4vHPV_doses",
        "feature",
        "outcome_family",
        "mechanistic_axis",
        "raw_effect_log2",
        "raw_geometric_mean_ratio",
        "raw_q_value",
        "maximum_floor_severity",
        "floor_status",
        "bpv_status",
        "method_status",
        "evidence_grade",
        "claim_readiness",
        "evidence_priority_score",
        "interpretation_guardrail",
        "biological_summary",
    ]

    return headline[
        selected_columns
    ].sort_values(
        [
            "analysis_context",
            "antigen_group",
            "mechanistic_axis",
            "evidence_priority_score",
        ],
        ascending=[
            True,
            True,
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )


def build_caution_registry(
    registry: pd.DataFrame,
) -> pd.DataFrame:
    caution = registry[
        registry[
            "evidence_grade"
        ].isin(
            QUALIFIED_GRADES
            | EXPLORATORY_GRADES
        )
    ].copy()

    caution[
        "biological_summary"
    ] = caution.apply(
        effect_sentence,
        axis=1,
    )

    selected_columns = [
        "analysis_context",
        "antigen_group",
        "antigen_target",
        "previous_4vHPV_doses",
        "feature",
        "outcome_family",
        "mechanistic_axis",
        "raw_effect_log2",
        "raw_geometric_mean_ratio",
        "raw_q_value",
        "floor_status",
        "bpv_status",
        "method_status",
        "evidence_grade",
        "claim_readiness",
        "interpretation_guardrail",
        "biological_summary",
    ]

    return caution[
        selected_columns
    ].sort_values(
        [
            "evidence_grade",
            "raw_q_value",
        ],
        ascending=[
            True,
            True,
        ],
        na_position="last",
    ).reset_index(
        drop=True
    )


def top_effect_lines(
    frame: pd.DataFrame,
    limit: int,
) -> list[str]:
    if frame.empty:
        return [
            "- No claim-ready effects were identified."
        ]

    ranked = frame.sort_values(
        [
            "evidence_priority_score",
            "raw_q_value",
        ],
        ascending=[
            False,
            True,
        ],
    ).head(
        limit
    )

    return [
        "- " + effect_sentence(row)
        for _, row in ranked.iterrows()
    ]


def write_report(
    registry: pd.DataFrame,
    context_summary: pd.DataFrame,
    antigen_summary: pd.DataFrame,
    axis_summary: pd.DataFrame,
    headline: pd.DataFrame,
    caution: pd.DataFrame,
    decision_value: str,
    report_path: Path,
) -> None:
    claim_ready = registry[
        registry[
            "evidence_tier"
        ]
        == "claim_ready"
    ]

    primary = claim_ready[
        claim_ready[
            "analysis_context"
        ]
        == "primary_2vHPV_induction"
    ]

    vaccine_recall = claim_ready[
        (
            claim_ready[
                "analysis_context"
            ]
            == "heterologous_2vHPV_recall"
        )
        & (
            claim_ready[
                "antigen_group"
            ]
            == "vaccine_type"
        )
    ]

    cross_reactive_recall = claim_ready[
        (
            claim_ready[
                "analysis_context"
            ]
            == "heterologous_2vHPV_recall"
        )
        & (
            claim_ready[
                "antigen_group"
            ]
            == "cross_reactive_type"
        )
    ]

    grade_counts = (
        registry[
            "evidence_grade"
        ]
        .value_counts()
        .to_dict()
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
            "# Phase 2B2D2 Fiji biological synthesis\n\n"
        )

        report.write(
            "## Decision\n\n"
        )

        report.write(
            f"**{decision_value}**\n\n"
        )

        report.write(
            "## Integrated analytical scope\n\n"
        )

        report.write(
            f"- Total antigen-feature-dose trajectories: "
            f"{len(registry)}\n"
        )

        report.write(
            f"- Claim-ready trajectories: "
            f"{len(claim_ready)}\n"
        )

        report.write(
            f"- BPV-calibrated robust trajectories: "
            f"{grade_counts.get('A1_robust_bpv_calibrated', 0)}\n"
        )

        report.write(
            f"- Robust ADCP/neutralization trajectories without "
            f"matched BPV calibration: "
            f"{grade_counts.get('A2_robust_without_bpv_calibration', 0)}\n"
        )

        report.write(
            f"- BPV-attenuated trajectories: "
            f"{grade_counts.get('B2_attenuated_after_bpv_calibration', 0)}\n"
        )

        report.write(
            f"- BPV direction-changing trajectories: "
            f"{grade_counts.get('B3_direction_changed_after_bpv_calibration', 0)}\n"
        )

        report.write(
            f"- Floor-sensitive trajectories lacking support: "
            f"{grade_counts.get('B4_not_supported_after_floor_sensitivity', 0)}\n\n"
        )

        report.write(
            "## Biological synthesis\n\n"
        )

        report.write(
            "### 1. Primary 2vHPV induction establishes a broad "
            "vaccine-type humoral state 2vHPV induction establishes a broad "
            "vaccine-type humoral state\n\n"
        )

        report.write(
            "The primary day-28 response is characterized by strong "
            "vaccine-type antibody abundance, subclass remodeling, "
            "Fc-receptor engagement, and functional activity. HPV16 "
            "shows the broadest and largest coordinated response, "
            "including IgG, IgG1, IgG3, IgA1, IgA2, FcγR2B and "
            "FcγR3A. HPV18 also shows robust IgG and IgG1 induction, "
            "with additional supported antibody features. These data "
            "are consistent with effective priming of antibody-producing "
            "and Fc-programmed immunity after 2vHPV vaccination.\n\n"
        )

        for line in top_effect_lines(
            primary,
            10,
        ):
            report.write(
                line + "\n"
            )

        report.write(
            "\n### 2. Heterologous recall preserves vaccine-type "
            "neutralization while expanding systems-serology breadth\n\n"
        )

        report.write(
            "Six years after previous 4vHPV vaccination, heterologous "
            "2vHPV exposure induces substantial HPV16 and HPV18 recall. "
            "Neutralization is among the strongest vaccine-type recall "
            "signals, although neutralization and ADCP lack matched BPV "
            "assays and are therefore graded separately from the "
            "BPV-calibrated binding and Fc-receptor features.\n\n"
        )

        for line in top_effect_lines(
            vaccine_recall,
            8,
        ):
            report.write(
                line + "\n"
            )

        report.write(
            "\n### 3. Recall reveals extensive cross-reactive "
            "antibody and Fc architecture\n\n"
        )

        report.write(
            "The clearest recall-specific systems pattern is the "
            "expansion of cross-reactive responses against HPV31, "
            "HPV33, HPV45, HPV52 and HPV58. Claim-ready responses "
            "include IgG abundance, IgG1 and IgG3 subclass architecture, "
            "and FcγR3A engagement. This supports a broad cross-reactive "
            "antibody-effector state after heterologous boosting rather "
            "than a response restricted to HPV16 and HPV18 alone.\n\n"
        )

        for line in top_effect_lines(
            cross_reactive_recall,
            12,
        ):
            report.write(
                line + "\n"
            )

        report.write(
            "\n### 4. Fc-programmed immunity is a major component "
            "of the recall landscape\n\n"
        )

        report.write(
            "The combined IgG-subclass and Fcγ-receptor results indicate "
            "that recall is not explained solely by increased total "
            "antibody abundance. Strong IgG1, IgG3 and FcγR3A signals "
            "across several cross-reactive HPV types indicate remodeling "
            "of antibody effector quality and potential communication "
            "with Fcγ-receptor-bearing innate immune cells. ADCP provides "
            "a direct functional correlate for phagocyte-directed antibody "
            "activity where it was measured.\n\n"
        )

        report.write(
            "### 5. BPV calibration materially refines HPV-specific "
            "interpretation\n\n"
        )

        report.write(
            "Matched BPV subtraction supports 132 robust HPV-associated "
            "trajectories, but attenuates 52 raw significant effects and "
            "changes the direction of 12 raw significant effects. "
            "Consequently, uncalibrated movement should not automatically "
            "be interpreted as HPV-specific. BPV-calibrated effects form "
            "the strongest basis for antigen-associated systems-serology "
            "claims.\n\n"
        )

        report.write(
            "### 6. Assay-floor sensitivity changes the evidential "
            "status of a subset of features\n\n"
        )

        report.write(
            "Moderate- and high-floor assays were evaluated using "
            "detection-transition and conditional-magnitude components. "
            "Sixteen raw significant trajectories were not supported "
            "after this sensitivity analysis and therefore require "
            "qualification rather than primary biological promotion.\n\n"
        )

        report.write(
            "### 7. HPV16–FcγR2A recall is directionally stable but "
            "nominally method-sensitive\n\n"
        )

        report.write(
            "The HPV16–FcγR2A random-intercept model reached a boundary "
            "solution with effectively zero participant-level variance. "
            "Mixed-model and participant-clustered fixed-effect estimates "
            "were numerically identical and agreed in direction for all "
            "eleven tested contrasts. Three recall contrasts differed in "
            "nominal significance because of their standard errors. These "
            "contrasts should therefore be described by their confirmed "
            "direction and magnitude, while avoiding promotion as uniquely "
            "FDR-robust mixed-model findings.\n\n"
        )

        report.write(
            "## Mechanistic interpretation boundary\n\n"
        )

        report.write(
            "The Fiji data provide downstream systems-serology evidence "
            "for antibody quantity, subclass architecture, Fc-receptor "
            "communication, phagocyte-directed function and neutralization. "
            "These patterns are biologically compatible with effective "
            "antigen uptake, antigen presentation, T follicular helper–B-cell "
            "coordination, memory B-cell recall and Fc-effector programming. "
            "They do not directly measure intracellular VLP trafficking, "
            "endolysosomal routing, HLA-II loading or APC signaling. Those "
            "mechanistic processes must be integrated through separate "
            "molecular or transcriptomic datasets rather than inferred as "
            "directly observed in this trial.\n\n"
        )

        report.write(
            "## Qualified and exploratory findings\n\n"
        )

        report.write(
            f"The caution registry contains {len(caution)} trajectories "
            "that are BPV-attenuated, direction-changing, unsupported after "
            "floor sensitivity, or dependent on calibration. These effects "
            "remain biologically informative but require explicit analytical "
            "qualification in figures and manuscript text.\n"
        )


def main() -> None:
    for path in [
        REGISTRY_INPUT,
        D1_DECISION,
    ]:
        if not path.exists():
            sys.exit(
                f"ERROR: Required input missing: {path}"
            )

    decision_frame = pd.read_csv(
        D1_DECISION,
        sep="\t",
    )

    observed_decision = str(
        decision_frame.loc[
            0,
            "decision",
        ]
    )

    if observed_decision != EXPECTED_D1_DECISION:
        sys.exit(
            "ERROR: Phase 2B2D1 decision is "
            f"{observed_decision}; expected "
            f"{EXPECTED_D1_DECISION}."
        )

    registry = pd.read_csv(
        REGISTRY_INPUT,
        sep="\t",
    )

    required_columns = {
        "evidence_id",
        "analysis_context",
        "antigen_group",
        "antigen_target",
        "previous_4vHPV_doses",
        "feature",
        "outcome_family",
        "raw_effect_log2",
        "raw_geometric_mean_ratio",
        "raw_q_value",
        "maximum_floor_severity",
        "floor_status",
        "bpv_status",
        "method_status",
        "evidence_grade",
        "claim_readiness",
        "interpretation_guardrail",
        "evidence_priority_score",
        "mixed_direction_agreement",
    }

    require_columns(
        registry,
        required_columns,
        "Integrated evidence registry",
    )

    registry = prepare_registry(
        registry
    )

    context_summary = build_context_summary(
        registry
    )

    antigen_summary = build_antigen_summary(
        registry
    )

    axis_summary = build_axis_summary(
        registry
    )

    headline = build_headline_effects(
        registry
    )

    caution = build_caution_registry(
        registry
    )

    failures: list[str] = []

    if len(registry) != 368:
        failures.append(
            f"Expected 368 registry rows, observed {len(registry)}."
        )

    expected_context_counts = {
        "primary_2vHPV_induction": 81,
        "heterologous_2vHPV_recall": 243,
        "heterologous_bpv_control": 44,
    }

    observed_context_counts = (
        registry[
            "analysis_context"
        ]
        .value_counts()
        .to_dict()
    )

    for context, expected in expected_context_counts.items():
        observed = int(
            observed_context_counts.get(
                context,
                0,
            )
        )

        if observed != expected:
            failures.append(
                f"{context}: expected {expected} rows, "
                f"observed {observed}."
            )

    claim_ready_count = int(
        (
            registry[
                "evidence_tier"
            ]
            == "claim_ready"
        ).sum()
    )

    qualified_count = int(
        (
            registry[
                "evidence_tier"
            ]
            == "qualified"
        ).sum()
    )

    a1_count = int(
        (
            registry[
                "evidence_grade"
            ]
            == "A1_robust_bpv_calibrated"
        ).sum()
    )

    a2_count = int(
        (
            registry[
                "evidence_grade"
            ]
            == "A2_robust_without_bpv_calibration"
        ).sum()
    )

    if claim_ready_count != 148:
        failures.append(
            f"Expected 148 claim-ready effects, "
            f"observed {claim_ready_count}."
        )

    if qualified_count != 80:
        failures.append(
            f"Expected 80 qualified effects, "
            f"observed {qualified_count}."
        )

    if a1_count != 132:
        failures.append(
            f"Expected 132 A1 effects, observed {a1_count}."
        )

    if a2_count != 16:
        failures.append(
            f"Expected 16 A2 effects, observed {a2_count}."
        )

    mixed_direction_failures = int(
        (
            registry[
                "mixed_direction_agreement"
            ].astype(str)
            != "yes"
        ).sum()
    )

    if mixed_direction_failures:
        failures.append(
            f"{mixed_direction_failures} mixed-model directions "
            "are discordant."
        )

    if headline.empty:
        failures.append(
            "Headline-effect table is empty."
        )

    if context_summary.empty:
        failures.append(
            "Context summary is empty."
        )

    if antigen_summary.empty:
        failures.append(
            "Antigen summary is empty."
        )

    if axis_summary.empty:
        failures.append(
            "Mechanistic-axis summary is empty."
        )

    decision_value = (
        "READY_FOR_PHASE2B2D_COMMIT_AND_PHASE2C_MULTIVARIATE_ANALYSIS"
        if not failures
        else "PHASE2B2D2_REPAIR_REQUIRED"
    )

    context_output = (
        TABLES
        / "phase2B2D2_fiji_context_summary.tsv"
    )

    antigen_output = (
        TABLES
        / "phase2B2D2_fiji_antigen_summary.tsv"
    )

    axis_output = (
        TABLES
        / "phase2B2D2_fiji_mechanistic_axis_summary.tsv"
    )

    headline_output = (
        TABLES
        / "phase2B2D2_fiji_headline_effects.tsv"
    )

    caution_output = (
        TABLES
        / "phase2B2D2_fiji_caution_registry.tsv"
    )

    decision_output = (
        TABLES
        / "phase2B2D2_fiji_biological_synthesis_decision.tsv"
    )

    write_tsv(
        context_summary,
        context_output,
    )

    write_tsv(
        antigen_summary,
        antigen_output,
    )

    write_tsv(
        axis_summary,
        axis_output,
    )

    write_tsv(
        headline,
        headline_output,
    )

    write_tsv(
        caution,
        caution_output,
    )

    decision = pd.DataFrame(
        [
            {
                "decision": decision_value,
                "integrated_effect_rows": len(
                    registry
                ),
                "primary_effect_rows": int(
                    observed_context_counts.get(
                        "primary_2vHPV_induction",
                        0,
                    )
                ),
                "recall_effect_rows": int(
                    observed_context_counts.get(
                        "heterologous_2vHPV_recall",
                        0,
                    )
                ),
                "bpv_control_rows": int(
                    observed_context_counts.get(
                        "heterologous_bpv_control",
                        0,
                    )
                ),
                "claim_ready_effects": claim_ready_count,
                "qualified_effects": qualified_count,
                "bpv_calibrated_robust_effects": a1_count,
                "robust_unshared_assay_effects": a2_count,
                "context_summary_rows": len(
                    context_summary
                ),
                "antigen_summary_rows": len(
                    antigen_summary
                ),
                "mechanistic_axis_summary_rows": len(
                    axis_summary
                ),
                "headline_effect_rows": len(
                    headline
                ),
                "caution_registry_rows": len(
                    caution
                ),
                "mixed_direction_disagreements": (
                    mixed_direction_failures
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
        / "phase2B2D2_fiji_biological_synthesis_report.md"
    )

    write_report(
        registry=registry,
        context_summary=context_summary,
        antigen_summary=antigen_summary,
        axis_summary=axis_summary,
        headline=headline,
        caution=caution,
        decision_value=decision_value,
        report_path=report_path,
    )

    print(
        "===== PHASE 2B2D2 COMPLETE ====="
    )

    print(
        f"Decision: {decision_value}"
    )

    print(
        f"Integrated effects: {len(registry)}"
    )

    print(
        f"Claim-ready effects: {claim_ready_count}"
    )

    print(
        f"Qualified effects: {qualified_count}"
    )

    print(
        f"Context summaries: {len(context_summary)}"
    )

    print(
        f"Antigen summaries: {len(antigen_summary)}"
    )

    print(
        f"Mechanistic-axis summaries: {len(axis_summary)}"
    )

    print(
        f"Headline effects: {len(headline)}"
    )

    print(
        f"Caution effects: {len(caution)}"
    )

    print(
        f"Report: {report_path}"
    )

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
