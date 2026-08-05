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

PRIMARY_INPUT = (
    TABLES
    / "phase2C2B_fiji_primary_vs_recall_breadth_tests.tsv"
)

GLOBAL_INPUT = (
    TABLES
    / "phase2C2B_fiji_recall_dose_global_tests.tsv"
)

TREND_INPUT = (
    TABLES
    / "phase2C2B_fiji_recall_dose_trend_tests.tsv"
)

PAIRWISE_INPUT = (
    TABLES
    / "phase2C2B_fiji_recall_dose_pairwise_tests.tsv"
)

CALIBRATION_INPUT = (
    TABLES
    / "phase2C2B_fiji_raw_vs_bpv_breadth_registry.tsv"
)

C2B_DECISION_INPUT = (
    TABLES
    / "phase2C2B_fiji_breadth_inference_decision.tsv"
)

EXPECTED_C2B_DECISION = (
    "READY_FOR_PHASE2C2C_BREADTH_SYNTHESIS_AND_PHASE2C3_FUNCTIONAL_COUPLING"
)

MECHANISTIC_AXES = [
    "binding_antibody_abundance",
    "igg_subclass_architecture",
    "fc_receptor_communication",
    "global_shared_serology",
]

STATUS_ORDER = {
    "supported_raw_and_bpv_calibrated": 1,
    "attenuated_after_bpv_calibration": 2,
    "direction_changed_after_bpv_calibration": 3,
    "emerges_after_bpv_calibration": 4,
    "not_fdr_significant": 5,
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


def metric_label(
    metric: str,
) -> str:
    labels = {
        "mean_cross_reactive_score": (
            "mean response across five cross-reactive HPV types"
        ),
        "median_cross_reactive_score": (
            "median response across five cross-reactive HPV types"
        ),
        "minimum_cross_reactive_score": (
            "minimum response retained across the five HPV types"
        ),
        "cross_antigen_standard_deviation": (
            "between-antigen response heterogeneity"
        ),
        "positive_antigen_count": (
            "number of cross-reactive HPV types with positive response"
        ),
        "twofold_antigen_count": (
            "number of cross-reactive HPV types exceeding a twofold response"
        ),
    }

    return labels.get(
        str(metric),
        str(metric),
    )


def axis_label(
    axis: str,
) -> str:
    labels = {
        "binding_antibody_abundance": (
            "binding-antibody abundance"
        ),
        "igg_subclass_architecture": (
            "IgG-subclass architecture"
        ),
        "fc_receptor_communication": (
            "Fc-receptor communication"
        ),
        "global_shared_serology": (
            "global shared systems-serology"
        ),
    }

    return labels.get(
        str(axis),
        str(axis),
    )


def biological_sentence(
    row: pd.Series,
) -> str:
    axis = axis_label(
        str(row["mechanistic_axis"])
    )

    metric = metric_label(
        str(row["metric"])
    )

    effect = float(
        row["bpv_calibrated_effect"]
    )

    q_value = float(
        row["bpv_calibrated_q_value"]
    )

    if str(row["metric"]) in {
        "positive_antigen_count",
        "twofold_antigen_count",
    }:
        effect_text = (
            f"{effect:.1f} additional cross-reactive HPV types"
        )
    else:
        effect_text = (
            f"{effect:.3f} log2-score units"
        )

    return (
        f"Heterologous recall showed greater {axis}, with the "
        f"{metric} higher by {effect_text} than primary induction "
        f"after matched BPV calibration (BH q={q_value:.3g})."
    )


def prepare_calibration_registry(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    output = frame.copy()

    for column in [
        "raw_effect",
        "bpv_calibrated_effect",
        "raw_q_value",
        "bpv_calibrated_q_value",
    ]:
        output[column] = pd.to_numeric(
            output[column],
            errors="coerce",
        )

    output[
        "status_order"
    ] = output[
        "bpv_calibration_status"
    ].map(
        STATUS_ORDER
    ).fillna(99)

    return output


def construct_claim_ready(
    registry: pd.DataFrame,
) -> pd.DataFrame:
    claim_ready = registry[
        registry[
            "bpv_calibration_status"
        ]
        == "supported_raw_and_bpv_calibrated"
    ].copy()

    claim_ready[
        "evidence_grade"
    ] = (
        "A1_crossreactive_breadth_supported_raw_and_bpv_calibrated"
    )

    claim_ready[
        "claim_readiness"
    ] = "claim_ready"

    claim_ready[
        "biological_interpretation"
    ] = claim_ready.apply(
        biological_sentence,
        axis=1,
    )

    selected_columns = [
        "analysis_type",
        "mechanistic_axis",
        "metric",
        "comparison",
        "raw_effect",
        "bpv_calibrated_effect",
        "raw_q_value",
        "bpv_calibrated_q_value",
        "bpv_calibration_status",
        "evidence_grade",
        "claim_readiness",
        "biological_interpretation",
    ]

    return claim_ready[
        selected_columns
    ].sort_values(
        [
            "bpv_calibrated_q_value",
            "mechanistic_axis",
            "metric",
        ]
    ).reset_index(
        drop=True
    )


def construct_qualified(
    registry: pd.DataFrame,
) -> pd.DataFrame:
    qualified = registry[
        registry[
            "bpv_calibration_status"
        ].isin(
            [
                "attenuated_after_bpv_calibration",
                "direction_changed_after_bpv_calibration",
                "emerges_after_bpv_calibration",
            ]
        )
    ].copy()

    grade_map = {
        "attenuated_after_bpv_calibration": (
            "B1_raw_only_attenuated_after_bpv_calibration"
        ),
        "direction_changed_after_bpv_calibration": (
            "C1_direction_unstable_after_bpv_calibration"
        ),
        "emerges_after_bpv_calibration": (
            "C2_emerges_after_bpv_calibration"
        ),
    }

    readiness_map = {
        "attenuated_after_bpv_calibration": (
            "qualified_raw_only"
        ),
        "direction_changed_after_bpv_calibration": (
            "exploratory_direction_unstable"
        ),
        "emerges_after_bpv_calibration": (
            "exploratory_calibrated_only"
        ),
    }

    qualified[
        "evidence_grade"
    ] = qualified[
        "bpv_calibration_status"
    ].map(
        grade_map
    )

    qualified[
        "claim_readiness"
    ] = qualified[
        "bpv_calibration_status"
    ].map(
        readiness_map
    )

    qualified[
        "interpretation_guardrail"
    ] = np.select(
        [
            qualified[
                "bpv_calibration_status"
            ]
            == "attenuated_after_bpv_calibration",
            qualified[
                "bpv_calibration_status"
            ]
            == "direction_changed_after_bpv_calibration",
            qualified[
                "bpv_calibration_status"
            ]
            == "emerges_after_bpv_calibration",
        ],
        [
            (
                "Raw association did not remain FDR significant after "
                "matched BPV calibration and should not be promoted as "
                "an HPV-associated breadth effect."
            ),
            (
                "Effect direction differed after BPV calibration; both "
                "raw and calibrated results were non-robust and this "
                "comparison is exploratory."
            ),
            (
                "Association appeared only after BPV calibration and "
                "requires exploratory treatment."
            ),
        ],
        default="Qualification required.",
    )

    selected_columns = [
        "analysis_type",
        "mechanistic_axis",
        "metric",
        "comparison",
        "raw_effect",
        "bpv_calibrated_effect",
        "raw_q_value",
        "bpv_calibrated_q_value",
        "bpv_calibration_status",
        "evidence_grade",
        "claim_readiness",
        "interpretation_guardrail",
    ]

    return qualified[
        selected_columns
    ].sort_values(
        [
            "evidence_grade",
            "bpv_calibrated_q_value",
            "raw_q_value",
        ],
        na_position="last",
    ).reset_index(
        drop=True
    )


def construct_axis_summary(
    registry: pd.DataFrame,
    claim_ready: pd.DataFrame,
    qualified: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for axis in MECHANISTIC_AXES:
        axis_claim = claim_ready[
            claim_ready[
                "mechanistic_axis"
            ]
            == axis
        ]

        axis_qualified = qualified[
            qualified[
                "mechanistic_axis"
            ]
            == axis
        ]

        attenuated = axis_qualified[
            axis_qualified[
                "bpv_calibration_status"
            ]
            == "attenuated_after_bpv_calibration"
        ]

        unstable = axis_qualified[
            axis_qualified[
                "bpv_calibration_status"
            ]
            == "direction_changed_after_bpv_calibration"
        ]

        if axis_claim.empty:
            metrics = ""
            minimum_q = np.nan
            median_effect = np.nan
            maximum_effect = np.nan
        else:
            metrics = ", ".join(
                sorted(
                    axis_claim[
                        "metric"
                    ].astype(str).unique()
                )
            )

            minimum_q = float(
                axis_claim[
                    "bpv_calibrated_q_value"
                ].min()
            )

            median_effect = float(
                axis_claim[
                    "bpv_calibrated_effect"
                ].median()
            )

            maximum_effect = float(
                axis_claim[
                    "bpv_calibrated_effect"
                ].max()
            )

        rows.append(
            {
                "mechanistic_axis": axis,
                "axis_label": axis_label(axis),
                "claim_ready_findings": len(
                    axis_claim
                ),
                "claim_ready_metrics": metrics,
                "minimum_bpv_calibrated_q_value": minimum_q,
                "median_bpv_calibrated_effect": median_effect,
                "maximum_bpv_calibrated_effect": maximum_effect,
                "raw_only_attenuated_findings": len(
                    attenuated
                ),
                "direction_unstable_findings": len(
                    unstable
                ),
            }
        )

    return pd.DataFrame(rows)


def analysis_fdr_counts(
    frame: pd.DataFrame,
    analysis_type: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for matrix_type, subset in frame.groupby(
        "matrix_type",
        observed=True,
    ):
        q_values = pd.to_numeric(
            subset[
                "bh_q_value"
            ],
            errors="coerce",
        )

        rows.append(
            {
                "analysis_type": analysis_type,
                "matrix_type": matrix_type,
                "tests": len(subset),
                "fdr_significant_tests": int(
                    (
                        q_values
                        < 0.05
                    ).sum()
                ),
                "minimum_q_value": float(
                    q_values.min()
                ),
            }
        )

    return rows


def construct_analysis_summary(
    primary: pd.DataFrame,
    global_tests: pd.DataFrame,
    trend: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    rows.extend(
        analysis_fdr_counts(
            primary,
            "primary_vs_recall",
        )
    )

    rows.extend(
        analysis_fdr_counts(
            global_tests,
            "recall_dose_global",
        )
    )

    rows.extend(
        analysis_fdr_counts(
            trend,
            "recall_dose_trend",
        )
    )

    rows.extend(
        analysis_fdr_counts(
            pairwise,
            "recall_dose_pairwise",
        )
    )

    return pd.DataFrame(rows).sort_values(
        [
            "analysis_type",
            "matrix_type",
        ]
    ).reset_index(
        drop=True
    )


def main() -> None:
    required_paths = [
        PRIMARY_INPUT,
        GLOBAL_INPUT,
        TREND_INPUT,
        PAIRWISE_INPUT,
        CALIBRATION_INPUT,
        C2B_DECISION_INPUT,
    ]

    for path in required_paths:
        if not path.exists():
            sys.exit(
                f"ERROR: Required input is missing: {path}"
            )

    decision_frame = pd.read_csv(
        C2B_DECISION_INPUT,
        sep="\t",
    )

    observed_decision = str(
        decision_frame.loc[
            0,
            "decision",
        ]
    )

    if observed_decision != EXPECTED_C2B_DECISION:
        sys.exit(
            "ERROR: Phase 2C2B decision is "
            f"{observed_decision}; expected "
            f"{EXPECTED_C2B_DECISION}."
        )

    primary = pd.read_csv(
        PRIMARY_INPUT,
        sep="\t",
    )

    global_tests = pd.read_csv(
        GLOBAL_INPUT,
        sep="\t",
    )

    trend = pd.read_csv(
        TREND_INPUT,
        sep="\t",
    )

    pairwise = pd.read_csv(
        PAIRWISE_INPUT,
        sep="\t",
    )

    registry = pd.read_csv(
        CALIBRATION_INPUT,
        sep="\t",
    )

    require_columns(
        registry,
        {
            "analysis_type",
            "mechanistic_axis",
            "metric",
            "comparison",
            "raw_effect",
            "bpv_calibrated_effect",
            "raw_q_value",
            "bpv_calibrated_q_value",
            "bpv_calibration_status",
        },
        "Raw-versus-BPV breadth registry",
    )

    registry = prepare_calibration_registry(
        registry
    )

    claim_ready = construct_claim_ready(
        registry
    )

    qualified = construct_qualified(
        registry
    )

    axis_summary = construct_axis_summary(
        registry,
        claim_ready,
        qualified,
    )

    analysis_summary = construct_analysis_summary(
        primary,
        global_tests,
        trend,
        pairwise,
    )

    status_counts = (
        registry[
            "bpv_calibration_status"
        ]
        .value_counts()
        .to_dict()
    )

    failures: list[str] = []

    expected_status_counts = {
        "supported_raw_and_bpv_calibrated": 16,
        "attenuated_after_bpv_calibration": 7,
        "direction_changed_after_bpv_calibration": 5,
        "emerges_after_bpv_calibration": 0,
        "not_fdr_significant": 116,
    }

    for status, expected in expected_status_counts.items():
        observed = int(
            status_counts.get(
                status,
                0,
            )
        )

        if observed != expected:
            failures.append(
                f"{status}: expected {expected}, observed {observed}."
            )

    if len(registry) != 144:
        failures.append(
            f"Expected 144 registry rows, observed {len(registry)}."
        )

    if len(claim_ready) != 16:
        failures.append(
            f"Expected 16 claim-ready findings, observed {len(claim_ready)}."
        )

    if len(qualified) != 12:
        failures.append(
            f"Expected 12 qualified/exploratory findings, "
            f"observed {len(qualified)}."
        )

    if not (
        claim_ready[
            "analysis_type"
        ]
        == "primary_vs_recall"
    ).all():
        failures.append(
            "Some claim-ready findings are not primary-versus-recall effects."
        )

    expected_axis_counts = {
        "binding_antibody_abundance": 0,
        "igg_subclass_architecture": 6,
        "fc_receptor_communication": 5,
        "global_shared_serology": 5,
    }

    observed_axis_counts = (
        claim_ready[
            "mechanistic_axis"
        ]
        .value_counts()
        .to_dict()
    )

    for axis, expected in expected_axis_counts.items():
        observed = int(
            observed_axis_counts.get(
                axis,
                0,
            )
        )

        if observed != expected:
            failures.append(
                f"{axis}: expected {expected} claim-ready findings, "
                f"observed {observed}."
            )

    global_significant = int(
        (
            pd.to_numeric(
                global_tests[
                    "bh_q_value"
                ],
                errors="coerce",
            )
            < 0.05
        ).sum()
    )

    pairwise_significant = int(
        (
            pd.to_numeric(
                pairwise[
                    "bh_q_value"
                ],
                errors="coerce",
            )
            < 0.05
        ).sum()
    )

    calibrated_trend_significant = int(
        (
            (
                trend[
                    "matrix_type"
                ]
                == "bpv_calibrated_log2_change"
            )
            & (
                pd.to_numeric(
                    trend[
                        "bh_q_value"
                    ],
                    errors="coerce",
                )
                < 0.05
            )
        ).sum()
    )

    raw_trend_significant = int(
        (
            (
                trend[
                    "matrix_type"
                ]
                == "raw_log2_change"
            )
            & (
                pd.to_numeric(
                    trend[
                        "bh_q_value"
                    ],
                    errors="coerce",
                )
                < 0.05
            )
        ).sum()
    )

    if global_significant != 0:
        failures.append(
            f"Expected 0 recall global FDR findings, "
            f"observed {global_significant}."
        )

    if pairwise_significant != 0:
        failures.append(
            f"Expected 0 recall pairwise FDR findings, "
            f"observed {pairwise_significant}."
        )

    if calibrated_trend_significant != 0:
        failures.append(
            "BPV-calibrated recall-dose trends unexpectedly reached FDR."
        )

    if raw_trend_significant != 5:
        failures.append(
            f"Expected 5 raw recall-dose trends, "
            f"observed {raw_trend_significant}."
        )

    decision_value = (
        "READY_FOR_PHASE2C2_COMMIT_AND_PHASE2C3_FUNCTIONAL_COUPLING"
        if not failures
        else "PHASE2C2C_REPAIR_REQUIRED"
    )

    claim_output = (
        TABLES
        / "phase2C2C_fiji_claim_ready_breadth_findings.tsv"
    )

    qualified_output = (
        TABLES
        / "phase2C2C_fiji_qualified_breadth_findings.tsv"
    )

    axis_output = (
        TABLES
        / "phase2C2C_fiji_breadth_axis_summary.tsv"
    )

    analysis_output = (
        TABLES
        / "phase2C2C_fiji_breadth_analysis_summary.tsv"
    )

    decision_output = (
        TABLES
        / "phase2C2C_fiji_breadth_synthesis_decision.tsv"
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
        axis_summary,
        axis_output,
    )

    write_tsv(
        analysis_summary,
        analysis_output,
    )

    decision = pd.DataFrame(
        [
            {
                "decision": decision_value,
                "calibration_registry_rows": len(registry),
                "claim_ready_findings": len(claim_ready),
                "qualified_or_exploratory_findings": len(qualified),
                "supported_raw_and_bpv_calibrated": int(
                    status_counts.get(
                        "supported_raw_and_bpv_calibrated",
                        0,
                    )
                ),
                "attenuated_after_bpv_calibration": int(
                    status_counts.get(
                        "attenuated_after_bpv_calibration",
                        0,
                    )
                ),
                "direction_changed_after_bpv_calibration": int(
                    status_counts.get(
                        "direction_changed_after_bpv_calibration",
                        0,
                    )
                ),
                "raw_recall_dose_trends": raw_trend_significant,
                "bpv_calibrated_recall_dose_trends": (
                    calibrated_trend_significant
                ),
                "recall_global_fdr_findings": global_significant,
                "recall_pairwise_fdr_findings": pairwise_significant,
                "validation_failures": "; ".join(failures),
            }
        ]
    )

    write_tsv(
        decision,
        decision_output,
    )

    report_path = (
        REPORTS
        / "phase2C2C_fiji_crossreactive_breadth_synthesis_report.md"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as report:
        report.write(
            "# Phase 2C2C Fiji cross-reactive breadth synthesis\n\n"
        )

        report.write("## Decision\n\n")
        report.write(f"**{decision_value}**\n\n")

        report.write("## Principal biological result\n\n")

        report.write(
            "Heterologous 2vHPV recall after previous 4vHPV vaccination "
            "produced a broader cross-reactive systems-serology state than "
            "primary 2vHPV induction. Sixteen breadth findings remained "
            "FDR significant in both the raw and matched BPV-calibrated "
            "analyses. These findings were concentrated in IgG-subclass "
            "architecture, Fc-receptor communication and the global shared "
            "serology axis.\n\n"
        )

        report.write(
            "BPV-calibrated recall effects included higher mean and median "
            "cross-reactive scores, stronger minimum responses retained "
            "across HPV31, HPV33, HPV45, HPV52 and HPV58, and larger numbers "
            "of HPV types with positive or greater-than-twofold responses. "
            "This identifies memory recall as a coordinated expansion of "
            "cross-reactive antibody quality and Fc-effector breadth rather "
            "than merely an increase in one antigen or one assay feature.\n\n"
        )

        report.write("## Claim-ready findings\n\n")

        for sentence in claim_ready[
            "biological_interpretation"
        ]:
            report.write(f"- {sentence}\n")

        report.write(
            "\n## Previous-dose-number results\n\n"
        )

        report.write(
            "No global difference among the one-, two- and three-dose "
            "previous-4vHPV groups survived FDR correction, and no pairwise "
            "dose-group comparison survived FDR correction. Five negative "
            "raw dose trends were detected, principally in IgG-subclass "
            "breadth, but none remained FDR significant after matched BPV "
            "calibration. Therefore, the principal supported distinction is "
            "primary induction versus established memory recall, not a "
            "monotonic or categorical previous-dose-number effect.\n\n"
        )

        report.write("## BPV calibration\n\n")

        report.write(
            "Seven raw findings were attenuated after BPV calibration. "
            "These included the raw binding-antibody breadth effects and "
            "the five raw recall-dose trends. Five additional comparisons "
            "changed direction after calibration, but these were not FDR "
            "significant in either representation. BPV-calibrated breadth "
            "therefore provides the preferred HPV-associated interpretation.\n\n"
        )

        report.write("## Interpretation boundary\n\n")

        report.write(
            "The Fiji breadth phenotype represents downstream antibody "
            "quantity, subclass architecture and Fc-receptor communication. "
            "It is compatible with memory B-cell recall, affinity-matured "
            "antibody repertoires and expanded Fc-effector communication, "
            "but it does not directly measure VLP uptake, intracellular "
            "routing, antigen processing, HLA-II presentation or Tfh–B-cell "
            "interactions. Those cellular mechanisms require integration "
            "with separate molecular datasets.\n"
        )

    print("===== PHASE 2C2C COMPLETE =====")
    print(f"Decision: {decision_value}")
    print(
        f"Claim-ready findings: {len(claim_ready)}"
    )
    print(
        "Qualified/exploratory findings: "
        f"{len(qualified)}"
    )
    print(
        f"Raw recall-dose trends: {raw_trend_significant}"
    )
    print(
        "BPV-calibrated recall-dose trends: "
        f"{calibrated_trend_significant}"
    )
    print(
        f"Recall global FDR findings: {global_significant}"
    )
    print(
        f"Recall pairwise FDR findings: {pairwise_significant}"
    )
    print(f"Report: {report_path}")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
