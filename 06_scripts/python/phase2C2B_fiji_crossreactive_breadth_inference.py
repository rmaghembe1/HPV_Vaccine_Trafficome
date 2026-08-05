#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import (
    kruskal,
    mannwhitneyu,
    spearmanr,
    ttest_ind,
)
from statsmodels.stats.multitest import multipletests


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

BREADTH_INPUT = (
    PROCESSED
    / "phase2C2A_fiji_crossreactive_breadth_scores.tsv"
)

C2A_DECISION_INPUT = (
    TABLES
    / "phase2C2A_fiji_crossreactive_breadth_decision.tsv"
)

EXPECTED_C2A_DECISION = (
    "READY_FOR_PHASE2C2B_CROSSREACTIVE_BREADTH_INFERENCE"
)

METRICS = [
    "mean_cross_reactive_score",
    "median_cross_reactive_score",
    "minimum_cross_reactive_score",
    "cross_antigen_standard_deviation",
    "positive_antigen_count",
    "twofold_antigen_count",
]

MATRIX_TYPES = [
    "raw_log2_change",
    "bpv_calibrated_log2_change",
]

MECHANISTIC_AXES = [
    "binding_antibody_abundance",
    "igg_subclass_architecture",
    "fc_receptor_communication",
    "global_shared_serology",
]

DOSE_PAIRS = [
    (1, 2),
    (1, 3),
    (2, 3),
]


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


def add_bh_fdr(
    frame: pd.DataFrame,
    group_columns: list[str],
    p_column: str,
    q_column: str = "bh_q_value",
) -> pd.DataFrame:
    output = frame.copy()

    output[q_column] = np.nan

    grouped = output.groupby(
        group_columns,
        observed=True,
        dropna=False,
    )

    for _, indices in grouped.groups.items():
        index_list = list(indices)

        p_values = pd.to_numeric(
            output.loc[
                index_list,
                p_column,
            ],
            errors="coerce",
        )

        valid = p_values.notna()

        if not valid.any():
            continue

        adjusted = multipletests(
            p_values.loc[valid],
            method="fdr_bh",
        )[1]

        valid_indices = p_values.loc[
            valid
        ].index

        output.loc[
            valid_indices,
            q_column,
        ] = adjusted

    output[
        "fdr_significant"
    ] = (
        output[q_column]
        < 0.05
    )

    return output


def hedges_g(
    group_b: np.ndarray,
    group_a: np.ndarray,
) -> float:
    group_b = np.asarray(
        group_b,
        dtype=float,
    )

    group_a = np.asarray(
        group_a,
        dtype=float,
    )

    n_b = len(group_b)
    n_a = len(group_a)

    if n_b < 2 or n_a < 2:
        return np.nan

    variance_b = np.var(
        group_b,
        ddof=1,
    )

    variance_a = np.var(
        group_a,
        ddof=1,
    )

    denominator_df = (
        n_b
        + n_a
        - 2
    )

    pooled_variance = (
        (
            (n_b - 1)
            * variance_b
        )
        + (
            (n_a - 1)
            * variance_a
        )
    ) / denominator_df

    if (
        not np.isfinite(
            pooled_variance
        )
        or pooled_variance <= 0
    ):
        return np.nan

    cohens_d = (
        np.mean(group_b)
        - np.mean(group_a)
    ) / np.sqrt(
        pooled_variance
    )

    correction = (
        1
        - (
            3
            / (
                4
                * denominator_df
                - 1
            )
        )
    )

    return float(
        correction
        * cohens_d
    )


def two_group_statistics(
    group_b: pd.Series,
    group_a: pd.Series,
) -> dict[str, float]:
    group_b = pd.to_numeric(
        group_b,
        errors="coerce",
    ).dropna().to_numpy(
        dtype=float
    )

    group_a = pd.to_numeric(
        group_a,
        errors="coerce",
    ).dropna().to_numpy(
        dtype=float
    )

    if not len(group_b) or not len(group_a):
        return {
            "n_group_a": len(group_a),
            "n_group_b": len(group_b),
            "mean_group_a": np.nan,
            "mean_group_b": np.nan,
            "mean_difference_group_b_minus_a": np.nan,
            "median_group_a": np.nan,
            "median_group_b": np.nan,
            "median_difference_group_b_minus_a": np.nan,
            "mann_whitney_u": np.nan,
            "mann_whitney_p_value": np.nan,
            "rank_biserial_group_b_minus_a": np.nan,
            "welch_t_statistic": np.nan,
            "welch_p_value": np.nan,
            "hedges_g_group_b_minus_a": np.nan,
        }

    mann = mannwhitneyu(
        group_b,
        group_a,
        alternative="two-sided",
        method="auto",
    )

    welch = ttest_ind(
        group_b,
        group_a,
        equal_var=False,
        nan_policy="omit",
    )

    rank_biserial = (
        (
            2
            * float(mann.statistic)
        )
        / (
            len(group_b)
            * len(group_a)
        )
        - 1
    )

    return {
        "n_group_a": len(group_a),
        "n_group_b": len(group_b),
        "mean_group_a": float(
            np.mean(group_a)
        ),
        "mean_group_b": float(
            np.mean(group_b)
        ),
        "mean_difference_group_b_minus_a": float(
            np.mean(group_b)
            - np.mean(group_a)
        ),
        "median_group_a": float(
            np.median(group_a)
        ),
        "median_group_b": float(
            np.median(group_b)
        ),
        "median_difference_group_b_minus_a": float(
            np.median(group_b)
            - np.median(group_a)
        ),
        "mann_whitney_u": float(
            mann.statistic
        ),
        "mann_whitney_p_value": float(
            mann.pvalue
        ),
        "rank_biserial_group_b_minus_a": float(
            rank_biserial
        ),
        "welch_t_statistic": float(
            welch.statistic
        ),
        "welch_p_value": float(
            welch.pvalue
        ),
        "hedges_g_group_b_minus_a": hedges_g(
            group_b,
            group_a,
        ),
    }


def primary_vs_recall_tests(
    breadth: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for matrix_type in MATRIX_TYPES:
        for axis in MECHANISTIC_AXES:
            subset = breadth[
                (
                    breadth[
                        "matrix_type"
                    ]
                    == matrix_type
                )
                & (
                    breadth[
                        "mechanistic_axis"
                    ]
                    == axis
                )
            ]

            primary = subset[
                subset[
                    "previous_4vHPV_doses"
                ]
                == 0
            ]

            recall = subset[
                subset[
                    "previous_4vHPV_doses"
                ]
                > 0
            ]

            for metric in METRICS:
                statistics = two_group_statistics(
                    recall[metric],
                    primary[metric],
                )

                rows.append(
                    {
                        "matrix_type": matrix_type,
                        "mechanistic_axis": axis,
                        "metric": metric,
                        "group_a": "primary_dose0",
                        "group_b": "recall_all_doses",
                        **statistics,
                    }
                )

    tests = pd.DataFrame(rows)

    tests = add_bh_fdr(
        tests,
        group_columns=[
            "matrix_type",
        ],
        p_column="mann_whitney_p_value",
    )

    return tests


def recall_global_tests(
    breadth: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    recall = breadth[
        breadth[
            "previous_4vHPV_doses"
        ]
        > 0
    ]

    for matrix_type in MATRIX_TYPES:
        for axis in MECHANISTIC_AXES:
            subset = recall[
                (
                    recall[
                        "matrix_type"
                    ]
                    == matrix_type
                )
                & (
                    recall[
                        "mechanistic_axis"
                    ]
                    == axis
                )
            ]

            for metric in METRICS:
                groups = [
                    pd.to_numeric(
                        subset.loc[
                            subset[
                                "previous_4vHPV_doses"
                            ]
                            == dose,
                            metric,
                        ],
                        errors="coerce",
                    )
                    .dropna()
                    .to_numpy(
                        dtype=float
                    )
                    for dose in [
                        1,
                        2,
                        3,
                    ]
                ]

                try:
                    result = kruskal(
                        *groups,
                        nan_policy="omit",
                    )

                    statistic = float(
                        result.statistic
                    )

                    p_value = float(
                        result.pvalue
                    )
                except ValueError:
                    statistic = 0.0
                    p_value = 1.0

                total_n = sum(
                    len(group)
                    for group in groups
                )

                group_count = 3

                epsilon_squared = (
                    (
                        statistic
                        - group_count
                        + 1
                    )
                    / (
                        total_n
                        - group_count
                    )
                    if total_n > group_count
                    else np.nan
                )

                epsilon_squared = max(
                    float(
                        epsilon_squared
                    ),
                    0.0,
                )

                rows.append(
                    {
                        "matrix_type": matrix_type,
                        "mechanistic_axis": axis,
                        "metric": metric,
                        "dose1_n": len(
                            groups[0]
                        ),
                        "dose2_n": len(
                            groups[1]
                        ),
                        "dose3_n": len(
                            groups[2]
                        ),
                        "dose1_median": float(
                            np.median(
                                groups[0]
                            )
                        ),
                        "dose2_median": float(
                            np.median(
                                groups[1]
                            )
                        ),
                        "dose3_median": float(
                            np.median(
                                groups[2]
                            )
                        ),
                        "kruskal_wallis_h": statistic,
                        "kruskal_wallis_p_value": p_value,
                        "epsilon_squared": epsilon_squared,
                    }
                )

    tests = pd.DataFrame(rows)

    tests = add_bh_fdr(
        tests,
        group_columns=[
            "matrix_type",
        ],
        p_column="kruskal_wallis_p_value",
    )

    return tests


def recall_trend_tests(
    breadth: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    recall = breadth[
        breadth[
            "previous_4vHPV_doses"
        ]
        > 0
    ]

    for matrix_type in MATRIX_TYPES:
        for axis in MECHANISTIC_AXES:
            subset = recall[
                (
                    recall[
                        "matrix_type"
                    ]
                    == matrix_type
                )
                & (
                    recall[
                        "mechanistic_axis"
                    ]
                    == axis
                )
            ]

            for metric in METRICS:
                values = pd.to_numeric(
                    subset[metric],
                    errors="coerce",
                )

                doses = pd.to_numeric(
                    subset[
                        "previous_4vHPV_doses"
                    ],
                    errors="coerce",
                )

                valid = (
                    values.notna()
                    & doses.notna()
                )

                result = spearmanr(
                    doses.loc[valid],
                    values.loc[valid],
                )

                rows.append(
                    {
                        "matrix_type": matrix_type,
                        "mechanistic_axis": axis,
                        "metric": metric,
                        "participants": int(
                            valid.sum()
                        ),
                        "spearman_rho": float(
                            result.statistic
                        ),
                        "spearman_p_value": float(
                            result.pvalue
                        ),
                    }
                )

    tests = pd.DataFrame(rows)

    tests = add_bh_fdr(
        tests,
        group_columns=[
            "matrix_type",
        ],
        p_column="spearman_p_value",
    )

    return tests


def recall_pairwise_tests(
    breadth: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    recall = breadth[
        breadth[
            "previous_4vHPV_doses"
        ]
        > 0
    ]

    for matrix_type in MATRIX_TYPES:
        for axis in MECHANISTIC_AXES:
            subset = recall[
                (
                    recall[
                        "matrix_type"
                    ]
                    == matrix_type
                )
                & (
                    recall[
                        "mechanistic_axis"
                    ]
                    == axis
                )
            ]

            for dose_a, dose_b in DOSE_PAIRS:
                comparison = (
                    f"dose{dose_b}_minus_dose{dose_a}"
                )

                group_a = subset[
                    subset[
                        "previous_4vHPV_doses"
                    ]
                    == dose_a
                ]

                group_b = subset[
                    subset[
                        "previous_4vHPV_doses"
                    ]
                    == dose_b
                ]

                for metric in METRICS:
                    statistics = two_group_statistics(
                        group_b[metric],
                        group_a[metric],
                    )

                    rows.append(
                        {
                            "matrix_type": matrix_type,
                            "mechanistic_axis": axis,
                            "metric": metric,
                            "dose_a": dose_a,
                            "dose_b": dose_b,
                            "comparison": comparison,
                            **statistics,
                        }
                    )

    tests = pd.DataFrame(rows)

    tests = add_bh_fdr(
        tests,
        group_columns=[
            "matrix_type",
            "comparison",
        ],
        p_column="mann_whitney_p_value",
    )

    return tests


def calibration_comparison(
    frame: pd.DataFrame,
    analysis_type: str,
    keys: list[str],
    effect_column: str,
    directional: bool,
) -> pd.DataFrame:
    raw = frame[
        frame[
            "matrix_type"
        ]
        == "raw_log2_change"
    ][
        keys
        + [
            effect_column,
            "bh_q_value",
        ]
    ].rename(
        columns={
            effect_column: "raw_effect",
            "bh_q_value": "raw_q_value",
        }
    )

    calibrated = frame[
        frame[
            "matrix_type"
        ]
        == "bpv_calibrated_log2_change"
    ][
        keys
        + [
            effect_column,
            "bh_q_value",
        ]
    ].rename(
        columns={
            effect_column: "bpv_calibrated_effect",
            "bh_q_value": "bpv_calibrated_q_value",
        }
    )

    merged = raw.merge(
        calibrated,
        on=keys,
        how="outer",
        validate="one_to_one",
    )

    merged[
        "analysis_type"
    ] = analysis_type

    statuses: list[str] = []

    for row in merged.itertuples(
        index=False
    ):
        raw_effect = float(
            row.raw_effect
        )

        calibrated_effect = float(
            row.bpv_calibrated_effect
        )

        raw_q = float(
            row.raw_q_value
        )

        calibrated_q = float(
            row.bpv_calibrated_q_value
        )

        direction_changed = (
            directional
            and np.isfinite(
                raw_effect
            )
            and np.isfinite(
                calibrated_effect
            )
            and (
                raw_effect
                * calibrated_effect
                < 0
            )
        )

        if direction_changed:
            status = (
                "direction_changed_after_bpv_calibration"
            )
        elif (
            raw_q < 0.05
            and calibrated_q < 0.05
        ):
            status = (
                "supported_raw_and_bpv_calibrated"
            )
        elif (
            raw_q < 0.05
            and calibrated_q >= 0.05
        ):
            status = (
                "attenuated_after_bpv_calibration"
            )
        elif (
            raw_q >= 0.05
            and calibrated_q < 0.05
        ):
            status = (
                "emerges_after_bpv_calibration"
            )
        else:
            status = (
                "not_fdr_significant"
            )

        statuses.append(status)

    merged[
        "bpv_calibration_status"
    ] = statuses

    return merged


def build_calibration_registry(
    primary: pd.DataFrame,
    global_tests: pd.DataFrame,
    trend: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> pd.DataFrame:
    frames = [
        calibration_comparison(
            primary,
            analysis_type=(
                "primary_vs_recall"
            ),
            keys=[
                "mechanistic_axis",
                "metric",
            ],
            effect_column=(
                "median_difference_group_b_minus_a"
            ),
            directional=True,
        ),
        calibration_comparison(
            global_tests,
            analysis_type=(
                "recall_dose_global"
            ),
            keys=[
                "mechanistic_axis",
                "metric",
            ],
            effect_column="epsilon_squared",
            directional=False,
        ),
        calibration_comparison(
            trend,
            analysis_type=(
                "recall_dose_trend"
            ),
            keys=[
                "mechanistic_axis",
                "metric",
            ],
            effect_column="spearman_rho",
            directional=True,
        ),
        calibration_comparison(
            pairwise,
            analysis_type=(
                "recall_dose_pairwise"
            ),
            keys=[
                "mechanistic_axis",
                "metric",
                "comparison",
            ],
            effect_column=(
                "median_difference_group_b_minus_a"
            ),
            directional=True,
        ),
    ]

    return pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )


def build_inference_summary(
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for analysis_type, frame in tables.items():
        for matrix_type, subset in frame.groupby(
            "matrix_type",
            observed=True,
        ):
            rows.append(
                {
                    "analysis_type": analysis_type,
                    "matrix_type": matrix_type,
                    "tests": len(
                        subset
                    ),
                    "nominal_p_below_0_05": int(
                        (
                            pd.to_numeric(
                                subset[
                                    "bh_q_value"
                                ],
                                errors="coerce",
                            )
                            < 0.05
                        ).sum()
                    ),
                    "fdr_significant_tests": int(
                        subset[
                            "fdr_significant"
                        ].sum()
                    ),
                    "minimum_q_value": float(
                        pd.to_numeric(
                            subset[
                                "bh_q_value"
                            ],
                            errors="coerce",
                        ).min()
                    ),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    for path in [
        BREADTH_INPUT,
        C2A_DECISION_INPUT,
    ]:
        if not path.exists():
            sys.exit(
                f"ERROR: Required input is missing: {path}"
            )

    c2a_decision = pd.read_csv(
        C2A_DECISION_INPUT,
        sep="\t",
    )

    observed_decision = str(
        c2a_decision.loc[
            0,
            "decision",
        ]
    )

    if observed_decision != EXPECTED_C2A_DECISION:
        sys.exit(
            "ERROR: Phase 2C2A decision is "
            f"{observed_decision}; expected "
            f"{EXPECTED_C2A_DECISION}."
        )

    breadth = pd.read_csv(
        BREADTH_INPUT,
        sep="\t",
        dtype={
            "participant_id": "string",
        },
    )

    require_columns(
        breadth,
        {
            "matrix_type",
            "participant_id",
            "previous_4vHPV_doses",
            "analysis_context",
            "mechanistic_axis",
            "evaluated_antigens",
            *METRICS,
        },
        "Cross-reactive breadth table",
    )

    breadth[
        "previous_4vHPV_doses"
    ] = pd.to_numeric(
        breadth[
            "previous_4vHPV_doses"
        ],
        errors="raise",
    ).astype(int)

    primary = primary_vs_recall_tests(
        breadth
    )

    global_tests = recall_global_tests(
        breadth
    )

    trend = recall_trend_tests(
        breadth
    )

    pairwise = recall_pairwise_tests(
        breadth
    )

    calibration_registry = (
        build_calibration_registry(
            primary,
            global_tests,
            trend,
            pairwise,
        )
    )

    inference_summary = (
        build_inference_summary(
            {
                "primary_vs_recall": primary,
                "recall_dose_global": global_tests,
                "recall_dose_trend": trend,
                "recall_dose_pairwise": pairwise,
            }
        )
    )

    failures: list[str] = []

    expected_counts = {
        "breadth": 640,
        "primary": 48,
        "global": 48,
        "trend": 48,
        "pairwise": 144,
        "calibration_registry": 144,
        "inference_summary": 8,
    }

    observed_counts = {
        "breadth": len(
            breadth
        ),
        "primary": len(
            primary
        ),
        "global": len(
            global_tests
        ),
        "trend": len(
            trend
        ),
        "pairwise": len(
            pairwise
        ),
        "calibration_registry": len(
            calibration_registry
        ),
        "inference_summary": len(
            inference_summary
        ),
    }

    for label, expected in expected_counts.items():
        observed = observed_counts[
            label
        ]

        if observed != expected:
            failures.append(
                f"{label}: expected {expected}, "
                f"observed {observed}."
            )

    for label, frame in [
        ("primary", primary),
        ("global", global_tests),
        ("trend", trend),
        ("pairwise", pairwise),
    ]:
        q_values = pd.to_numeric(
            frame[
                "bh_q_value"
            ],
            errors="coerce",
        )

        if q_values.isna().any():
            failures.append(
                f"{label}: missing BH q-values."
            )

        if not q_values.between(
            0,
            1,
        ).all():
            failures.append(
                f"{label}: invalid BH q-values."
            )

    decision_value = (
        "READY_FOR_PHASE2C2C_BREADTH_SYNTHESIS_AND_PHASE2C3_FUNCTIONAL_COUPLING"
        if not failures
        else "PHASE2C2B_REPAIR_REQUIRED"
    )

    primary_output = (
        TABLES
        / "phase2C2B_fiji_primary_vs_recall_breadth_tests.tsv"
    )

    global_output = (
        TABLES
        / "phase2C2B_fiji_recall_dose_global_tests.tsv"
    )

    trend_output = (
        TABLES
        / "phase2C2B_fiji_recall_dose_trend_tests.tsv"
    )

    pairwise_output = (
        TABLES
        / "phase2C2B_fiji_recall_dose_pairwise_tests.tsv"
    )

    registry_output = (
        TABLES
        / "phase2C2B_fiji_raw_vs_bpv_breadth_registry.tsv"
    )

    summary_output = (
        TABLES
        / "phase2C2B_fiji_breadth_inference_summary.tsv"
    )

    decision_output = (
        TABLES
        / "phase2C2B_fiji_breadth_inference_decision.tsv"
    )

    write_tsv(
        primary,
        primary_output,
    )

    write_tsv(
        global_tests,
        global_output,
    )

    write_tsv(
        trend,
        trend_output,
    )

    write_tsv(
        pairwise,
        pairwise_output,
    )

    write_tsv(
        calibration_registry,
        registry_output,
    )

    write_tsv(
        inference_summary,
        summary_output,
    )

    status_counts = (
        calibration_registry[
            "bpv_calibration_status"
        ]
        .value_counts()
        .to_dict()
    )

    decision = pd.DataFrame(
        [
            {
                "decision": decision_value,
                "participant_breadth_rows": len(
                    breadth
                ),
                "primary_vs_recall_tests": len(
                    primary
                ),
                "recall_global_tests": len(
                    global_tests
                ),
                "recall_trend_tests": len(
                    trend
                ),
                "recall_pairwise_tests": len(
                    pairwise
                ),
                "raw_vs_bpv_registry_rows": len(
                    calibration_registry
                ),
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
                "emerges_after_bpv_calibration": int(
                    status_counts.get(
                        "emerges_after_bpv_calibration",
                        0,
                    )
                ),
                "direction_changed_after_bpv_calibration": int(
                    status_counts.get(
                        "direction_changed_after_bpv_calibration",
                        0,
                    )
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
        / "phase2C2B_fiji_crossreactive_breadth_inference_report.md"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as report:
        report.write(
            "# Phase 2C2B Fiji cross-reactive breadth inference\n\n"
        )

        report.write(
            "## Decision\n\n"
        )

        report.write(
            f"**{decision_value}**\n\n"
        )

        report.write(
            f"- Primary-versus-recall tests: "
            f"{len(primary)}\n"
        )

        report.write(
            f"- Recall-dose global tests: "
            f"{len(global_tests)}\n"
        )

        report.write(
            f"- Recall-dose trend tests: "
            f"{len(trend)}\n"
        )

        report.write(
            f"- Recall-dose pairwise tests: "
            f"{len(pairwise)}\n"
        )

        report.write(
            f"- Raw-versus-BPV comparison rows: "
            f"{len(calibration_registry)}\n\n"
        )

        report.write(
            "The primary-versus-recall analysis compares distinct "
            "immunological states: primary 2vHPV induction in previously "
            "unvaccinated participants versus heterologous recall in "
            "participants previously receiving one, two or three 4vHPV "
            "doses. Recall-dose comparisons are between randomized "
            "schedule groups and must not be described as within-person "
            "waning. BPV-calibrated breadth is the preferred basis for "
            "HPV-associated interpretation.\n"
        )

    print(
        "===== PHASE 2C2B COMPLETE ====="
    )

    print(
        f"Decision: {decision_value}"
    )

    print(
        f"Primary-vs-recall tests: {len(primary)}"
    )

    print(
        f"Recall global tests: {len(global_tests)}"
    )

    print(
        f"Recall trend tests: {len(trend)}"
    )

    print(
        f"Recall pairwise tests: {len(pairwise)}"
    )

    print(
        "Raw-vs-BPV registry rows: "
        f"{len(calibration_registry)}"
    )

    print(
        f"Report: {report_path}"
    )

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
