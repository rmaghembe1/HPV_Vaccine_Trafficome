#!/usr/bin/env python3

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Iterable

try:
    import numpy as np
    import pandas as pd
    from scipy.stats import f as f_distribution
    from scipy.stats import t as t_distribution
except ImportError as exc:
    sys.exit(
        "ERROR: numpy, pandas and scipy are required.\n"
        "Install them with:\n"
        "  python -m pip install --user numpy pandas scipy\n"
        f"Original error: {exc}"
    )


ROOT = Path("/mnt/d/HPV_Vaccine_Trafficome_Project")
PROCESSED = ROOT / "07_data_processed" / "fiji_nct02276521"
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

FLOOR_DECISION = (
    TABLES
    / "phase2B2B_fiji_floor_sensitivity_decision.tsv"
)

PHASE2B1_WITHIN = (
    TABLES
    / "phase2B1_fiji_within_trajectory_tests.tsv"
)

SEVERITY_ORDER = {
    "none": 0,
    "low": 1,
    "moderate": 2,
    "high": 3,
}


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, na_rep="")


def finite_array(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def maximum_severity(values: Iterable[str]) -> str:
    cleaned = [
        str(value)
        for value in values
        if pd.notna(value)
        and str(value) in SEVERITY_ORDER
    ]

    if not cleaned:
        return "unresolved"

    return max(
        cleaned,
        key=lambda value: SEVERITY_ORDER[value],
    )


def one_sample_test(values: Iterable[float]) -> dict[str, float]:
    array = finite_array(values)
    n = len(array)

    output = {
        "n": float(n),
        "effect": np.nan,
        "standard_error": np.nan,
        "ci95_lower": np.nan,
        "ci95_upper": np.nan,
        "test_statistic": np.nan,
        "degrees_freedom": np.nan,
        "p_value": np.nan,
    }

    if n == 0:
        return output

    mean = float(np.mean(array))
    output["effect"] = mean

    if n < 2:
        return output

    standard_deviation = float(
        np.std(array, ddof=1)
    )

    standard_error = (
        standard_deviation / math.sqrt(n)
    )

    output["standard_error"] = standard_error
    output["degrees_freedom"] = n - 1

    if standard_error == 0:
        output["test_statistic"] = (
            0.0
            if mean == 0
            else np.sign(mean) * np.inf
        )
        output["p_value"] = (
            1.0
            if mean == 0
            else 0.0
        )
        output["ci95_lower"] = mean
        output["ci95_upper"] = mean
        return output

    statistic = mean / standard_error
    critical = float(
        t_distribution.ppf(
            0.975,
            df=n - 1,
        )
    )

    output["test_statistic"] = statistic
    output["p_value"] = float(
        2
        * t_distribution.sf(
            abs(statistic),
            df=n - 1,
        )
    )
    output["ci95_lower"] = mean - critical * standard_error
    output["ci95_upper"] = mean + critical * standard_error

    return output


def welch_test(
    first: Iterable[float],
    second: Iterable[float],
) -> dict[str, float]:
    a = finite_array(first)
    b = finite_array(second)

    output = {
        "n_first": float(len(a)),
        "n_second": float(len(b)),
        "effect": np.nan,
        "standard_error": np.nan,
        "ci95_lower": np.nan,
        "ci95_upper": np.nan,
        "test_statistic": np.nan,
        "degrees_freedom": np.nan,
        "p_value": np.nan,
    }

    if len(a) == 0 or len(b) == 0:
        return output

    difference = float(
        np.mean(a) - np.mean(b)
    )

    output["effect"] = difference

    if len(a) < 2 or len(b) < 2:
        return output

    variance_a = float(
        np.var(a, ddof=1)
    )
    variance_b = float(
        np.var(b, ddof=1)
    )

    component_a = variance_a / len(a)
    component_b = variance_b / len(b)

    standard_error = math.sqrt(
        component_a + component_b
    )

    output["standard_error"] = standard_error

    if standard_error == 0:
        output["test_statistic"] = (
            0.0
            if difference == 0
            else np.sign(difference) * np.inf
        )
        output["p_value"] = (
            1.0
            if difference == 0
            else 0.0
        )
        output["ci95_lower"] = difference
        output["ci95_upper"] = difference
        return output

    denominator = (
        component_a ** 2 / (len(a) - 1)
        + component_b ** 2 / (len(b) - 1)
    )

    degrees_freedom = (
        (component_a + component_b) ** 2
        / denominator
        if denominator > 0
        else len(a) + len(b) - 2
    )

    statistic = difference / standard_error
    critical = float(
        t_distribution.ppf(
            0.975,
            df=degrees_freedom,
        )
    )

    output["degrees_freedom"] = degrees_freedom
    output["test_statistic"] = statistic
    output["p_value"] = float(
        2
        * t_distribution.sf(
            abs(statistic),
            df=degrees_freedom,
        )
    )
    output["ci95_lower"] = difference - critical * standard_error
    output["ci95_upper"] = difference + critical * standard_error

    return output


def welch_anova(groups: list[np.ndarray]) -> dict[str, float]:
    cleaned = [
        finite_array(group)
        for group in groups
    ]

    output = {
        "participants": float(
            sum(len(group) for group in cleaned)
        ),
        "test_statistic": np.nan,
        "df_numerator": np.nan,
        "df_denominator": np.nan,
        "p_value": np.nan,
    }

    if (
        len(cleaned) < 2
        or any(len(group) < 2 for group in cleaned)
    ):
        return output

    sizes = np.asarray(
        [len(group) for group in cleaned],
        dtype=float,
    )

    means = np.asarray(
        [np.mean(group) for group in cleaned],
        dtype=float,
    )

    variances = np.asarray(
        [
            np.var(group, ddof=1)
            for group in cleaned
        ],
        dtype=float,
    )

    if np.any(variances <= 0):
        return output

    weights = sizes / variances
    total_weight = np.sum(weights)
    weighted_mean = float(
        np.sum(weights * means) / total_weight
    )

    group_count = len(cleaned)

    numerator = float(
        np.sum(
            weights
            * (means - weighted_mean) ** 2
        )
        / (group_count - 1)
    )

    correction = float(
        np.sum(
            (
                1 / (sizes - 1)
            )
            * (
                1
                - weights / total_weight
            ) ** 2
        )
    )

    denominator = (
        1
        + (
            2
            * (group_count - 2)
            / (group_count ** 2 - 1)
        )
        * correction
    )

    statistic = numerator / denominator
    df_numerator = group_count - 1
    df_denominator = (
        (group_count ** 2 - 1)
        / (3 * correction)
    )

    output["test_statistic"] = statistic
    output["df_numerator"] = df_numerator
    output["df_denominator"] = df_denominator
    output["p_value"] = float(
        f_distribution.sf(
            statistic,
            df_numerator,
            df_denominator,
        )
    )

    return output


def bh_adjust(values: pd.Series) -> pd.Series:
    p = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    adjusted = np.full(len(p), np.nan)
    valid = np.where(np.isfinite(p))[0]

    if len(valid) == 0:
        return pd.Series(adjusted, index=values.index)

    order = np.argsort(p[valid])
    ordered_p = p[valid][order]
    count = len(ordered_p)

    ordered_q = ordered_p * count / np.arange(1, count + 1)
    ordered_q = np.minimum.accumulate(ordered_q[::-1])[::-1]
    ordered_q = np.minimum(ordered_q, 1.0)

    inverse = np.empty_like(order)
    inverse[order] = np.arange(count)
    adjusted[valid] = ordered_q[inverse]

    return pd.Series(adjusted, index=values.index)


def apply_bh(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["q_value"] = np.nan

    for _, index in output.groupby("bh_family", dropna=False).groups.items():
        output.loc[index, "q_value"] = bh_adjust(
            output.loc[index, "p_value"]
        )

    output["fdr_significant_0_05"] = np.where(
        output["q_value"] < 0.05,
        "yes",
        "no",
    )

    return output


def calibration_classification(row: pd.Series) -> str:
    raw_q = row["raw_q_value"]
    calibrated_q = row["calibrated_q_value"]

    raw_effect = row["raw_effect"]
    calibrated_effect = row["calibrated_effect"]

    raw_significant = (
        np.isfinite(raw_q)
        and raw_q < 0.05
    )

    calibrated_significant = (
        np.isfinite(calibrated_q)
        and calibrated_q < 0.05
    )

    if (
        np.isfinite(raw_effect)
        and np.isfinite(calibrated_effect)
        and np.sign(raw_effect)
        != np.sign(calibrated_effect)
    ):
        return "direction_changed_after_bpv_calibration"

    if raw_significant and calibrated_significant:
        return "bpv_calibrated_supported"

    if raw_significant and not calibrated_significant:
        return "attenuated_after_bpv_calibration"

    if not raw_significant and calibrated_significant:
        return "emerges_after_bpv_calibration"

    return "not_fdr_significant"


def main() -> None:
    for path in [
        PAIRED_INPUT,
        FLOOR_DECISION,
        PHASE2B1_WITHIN,
    ]:
        if not path.exists():
            sys.exit(
                f"ERROR: Required input missing: {path}"
            )

    floor_decision = pd.read_csv(
        FLOOR_DECISION,
        sep="\t",
    )

    if str(floor_decision.loc[0, "decision"]) != (
        "READY_FOR_PHASE2B2C_BPV_CALIBRATED_INFERENCE"
    ):
        sys.exit(
            "ERROR: Phase 2B2B does not authorize BPV-calibrated inference."
        )

    paired = pd.read_csv(
        PAIRED_INPUT,
        sep="\t",
        dtype={
            "participant_id": "string",
            "previous_4vHPV_doses": "Int64",
            "antigen_target": "string",
            "feature": "string",
        },
    )

    required = {
        "participant_id",
        "previous_4vHPV_doses",
        "antigen_target",
        "antigen_class",
        "feature",
        "assay_family",
        "outcome_family",
        "log2_change_authoritative",
        "paired_floor_severity",
    }

    missing = required - set(paired.columns)

    if missing:
        sys.exit(
            "ERROR: Missing columns: "
            + ", ".join(sorted(missing))
        )

    bpv = paired[
        paired["antigen_target"] == "BPV"
    ][
        [
            "participant_id",
            "previous_4vHPV_doses",
            "feature",
            "log2_change_authoritative",
            "paired_floor_severity",
        ]
    ].rename(
        columns={
            "log2_change_authoritative": "bpv_log2_change",
            "paired_floor_severity": "bpv_floor_severity",
        }
    )

    shared_features = sorted(
        bpv["feature"].dropna().unique()
    )

    hpv = paired[
        (paired["antigen_target"] != "BPV")
        & paired["feature"].isin(shared_features)
    ].copy()

    calibrated = hpv.merge(
        bpv,
        on=[
            "participant_id",
            "previous_4vHPV_doses",
            "feature",
        ],
        how="inner",
        validate="many_to_one",
    )

    calibrated["bpv_calibrated_log2_change"] = (
        calibrated["log2_change_authoritative"]
        - calibrated["bpv_log2_change"]
    )

    calibrated["combined_floor_severity"] = [
        maximum_severity(
            [hpv_floor, bpv_floor]
        )
        for hpv_floor, bpv_floor in zip(
            calibrated["paired_floor_severity"],
            calibrated["bpv_floor_severity"],
        )
    ]

    within_rows: list[dict[str, object]] = []
    global_rows: list[dict[str, object]] = []
    pairwise_rows: list[dict[str, object]] = []
    primary_recall_rows: list[dict[str, object]] = []

    for (
        antigen,
        feature,
    ), group in calibrated.groupby(
        [
            "antigen_target",
            "feature",
        ],
        dropna=False,
        observed=True,
    ):
        metadata = {
            "antigen_target": str(antigen),
            "antigen_class": str(
                group["antigen_class"].iloc[0]
            ),
            "feature": str(feature),
            "assay_family": str(
                group["assay_family"].iloc[0]
            ),
            "outcome_family": str(
                group["outcome_family"].iloc[0]
            ),
            "combined_floor_severity": maximum_severity(
                group["combined_floor_severity"]
            ),
        }

        for dose in [0, 1, 2, 3]:
            dose_group = group[
                group["previous_4vHPV_doses"] == dose
            ]

            result = one_sample_test(
                dose_group[
                    "bpv_calibrated_log2_change"
                ]
            )

            within_rows.append(
                {
                    "contrast_type": "within_trajectory",
                    "previous_4vHPV_doses": dose,
                    **metadata,
                    "participants": int(result["n"]),
                    "effect": result["effect"],
                    "standard_error": (
                        result["standard_error"]
                    ),
                    "ci95_lower": result["ci95_lower"],
                    "ci95_upper": result["ci95_upper"],
                    "test_statistic": (
                        result["test_statistic"]
                    ),
                    "degrees_freedom": (
                        result["degrees_freedom"]
                    ),
                    "p_value": result["p_value"],
                    "bpv_calibrated_ratio": (
                        2 ** result["effect"]
                        if np.isfinite(result["effect"])
                        else np.nan
                    ),
                    "bh_family": (
                        f"{metadata['outcome_family']}|"
                        "bpv_calibrated_within"
                    ),
                }
            )

        recall_groups = [
            group.loc[
                group["previous_4vHPV_doses"] == dose,
                "bpv_calibrated_log2_change",
            ].to_numpy()
            for dose in [1, 2, 3]
        ]

        global_result = welch_anova(
            recall_groups
        )

        global_rows.append(
            {
                "contrast_type": "global_recall_dose_effect",
                **metadata,
                "participants": int(
                    global_result["participants"]
                ),
                "test_statistic": (
                    global_result["test_statistic"]
                ),
                "df_numerator": (
                    global_result["df_numerator"]
                ),
                "df_denominator": (
                    global_result["df_denominator"]
                ),
                "p_value": global_result["p_value"],
                "bh_family": (
                    f"{metadata['outcome_family']}|"
                    "bpv_calibrated_global_recall"
                ),
            }
        )

        for higher, lower in [
            (2, 1),
            (3, 1),
            (3, 2),
        ]:
            higher_values = group.loc[
                group["previous_4vHPV_doses"] == higher,
                "bpv_calibrated_log2_change",
            ]

            lower_values = group.loc[
                group["previous_4vHPV_doses"] == lower,
                "bpv_calibrated_log2_change",
            ]

            result = welch_test(
                higher_values,
                lower_values,
            )

            pairwise_rows.append(
                {
                    "contrast_type": "pairwise_recall_dose_contrast",
                    "higher_dose": higher,
                    "lower_dose": lower,
                    "contrast_label": (
                        f"dose{higher}_minus_dose{lower}"
                    ),
                    **metadata,
                    "n_higher": int(
                        result["n_first"]
                    ),
                    "n_lower": int(
                        result["n_second"]
                    ),
                    "effect": result["effect"],
                    "standard_error": (
                        result["standard_error"]
                    ),
                    "ci95_lower": result["ci95_lower"],
                    "ci95_upper": result["ci95_upper"],
                    "test_statistic": (
                        result["test_statistic"]
                    ),
                    "degrees_freedom": (
                        result["degrees_freedom"]
                    ),
                    "p_value": result["p_value"],
                    "ratio_effect": (
                        2 ** result["effect"]
                        if np.isfinite(result["effect"])
                        else np.nan
                    ),
                    "bh_family": (
                        f"{metadata['outcome_family']}|"
                        "bpv_calibrated_pairwise_recall"
                    ),
                }
            )

        primary = group.loc[
            group["previous_4vHPV_doses"] == 0,
            "bpv_calibrated_log2_change",
        ]

        recall = group.loc[
            group["previous_4vHPV_doses"] > 0,
            "bpv_calibrated_log2_change",
        ]

        result = welch_test(
            primary,
            recall,
        )

        primary_recall_rows.append(
            {
                "contrast_type": "primary_minus_pooled_recall",
                **metadata,
                "primary_participants": int(
                    result["n_first"]
                ),
                "recall_participants": int(
                    result["n_second"]
                ),
                "effect": result["effect"],
                "standard_error": (
                    result["standard_error"]
                ),
                "ci95_lower": result["ci95_lower"],
                "ci95_upper": result["ci95_upper"],
                "test_statistic": (
                    result["test_statistic"]
                ),
                "degrees_freedom": (
                    result["degrees_freedom"]
                ),
                "p_value": result["p_value"],
                "ratio_effect": (
                    2 ** result["effect"]
                    if np.isfinite(result["effect"])
                    else np.nan
                ),
                "bh_family": (
                    f"{metadata['outcome_family']}|"
                    "bpv_calibrated_primary_recall"
                ),
            }
        )

    within = apply_bh(
        pd.DataFrame(within_rows)
    )

    global_tests = apply_bh(
        pd.DataFrame(global_rows)
    )

    pairwise = apply_bh(
        pd.DataFrame(pairwise_rows)
    )

    primary_recall = apply_bh(
        pd.DataFrame(primary_recall_rows)
    )

    raw_within = pd.read_csv(
        PHASE2B1_WITHIN,
        sep="\t",
    )

    comparison = within.merge(
        raw_within[
            [
                "antigen_target",
                "feature",
                "previous_4vHPV_doses",
                "mean_log2_change",
                "q_value",
            ]
        ].rename(
            columns={
                "mean_log2_change": "raw_effect",
                "q_value": "raw_q_value",
            }
        ),
        on=[
            "antigen_target",
            "feature",
            "previous_4vHPV_doses",
        ],
        how="left",
        validate="one_to_one",
    )

    comparison = comparison.rename(
        columns={
            "effect": "calibrated_effect",
            "q_value": "calibrated_q_value",
        }
    )

    comparison["calibration_classification"] = comparison.apply(
        calibration_classification,
        axis=1,
    )

    expected_calibrated_records = 80 * 77
    expected_within = 77 * 4
    expected_global = 77
    expected_pairwise = 77 * 3
    expected_primary_recall = 77

    failures: list[str] = []

    observed_contracts = [
        (
            "calibrated participant records",
            len(calibrated),
            expected_calibrated_records,
        ),
        (
            "within tests",
            len(within),
            expected_within,
        ),
        (
            "global tests",
            len(global_tests),
            expected_global,
        ),
        (
            "pairwise tests",
            len(pairwise),
            expected_pairwise,
        ),
        (
            "primary-recall tests",
            len(primary_recall),
            expected_primary_recall,
        ),
    ]

    for label, observed, expected in observed_contracts:
        if observed != expected:
            failures.append(
                f"{label}: expected {expected}, observed {observed}"
            )

    decision_value = (
        "READY_FOR_PHASE2B2_INTEGRATION_AND_BIOLOGICAL_SYNTHESIS"
        if not failures
        else "PHASE2B2C_REPAIR_REQUIRED"
    )

    classification_counts = (
        comparison[
            "calibration_classification"
        ]
        .value_counts()
        .to_dict()
    )

    decision_frame = pd.DataFrame(
        [
            {
                "decision": decision_value,
                "shared_bpv_features": len(shared_features),
                "hpv_antigens_calibrated": int(
                    calibrated[
                        "antigen_target"
                    ].nunique()
                ),
                "calibrated_participant_records": len(calibrated),
                "within_test_rows": len(within),
                "global_test_rows": len(global_tests),
                "pairwise_test_rows": len(pairwise),
                "primary_recall_rows": len(primary_recall),
                "bpv_calibrated_supported": int(
                    classification_counts.get(
                        "bpv_calibrated_supported",
                        0,
                    )
                ),
                "attenuated_after_bpv_calibration": int(
                    classification_counts.get(
                        "attenuated_after_bpv_calibration",
                        0,
                    )
                ),
                "emerges_after_bpv_calibration": int(
                    classification_counts.get(
                        "emerges_after_bpv_calibration",
                        0,
                    )
                ),
                "direction_changed_after_bpv_calibration": int(
                    classification_counts.get(
                        "direction_changed_after_bpv_calibration",
                        0,
                    )
                ),
                "validation_failures": "; ".join(failures),
            }
        ]
    )

    write_tsv(
        calibrated,
        PROCESSED / "phase2B2C_fiji_bpv_calibrated_participant_effects.tsv",
    )

    write_tsv(
        within,
        TABLES / "phase2B2C_fiji_bpv_calibrated_within_tests.tsv",
    )

    write_tsv(
        global_tests,
        TABLES / "phase2B2C_fiji_bpv_calibrated_global_dose_tests.tsv",
    )

    write_tsv(
        pairwise,
        TABLES / "phase2B2C_fiji_bpv_calibrated_pairwise_tests.tsv",
    )

    write_tsv(
        primary_recall,
        TABLES / "phase2B2C_fiji_bpv_calibrated_primary_recall_tests.tsv",
    )

    write_tsv(
        comparison,
        TABLES / "phase2B2C_fiji_raw_vs_bpv_calibrated_registry.tsv",
    )

    write_tsv(
        decision_frame,
        TABLES / "phase2B2C_fiji_bpv_calibrated_decision.tsv",
    )

    report_path = (
        REPORTS
        / "phase2B2C_fiji_bpv_calibrated_inference_report.md"
    )

    with report_path.open("w", encoding="utf-8") as report:
        report.write(
            "# Phase 2B2C Fiji BPV-calibrated inference\n\n"
        )
        report.write("## Decision\n\n")
        report.write(f"**{decision_value}**\n\n")
        report.write(
            f"- Shared BPV assay features: {len(shared_features)}\n"
        )
        report.write(
            f"- HPV antigens calibrated: "
            f"{calibrated['antigen_target'].nunique()}\n"
        )
        report.write(
            f"- Participant-level calibrated records: "
            f"{len(calibrated)}\n"
        )
        report.write(
            f"- BPV-calibrated supported within effects: "
            f"{classification_counts.get('bpv_calibrated_supported', 0)}\n"
        )
        report.write(
            f"- Raw effects attenuated after BPV calibration: "
            f"{classification_counts.get('attenuated_after_bpv_calibration', 0)}\n\n"
        )
        report.write(
            "The calibrated response is the participant-matched HPV "
            "log2 change minus the contemporaneous BPV log2 change for "
            "the same antibody feature. This separates HPV-associated "
            "remodeling from heterologous-control or assay-wide movement.\n"
        )

    print("===== PHASE 2B2C COMPLETE =====")
    print(f"Decision: {decision_value}")
    print(
        "Calibrated participant records: "
        f"{len(calibrated)}"
    )
    print(f"Within tests: {len(within)}")
    print(f"Global tests: {len(global_tests)}")
    print(f"Pairwise tests: {len(pairwise)}")
    print(
        "Primary-recall tests: "
        f"{len(primary_recall)}"
    )
    print(f"Report: {report_path}")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
