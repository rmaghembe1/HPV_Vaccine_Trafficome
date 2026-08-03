#!/usr/bin/env python3

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Iterable

try:
    import numpy as np
    import pandas as pd
    from scipy.stats import binomtest
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

MIXED_DECISION = (
    TABLES
    / "phase2B2A_fiji_mixed_model_decision.tsv"
)

PHASE2B1_WITHIN = (
    TABLES
    / "phase2B1_fiji_within_trajectory_tests.tsv"
)

PHASE2B1_PRIMARY_RECALL = (
    TABLES
    / "phase2B1_fiji_primary_vs_recall_tests.tsv"
)


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, na_rep="")


def finite_array(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


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


def above_floor(
    values: pd.Series,
    floor_values: pd.Series,
) -> pd.Series:
    values_numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    floors_numeric = pd.to_numeric(
        floor_values,
        errors="coerce",
    )

    tolerance = np.maximum(
        np.abs(floors_numeric.to_numpy(dtype=float)) * 1e-12,
        1e-12,
    )

    return pd.Series(
        values_numeric.to_numpy(dtype=float)
        > floors_numeric.to_numpy(dtype=float)
        + tolerance,
        index=values.index,
    )


def classify_row(row: pd.Series) -> str:
    continuous_q = row.get("continuous_q_value", np.nan)
    continuous_effect = row.get("continuous_effect", np.nan)

    if not np.isfinite(continuous_q) or continuous_q >= 0.05:
        return "continuous_result_not_fdr_significant"

    supportive = False
    available = False

    for prefix in [
        "detection",
        "conditional",
    ]:
        q_value = row.get(
            f"{prefix}_q_value",
            np.nan,
        )

        effect = row.get(
            f"{prefix}_effect",
            np.nan,
        )

        if np.isfinite(q_value):
            available = True

        if (
            np.isfinite(q_value)
            and q_value < 0.05
            and np.isfinite(effect)
            and np.sign(effect)
            == np.sign(continuous_effect)
        ):
            supportive = True

    if supportive:
        return "sensitivity_supported"

    if available:
        return "not_supported_after_floor_sensitivity"

    return "sensitivity_inconclusive"


def main() -> None:
    for path in [
        PAIRED_INPUT,
        MIXED_DECISION,
        PHASE2B1_WITHIN,
        PHASE2B1_PRIMARY_RECALL,
    ]:
        if not path.exists():
            sys.exit(
                f"ERROR: Required input missing: {path}"
            )

    mixed_decision = pd.read_csv(
        MIXED_DECISION,
        sep="\t",
    )

    if str(mixed_decision.loc[0, "decision"]) != (
        "READY_FOR_PHASE2B2B_FLOOR_SENSITIVITY"
    ):
        sys.exit(
            "ERROR: Phase 2B2A does not authorize floor sensitivity."
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
        "v1_numeric",
        "v2_numeric",
        "log2_change_authoritative",
        "v1_floor_value",
        "v2_floor_value",
        "paired_floor_severity",
    }

    missing = required - set(paired.columns)

    if missing:
        sys.exit(
            "ERROR: Missing columns: "
            + ", ".join(sorted(missing))
        )

    floor_data = paired[
        paired["paired_floor_severity"].isin(
            ["moderate", "high"]
        )
    ].copy()

    floor_features = (
        floor_data[
            [
                "antigen_target",
                "feature",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "antigen_target",
                "feature",
            ]
        )
    )

    detection_rows: list[dict[str, object]] = []
    conditional_rows: list[dict[str, object]] = []

    for _, feature_key in floor_features.iterrows():
        antigen = str(feature_key["antigen_target"])
        feature = str(feature_key["feature"])

        feature_group = floor_data[
            (floor_data["antigen_target"] == antigen)
            & (floor_data["feature"] == feature)
        ].copy()

        feature_group["above_v1"] = above_floor(
            feature_group["v1_numeric"],
            feature_group["v1_floor_value"],
        )

        feature_group["above_v2"] = above_floor(
            feature_group["v2_numeric"],
            feature_group["v2_floor_value"],
        )

        feature_group["detection_transition_score"] = (
            feature_group["above_v2"].astype(int)
            - feature_group["above_v1"].astype(int)
        )

        feature_group["above_both_visits"] = (
            feature_group["above_v1"]
            & feature_group["above_v2"]
        )

        metadata = {
            "antigen_target": antigen,
            "antigen_class": str(
                feature_group["antigen_class"].iloc[0]
            ),
            "feature": feature,
            "assay_family": str(
                feature_group["assay_family"].iloc[0]
            ),
            "outcome_family": str(
                feature_group["outcome_family"].iloc[0]
            ),
            "paired_floor_severity": str(
                feature_group["paired_floor_severity"].iloc[0]
            ),
        }

        for dose in [0, 1, 2, 3]:
            group = feature_group[
                feature_group["previous_4vHPV_doses"] == dose
            ].copy()

            gained = int(
                (
                    (~group["above_v1"])
                    & group["above_v2"]
                ).sum()
            )

            lost = int(
                (
                    group["above_v1"]
                    & (~group["above_v2"])
                ).sum()
            )

            stable_below = int(
                (
                    (~group["above_v1"])
                    & (~group["above_v2"])
                ).sum()
            )

            stable_above = int(
                (
                    group["above_v1"]
                    & group["above_v2"]
                ).sum()
            )

            discordant = gained + lost

            p_value = (
                float(
                    binomtest(
                        min(gained, lost),
                        n=discordant,
                        p=0.5,
                        alternative="two-sided",
                    ).pvalue
                )
                if discordant > 0
                else 1.0
            )

            sensitivity_id = (
                f"within|{antigen}|{feature}|dose{dose}"
            )

            detection_rows.append(
                {
                    "sensitivity_id": sensitivity_id,
                    "contrast_scope": "within_trajectory",
                    "previous_4vHPV_doses": dose,
                    **metadata,
                    "participants": len(group),
                    "stable_below": stable_below,
                    "gained_above_floor": gained,
                    "lost_above_floor": lost,
                    "stable_above": stable_above,
                    "discordant_pairs": discordant,
                    "effect": (
                        (gained - lost) / len(group)
                        if len(group)
                        else np.nan
                    ),
                    "odds_ratio_gain_vs_loss": (
                        (gained + 0.5)
                        / (lost + 0.5)
                    ),
                    "p_value": p_value,
                    "bh_family": (
                        f"{metadata['outcome_family']}|"
                        "floor_detection|within_trajectory"
                    ),
                }
            )

            conditional_group = group[
                group["above_both_visits"]
            ]

            conditional_test = one_sample_test(
                conditional_group[
                    "log2_change_authoritative"
                ]
            )

            conditional_rows.append(
                {
                    "sensitivity_id": sensitivity_id,
                    "contrast_scope": "within_trajectory",
                    "previous_4vHPV_doses": dose,
                    **metadata,
                    "participants_total": len(group),
                    "participants_above_both": int(
                        conditional_test["n"]
                    ),
                    "effect": conditional_test["effect"],
                    "standard_error": (
                        conditional_test["standard_error"]
                    ),
                    "ci95_lower": (
                        conditional_test["ci95_lower"]
                    ),
                    "ci95_upper": (
                        conditional_test["ci95_upper"]
                    ),
                    "test_statistic": (
                        conditional_test["test_statistic"]
                    ),
                    "degrees_freedom": (
                        conditional_test["degrees_freedom"]
                    ),
                    "p_value": conditional_test["p_value"],
                    "bh_family": (
                        f"{metadata['outcome_family']}|"
                        "conditional_magnitude|within_trajectory"
                    ),
                }
            )

        if antigen != "BPV":
            primary = feature_group[
                feature_group["previous_4vHPV_doses"] == 0
            ]

            recall = feature_group[
                feature_group["previous_4vHPV_doses"] > 0
            ]

            sensitivity_id = (
                f"primary_recall|{antigen}|{feature}"
            )

            detection_test = welch_test(
                primary["detection_transition_score"],
                recall["detection_transition_score"],
            )

            detection_rows.append(
                {
                    "sensitivity_id": sensitivity_id,
                    "contrast_scope": "primary_minus_pooled_recall",
                    "previous_4vHPV_doses": "",
                    **metadata,
                    "participants": (
                        int(detection_test["n_first"])
                        + int(detection_test["n_second"])
                    ),
                    "primary_participants": int(
                        detection_test["n_first"]
                    ),
                    "recall_participants": int(
                        detection_test["n_second"]
                    ),
                    "effect": detection_test["effect"],
                    "standard_error": (
                        detection_test["standard_error"]
                    ),
                    "ci95_lower": (
                        detection_test["ci95_lower"]
                    ),
                    "ci95_upper": (
                        detection_test["ci95_upper"]
                    ),
                    "test_statistic": (
                        detection_test["test_statistic"]
                    ),
                    "degrees_freedom": (
                        detection_test["degrees_freedom"]
                    ),
                    "p_value": detection_test["p_value"],
                    "bh_family": (
                        f"{metadata['outcome_family']}|"
                        "floor_detection|primary_minus_pooled_recall"
                    ),
                }
            )

            primary_conditional = primary[
                primary["above_both_visits"]
            ]

            recall_conditional = recall[
                recall["above_both_visits"]
            ]

            conditional_test = welch_test(
                primary_conditional[
                    "log2_change_authoritative"
                ],
                recall_conditional[
                    "log2_change_authoritative"
                ],
            )

            conditional_rows.append(
                {
                    "sensitivity_id": sensitivity_id,
                    "contrast_scope": "primary_minus_pooled_recall",
                    "previous_4vHPV_doses": "",
                    **metadata,
                    "participants_total": len(feature_group),
                    "primary_participants_above_both": int(
                        conditional_test["n_first"]
                    ),
                    "recall_participants_above_both": int(
                        conditional_test["n_second"]
                    ),
                    "effect": conditional_test["effect"],
                    "standard_error": (
                        conditional_test["standard_error"]
                    ),
                    "ci95_lower": (
                        conditional_test["ci95_lower"]
                    ),
                    "ci95_upper": (
                        conditional_test["ci95_upper"]
                    ),
                    "test_statistic": (
                        conditional_test["test_statistic"]
                    ),
                    "degrees_freedom": (
                        conditional_test["degrees_freedom"]
                    ),
                    "p_value": conditional_test["p_value"],
                    "bh_family": (
                        f"{metadata['outcome_family']}|"
                        "conditional_magnitude|primary_minus_pooled_recall"
                    ),
                }
            )

    detection = apply_bh(
        pd.DataFrame(detection_rows)
    )

    conditional = apply_bh(
        pd.DataFrame(conditional_rows)
    )

    continuous_within = pd.read_csv(
        PHASE2B1_WITHIN,
        sep="\t",
    )

    continuous_within = continuous_within[
        continuous_within["maximum_floor_severity"].isin(
            ["moderate", "high"]
        )
    ].copy()

    continuous_within["sensitivity_id"] = (
        "within|"
        + continuous_within["antigen_target"].astype(str)
        + "|"
        + continuous_within["feature"].astype(str)
        + "|dose"
        + continuous_within[
            "previous_4vHPV_doses"
        ].astype(str)
    )

    continuous_within = continuous_within[
        [
            "sensitivity_id",
            "antigen_target",
            "feature",
            "previous_4vHPV_doses",
            "mean_log2_change",
            "q_value",
        ]
    ].rename(
        columns={
            "mean_log2_change": "continuous_effect",
            "q_value": "continuous_q_value",
        }
    )

    continuous_primary_recall = pd.read_csv(
        PHASE2B1_PRIMARY_RECALL,
        sep="\t",
    )

    continuous_primary_recall = continuous_primary_recall[
        continuous_primary_recall[
            "maximum_floor_severity"
        ].isin(
            ["moderate", "high"]
        )
    ].copy()

    continuous_primary_recall["sensitivity_id"] = (
        "primary_recall|"
        + continuous_primary_recall[
            "antigen_target"
        ].astype(str)
        + "|"
        + continuous_primary_recall[
            "feature"
        ].astype(str)
    )

    continuous_primary_recall = continuous_primary_recall[
        [
            "sensitivity_id",
            "antigen_target",
            "feature",
            "difference_log2",
            "q_value",
        ]
    ].rename(
        columns={
            "difference_log2": "continuous_effect",
            "q_value": "continuous_q_value",
        }
    )

    continuous_registry = pd.concat(
        [
            continuous_within,
            continuous_primary_recall,
        ],
        ignore_index=True,
        sort=False,
    )

    robustness = continuous_registry.merge(
        detection[
            [
                "sensitivity_id",
                "effect",
                "q_value",
            ]
        ].rename(
            columns={
                "effect": "detection_effect",
                "q_value": "detection_q_value",
            }
        ),
        on="sensitivity_id",
        how="left",
        validate="one_to_one",
    )

    robustness = robustness.merge(
        conditional[
            [
                "sensitivity_id",
                "effect",
                "q_value",
            ]
        ].rename(
            columns={
                "effect": "conditional_effect",
                "q_value": "conditional_q_value",
            }
        ),
        on="sensitivity_id",
        how="left",
        validate="one_to_one",
    )

    robustness["robustness_classification"] = robustness.apply(
        classify_row,
        axis=1,
    )

    hpv_floor_features = floor_features[
        floor_features["antigen_target"] != "BPV"
    ]

    expected_rows = (
        len(floor_features) * 4
        + len(hpv_floor_features)
    )
    failures: list[str] = []

    if len(detection) != expected_rows:
        failures.append(
            f"Expected {expected_rows} detection rows, "
            f"observed {len(detection)}."
        )

    if len(conditional) != expected_rows:
        failures.append(
            f"Expected {expected_rows} conditional rows, "
            f"observed {len(conditional)}."
        )

    if len(robustness) != expected_rows:
        failures.append(
            f"Expected {expected_rows} robustness rows, "
            f"observed {len(robustness)}."
        )

    decision_value = (
        "READY_FOR_PHASE2B2C_BPV_CALIBRATED_INFERENCE"
        if not failures
        else "PHASE2B2B_REPAIR_REQUIRED"
    )

    classification_counts = (
        robustness[
            "robustness_classification"
        ]
        .value_counts()
        .to_dict()
    )

    decision_frame = pd.DataFrame(
        [
            {
                "decision": decision_value,
                "floor_prone_antigen_feature_pairs": (
                    len(floor_features)
                ),
                "detection_test_rows": len(detection),
                "conditional_magnitude_rows": len(conditional),
                "robustness_registry_rows": len(robustness),
                "sensitivity_supported": int(
                    classification_counts.get(
                        "sensitivity_supported",
                        0,
                    )
                ),
                "not_supported_after_floor_sensitivity": int(
                    classification_counts.get(
                        "not_supported_after_floor_sensitivity",
                        0,
                    )
                ),
                "sensitivity_inconclusive": int(
                    classification_counts.get(
                        "sensitivity_inconclusive",
                        0,
                    )
                ),
                "validation_failures": "; ".join(failures),
            }
        ]
    )

    write_tsv(
        detection,
        TABLES / "phase2B2B_fiji_floor_detection_tests.tsv",
    )

    write_tsv(
        conditional,
        TABLES / "phase2B2B_fiji_conditional_magnitude_tests.tsv",
    )

    write_tsv(
        robustness,
        TABLES / "phase2B2B_fiji_floor_robustness_registry.tsv",
    )

    write_tsv(
        decision_frame,
        TABLES / "phase2B2B_fiji_floor_sensitivity_decision.tsv",
    )

    report_path = (
        REPORTS
        / "phase2B2B_fiji_floor_sensitivity_report.md"
    )

    with report_path.open("w", encoding="utf-8") as report:
        report.write(
            "# Phase 2B2B Fiji floor-sensitive analysis\n\n"
        )
        report.write("## Decision\n\n")
        report.write(f"**{decision_value}**\n\n")
        report.write(
            f"- Moderate/high-floor antigen-feature pairs: "
            f"{len(floor_features)}\n"
        )
        report.write(
            f"- Detection-transition tests: {len(detection)}\n"
        )
        report.write(
            f"- Conditional magnitude tests: {len(conditional)}\n"
        )
        report.write(
            f"- Sensitivity-supported continuous results: "
            f"{classification_counts.get('sensitivity_supported', 0)}\n"
        )
        report.write(
            f"- Continuous results not supported after sensitivity "
            f"analysis: "
            f"{classification_counts.get('not_supported_after_floor_sensitivity', 0)}\n\n"
        )
        report.write(
            "The detection component evaluates movement above or below "
            "the observed assay floor. The conditional component evaluates "
            "log2 response magnitude only among participants above the "
            "floor at both visits.\n"
        )

    print("===== PHASE 2B2B COMPLETE =====")
    print(f"Decision: {decision_value}")
    print(
        "Floor-prone antigen-feature pairs: "
        f"{len(floor_features)}"
    )
    print(f"Detection rows: {len(detection)}")
    print(f"Conditional rows: {len(conditional)}")
    print(f"Robustness rows: {len(robustness)}")
    print(f"Report: {report_path}")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
