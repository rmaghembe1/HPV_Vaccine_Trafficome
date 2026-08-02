#!/usr/bin/env python3

from __future__ import annotations

import math
import sys
from itertools import combinations
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

PROCESSED_DIR = (
    ROOT
    / "07_data_processed"
    / "fiji_nct02276521"
)

TABLE_DIR = ROOT / "08_results" / "tables"

REPORT_DIR = (
    ROOT
    / "02_dataset_audit"
    / "hpv_specific"
    / "fiji_nct02276521"
)

PAIRED_INPUT = (
    PROCESSED_DIR
    / "phase2A_fiji_paired_effects_analysis_ready.tsv"
)

PHASE2A_DECISION = (
    TABLE_DIR
    / "phase2A_fiji_descriptive_landscape_decision.tsv"
)

SEVERITY_ORDER = {
    "none": 0,
    "low": 1,
    "moderate": 2,
    "high": 3,
    "unresolved": 4,
}

DOSE_PAIRS = [
    (2, 1),
    (3, 1),
    (3, 2),
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


def finite_array(
    values: Iterable[float],
) -> np.ndarray:
    array = np.asarray(
        list(values),
        dtype=float,
    )

    return array[
        np.isfinite(array)
    ]


def maximum_severity(
    values: Iterable[str],
) -> str:
    observed = [
        str(value)
        for value in values
        if pd.notna(value)
        and str(value) in SEVERITY_ORDER
    ]

    if not observed:
        return "unresolved"

    return max(
        observed,
        key=lambda value: (
            SEVERITY_ORDER[value]
        ),
    )


def sensitivity_requirement(
    severity: str,
) -> str:
    if severity == "high":
        return "two_part_floor_sensitivity_required"

    if severity == "moderate":
        return "above_floor_binary_sensitivity_required"

    if severity == "low":
        return "continuous_model_plus_floor_summary"

    if severity == "none":
        return "standard_continuous_model"

    return "manual_review_required"


def one_sample_test(
    values: Iterable[float],
    null_value: float = 0.0,
) -> dict[str, float]:
    array = finite_array(values)
    n = len(array)

    result = {
        "n": float(n),
        "mean": np.nan,
        "sd": np.nan,
        "se": np.nan,
        "ci95_lower": np.nan,
        "ci95_upper": np.nan,
        "test_statistic": np.nan,
        "degrees_freedom": np.nan,
        "p_value": np.nan,
        "cohen_dz": np.nan,
    }

    if n == 0:
        return result

    mean = float(np.mean(array))

    result["mean"] = mean

    if n < 2:
        return result

    sd = float(
        np.std(
            array,
            ddof=1,
        )
    )

    se = sd / math.sqrt(n)

    result["sd"] = sd
    result["se"] = se
    result["degrees_freedom"] = float(
        n - 1
    )

    if sd == 0:
        if mean == null_value:
            result["test_statistic"] = 0.0
            result["p_value"] = 1.0
            result["ci95_lower"] = mean
            result["ci95_upper"] = mean
            result["cohen_dz"] = 0.0
        else:
            result["test_statistic"] = (
                math.copysign(
                    math.inf,
                    mean - null_value,
                )
            )
            result["p_value"] = 0.0
            result["ci95_lower"] = mean
            result["ci95_upper"] = mean
            result["cohen_dz"] = (
                math.copysign(
                    math.inf,
                    mean - null_value,
                )
            )

        return result

    statistic = (
        mean - null_value
    ) / se

    p_value = float(
        2
        * t_distribution.sf(
            abs(statistic),
            df=n - 1,
        )
    )

    critical = float(
        t_distribution.ppf(
            0.975,
            df=n - 1,
        )
    )

    result["test_statistic"] = float(
        statistic
    )
    result["p_value"] = p_value
    result["ci95_lower"] = (
        mean - critical * se
    )
    result["ci95_upper"] = (
        mean + critical * se
    )
    result["cohen_dz"] = (
        mean - null_value
    ) / sd

    return result


def two_group_welch_test(
    first: Iterable[float],
    second: Iterable[float],
) -> dict[str, float]:
    a = finite_array(first)
    b = finite_array(second)

    n_a = len(a)
    n_b = len(b)

    result = {
        "n_first": float(n_a),
        "n_second": float(n_b),
        "mean_first": np.nan,
        "mean_second": np.nan,
        "difference": np.nan,
        "se": np.nan,
        "ci95_lower": np.nan,
        "ci95_upper": np.nan,
        "test_statistic": np.nan,
        "degrees_freedom": np.nan,
        "p_value": np.nan,
        "hedges_g": np.nan,
    }

    if n_a == 0 or n_b == 0:
        return result

    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    difference = mean_a - mean_b

    result["mean_first"] = mean_a
    result["mean_second"] = mean_b
    result["difference"] = difference

    if n_a < 2 or n_b < 2:
        return result

    variance_a = float(
        np.var(
            a,
            ddof=1,
        )
    )

    variance_b = float(
        np.var(
            b,
            ddof=1,
        )
    )

    component_a = variance_a / n_a
    component_b = variance_b / n_b
    se = math.sqrt(
        component_a + component_b
    )

    result["se"] = se

    if se == 0:
        result["test_statistic"] = (
            0.0
            if difference == 0
            else math.copysign(
                math.inf,
                difference,
            )
        )
        result["p_value"] = (
            1.0
            if difference == 0
            else 0.0
        )
        result["ci95_lower"] = difference
        result["ci95_upper"] = difference
    else:
        numerator = (
            component_a
            + component_b
        ) ** 2

        denominator = (
            component_a ** 2
            / (n_a - 1)
            + component_b ** 2
            / (n_b - 1)
        )

        degrees_freedom = (
            numerator / denominator
            if denominator > 0
            else n_a + n_b - 2
        )

        statistic = difference / se

        p_value = float(
            2
            * t_distribution.sf(
                abs(statistic),
                df=degrees_freedom,
            )
        )

        critical = float(
            t_distribution.ppf(
                0.975,
                df=degrees_freedom,
            )
        )

        result["degrees_freedom"] = (
            degrees_freedom
        )
        result["test_statistic"] = (
            statistic
        )
        result["p_value"] = p_value
        result["ci95_lower"] = (
            difference
            - critical * se
        )
        result["ci95_upper"] = (
            difference
            + critical * se
        )

    pooled_variance = (
        (
            (n_a - 1) * variance_a
            + (n_b - 1) * variance_b
        )
        / (n_a + n_b - 2)
    )

    if pooled_variance > 0:
        pooled_sd = math.sqrt(
            pooled_variance
        )

        cohen_d = difference / pooled_sd

        correction = (
            1
            - 3
            / (
                4
                * (
                    n_a
                    + n_b
                )
                - 9
            )
        )

        result["hedges_g"] = (
            correction * cohen_d
        )

    return result


def welch_anova(
    groups: list[np.ndarray],
) -> dict[str, float]:
    cleaned = [
        finite_array(group)
        for group in groups
    ]

    result = {
        "group_count": float(
            len(cleaned)
        ),
        "total_n": float(
            sum(
                len(group)
                for group in cleaned
            )
        ),
        "test_statistic": np.nan,
        "df_numerator": np.nan,
        "df_denominator": np.nan,
        "p_value": np.nan,
        "range_of_group_means": np.nan,
    }

    if (
        len(cleaned) < 2
        or any(
            len(group) < 2
            for group in cleaned
        )
    ):
        return result

    sizes = np.asarray(
        [
            len(group)
            for group in cleaned
        ],
        dtype=float,
    )

    means = np.asarray(
        [
            np.mean(group)
            for group in cleaned
        ],
        dtype=float,
    )

    variances = np.asarray(
        [
            np.var(
                group,
                ddof=1,
            )
            for group in cleaned
        ],
        dtype=float,
    )

    if np.any(
        variances <= 0
    ):
        return result

    weights = sizes / variances
    total_weight = np.sum(weights)

    weighted_mean = float(
        np.sum(
            weights * means
        )
        / total_weight
    )

    k = len(cleaned)

    numerator = float(
        np.sum(
            weights
            * (
                means
                - weighted_mean
            ) ** 2
        )
        / (k - 1)
    )

    correction_term = float(
        np.sum(
            (
                1
                / (
                    sizes
                    - 1
                )
            )
            * (
                1
                - weights
                / total_weight
            ) ** 2
        )
    )

    denominator = (
        1
        + (
            2
            * (
                k
                - 2
            )
            / (
                k ** 2
                - 1
            )
        )
        * correction_term
    )

    statistic = (
        numerator / denominator
    )

    df_numerator = float(
        k - 1
    )

    df_denominator = float(
        (
            k ** 2
            - 1
        )
        / (
            3
            * correction_term
        )
    )

    p_value = float(
        f_distribution.sf(
            statistic,
            df_numerator,
            df_denominator,
        )
    )

    result["test_statistic"] = statistic
    result["df_numerator"] = (
        df_numerator
    )
    result["df_denominator"] = (
        df_denominator
    )
    result["p_value"] = p_value
    result["range_of_group_means"] = float(
        np.max(means)
        - np.min(means)
    )

    return result


def bh_adjust(
    p_values: pd.Series,
) -> pd.Series:
    values = pd.to_numeric(
        p_values,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    adjusted = np.full(
        len(values),
        np.nan,
        dtype=float,
    )

    valid_indices = np.where(
        np.isfinite(values)
    )[0]

    if len(valid_indices) == 0:
        return pd.Series(
            adjusted,
            index=p_values.index,
        )

    valid_values = values[
        valid_indices
    ]

    ordering = np.argsort(
        valid_values
    )

    ordered_values = valid_values[
        ordering
    ]

    m = len(ordered_values)

    ordered_adjusted = (
        ordered_values
        * m
        / np.arange(
            1,
            m + 1,
        )
    )

    ordered_adjusted = (
        np.minimum.accumulate(
            ordered_adjusted[::-1]
        )[::-1]
    )

    ordered_adjusted = np.minimum(
        ordered_adjusted,
        1.0,
    )

    reverse = np.empty_like(
        ordering
    )

    reverse[ordering] = np.arange(
        m
    )

    adjusted_valid = (
        ordered_adjusted[
            reverse
        ]
    )

    adjusted[
        valid_indices
    ] = adjusted_valid

    return pd.Series(
        adjusted,
        index=p_values.index,
    )


def apply_bh(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    output = frame.copy()

    output["q_value"] = np.nan

    for _, indices in output.groupby(
        "bh_family",
        dropna=False,
    ).groups.items():
        output.loc[
            indices,
            "q_value",
        ] = bh_adjust(
            output.loc[
                indices,
                "p_value",
            ]
        )

    output["fdr_significant_0_05"] = np.where(
        output["q_value"] < 0.05,
        "yes",
        "no",
    )

    return output


def shared_metadata(
    group: pd.DataFrame,
) -> dict[str, object]:
    return {
        "antigen_target": str(
            group[
                "antigen_target"
            ].iloc[0]
        ),
        "antigen_class": str(
            group[
                "antigen_class"
            ].iloc[0]
        ),
        "feature": str(
            group[
                "feature"
            ].iloc[0]
        ),
        "assay_family": str(
            group[
                "assay_family"
            ].iloc[0]
        ),
        "outcome_family": str(
            group[
                "outcome_family"
            ].iloc[0]
        ),
        "maximum_floor_severity": (
            maximum_severity(
                group[
                    "paired_floor_severity"
                ]
            )
        ),
    }


def main() -> None:
    for required_file in [
        PAIRED_INPUT,
        PHASE2A_DECISION,
    ]:
        if not required_file.exists():
            sys.exit(
                "ERROR: Required input "
                f"missing: {required_file}"
            )

    phase2a = pd.read_csv(
        PHASE2A_DECISION,
        sep="\t",
    )

    if str(
        phase2a.loc[
            0,
            "decision",
        ]
    ) != (
        "READY_FOR_PHASE2B_INFERENTIAL_MODELING"
    ):
        sys.exit(
            "ERROR: Phase 2A decision "
            "does not authorize inference."
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

    required_columns = {
        "participant_id",
        "previous_4vHPV_doses",
        "antigen_target",
        "antigen_class",
        "feature",
        "assay_family",
        "outcome_family",
        "log2_v1_authoritative",
        "log2_v2_authoritative",
        "log2_change_authoritative",
        "paired_floor_severity",
    }

    missing_columns = (
        required_columns
        - set(paired.columns)
    )

    if missing_columns:
        sys.exit(
            "ERROR: Missing paired "
            "columns: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

    if len(paired) != 7360:
        sys.exit(
            "ERROR: Expected 7360 paired "
            f"records; observed {len(paired)}."
        )

    key_columns = [
        "antigen_target",
        "feature",
    ]

    within_rows: list[
        dict[str, object]
    ] = []

    global_rows: list[
        dict[str, object]
    ] = []

    pairwise_rows: list[
        dict[str, object]
    ] = []

    primary_recall_rows: list[
        dict[str, object]
    ] = []

    for _, feature_group in paired.groupby(
        key_columns,
        dropna=False,
        observed=True,
    ):
        metadata = shared_metadata(
            feature_group
        )

        antigen = metadata[
            "antigen_target"
        ]

        outcome_family = metadata[
            "outcome_family"
        ]

        floor_severity = metadata[
            "maximum_floor_severity"
        ]

        sensitivity = sensitivity_requirement(
            floor_severity
        )

        # -----------------------------------------------------
        # Within-group primary, recall and BPV-control effects
        # -----------------------------------------------------

        for dose in [
            0,
            1,
            2,
            3,
        ]:
            dose_group = feature_group[
                feature_group[
                    "previous_4vHPV_doses"
                ] == dose
            ]

            result = one_sample_test(
                dose_group[
                    "log2_change_authoritative"
                ]
            )

            if antigen == "BPV":
                model_id = "HPVVT-FJ-M06"
                model_family = (
                    "Heterologous BPV control"
                )
                contrast_type = (
                    "bpv_within_dose_change"
                )
            elif dose == 0:
                model_id = "HPVVT-FJ-M01"
                model_family = (
                    "Primary 2vHPV induction"
                )
                contrast_type = (
                    "primary_induction"
                )
            else:
                model_id = "HPVVT-FJ-M03"
                model_family = (
                    "Heterologous 2vHPV recall"
                )
                contrast_type = (
                    "dose_specific_recall"
                )

            within_rows.append(
                {
                    "model_id": model_id,
                    "model_family": (
                        model_family
                    ),
                    "contrast_type": (
                        contrast_type
                    ),
                    **metadata,
                    "previous_4vHPV_doses": (
                        dose
                    ),
                    "participants": int(
                        result["n"]
                    ),
                    "mean_log2_change": (
                        result["mean"]
                    ),
                    "sd_log2_change": (
                        result["sd"]
                    ),
                    "se_log2_change": (
                        result["se"]
                    ),
                    "ci95_lower_log2_change": (
                        result[
                            "ci95_lower"
                        ]
                    ),
                    "ci95_upper_log2_change": (
                        result[
                            "ci95_upper"
                        ]
                    ),
                    "geometric_mean_ratio": (
                        2 ** result["mean"]
                        if np.isfinite(
                            result["mean"]
                        )
                        else np.nan
                    ),
                    "gmr_ci95_lower": (
                        2 ** result[
                            "ci95_lower"
                        ]
                        if np.isfinite(
                            result[
                                "ci95_lower"
                            ]
                        )
                        else np.nan
                    ),
                    "gmr_ci95_upper": (
                        2 ** result[
                            "ci95_upper"
                        ]
                        if np.isfinite(
                            result[
                                "ci95_upper"
                            ]
                        )
                        else np.nan
                    ),
                    "test_statistic": (
                        result[
                            "test_statistic"
                        ]
                    ),
                    "degrees_freedom": (
                        result[
                            "degrees_freedom"
                        ]
                    ),
                    "p_value": (
                        result["p_value"]
                    ),
                    "cohen_dz": (
                        result["cohen_dz"]
                    ),
                    "sensitivity_requirement": (
                        sensitivity
                    ),
                    "bh_family": (
                        f"{model_id}|"
                        f"{outcome_family}|"
                        f"{contrast_type}"
                    ),
                }
            )

        # -----------------------------------------------------
        # Persistence and recall dose heterogeneity
        # -----------------------------------------------------

        previous = feature_group[
            feature_group[
                "previous_4vHPV_doses"
            ].isin(
                [1, 2, 3]
            )
        ]

        persistence_groups = [
            previous.loc[
                previous[
                    "previous_4vHPV_doses"
                ] == dose,
                "log2_v1_authoritative",
            ].to_numpy()
            for dose in [1, 2, 3]
        ]

        recall_groups = [
            previous.loc[
                previous[
                    "previous_4vHPV_doses"
                ] == dose,
                "log2_change_authoritative",
            ].to_numpy()
            for dose in [1, 2, 3]
        ]

        for (
            model_id,
            model_family,
            contrast_type,
            groups,
        ) in [
            (
                (
                    "HPVVT-FJ-M06"
                    if antigen == "BPV"
                    else "HPVVT-FJ-M02"
                ),
                (
                    "Heterologous BPV control"
                    if antigen == "BPV"
                    else "Six-year 4vHPV persistence"
                ),
                "global_persistence_dose_effect",
                persistence_groups,
            ),
            (
                (
                    "HPVVT-FJ-M06"
                    if antigen == "BPV"
                    else "HPVVT-FJ-M03"
                ),
                (
                    "Heterologous BPV control"
                    if antigen == "BPV"
                    else "Heterologous 2vHPV recall"
                ),
                "global_recall_dose_effect",
                recall_groups,
            ),
        ]:
            result = welch_anova(
                groups
            )

            global_rows.append(
                {
                    "model_id": model_id,
                    "model_family": (
                        model_family
                    ),
                    "contrast_type": (
                        contrast_type
                    ),
                    **metadata,
                    "dose_groups": "1;2;3",
                    "participants": int(
                        result["total_n"]
                    ),
                    "test_statistic": (
                        result[
                            "test_statistic"
                        ]
                    ),
                    "df_numerator": (
                        result[
                            "df_numerator"
                        ]
                    ),
                    "df_denominator": (
                        result[
                            "df_denominator"
                        ]
                    ),
                    "p_value": (
                        result["p_value"]
                    ),
                    "range_of_group_means_log2": (
                        result[
                            "range_of_group_means"
                        ]
                    ),
                    "sensitivity_requirement": (
                        sensitivity
                    ),
                    "bh_family": (
                        f"{model_id}|"
                        f"{outcome_family}|"
                        f"{contrast_type}"
                    ),
                }
            )

        # -----------------------------------------------------
        # Pairwise persistence and recall dose contrasts
        # -----------------------------------------------------

        for (
            higher_dose,
            lower_dose,
        ) in DOSE_PAIRS:
            high = previous[
                previous[
                    "previous_4vHPV_doses"
                ] == higher_dose
            ]

            low = previous[
                previous[
                    "previous_4vHPV_doses"
                ] == lower_dose
            ]

            for (
                model_id,
                model_family,
                contrast_type,
                value_column,
            ) in [
                (
                    (
                        "HPVVT-FJ-M06"
                        if antigen == "BPV"
                        else "HPVVT-FJ-M02"
                    ),
                    (
                        "Heterologous BPV control"
                        if antigen == "BPV"
                        else "Six-year 4vHPV persistence"
                    ),
                    "pairwise_persistence_dose_contrast",
                    "log2_v1_authoritative",
                ),
                (
                    (
                        "HPVVT-FJ-M06"
                        if antigen == "BPV"
                        else "HPVVT-FJ-M03"
                    ),
                    (
                        "Heterologous BPV control"
                        if antigen == "BPV"
                        else "Heterologous 2vHPV recall"
                    ),
                    "pairwise_recall_dose_contrast",
                    "log2_change_authoritative",
                ),
            ]:
                result = two_group_welch_test(
                    high[value_column],
                    low[value_column],
                )

                pairwise_rows.append(
                    {
                        "model_id": model_id,
                        "model_family": (
                            model_family
                        ),
                        "contrast_type": (
                            contrast_type
                        ),
                        **metadata,
                        "higher_dose": (
                            higher_dose
                        ),
                        "lower_dose": (
                            lower_dose
                        ),
                        "contrast_label": (
                            f"dose{higher_dose}"
                            f"_minus_"
                            f"dose{lower_dose}"
                        ),
                        "n_higher_dose": int(
                            result[
                                "n_first"
                            ]
                        ),
                        "n_lower_dose": int(
                            result[
                                "n_second"
                            ]
                        ),
                        "higher_dose_mean_log2": (
                            result[
                                "mean_first"
                            ]
                        ),
                        "lower_dose_mean_log2": (
                            result[
                                "mean_second"
                            ]
                        ),
                        "difference_log2": (
                            result[
                                "difference"
                            ]
                        ),
                        "ci95_lower_difference": (
                            result[
                                "ci95_lower"
                            ]
                        ),
                        "ci95_upper_difference": (
                            result[
                                "ci95_upper"
                            ]
                        ),
                        "ratio_of_geometric_means": (
                            2 ** result[
                                "difference"
                            ]
                            if np.isfinite(
                                result[
                                    "difference"
                                ]
                            )
                            else np.nan
                        ),
                        "ratio_ci95_lower": (
                            2 ** result[
                                "ci95_lower"
                            ]
                            if np.isfinite(
                                result[
                                    "ci95_lower"
                                ]
                            )
                            else np.nan
                        ),
                        "ratio_ci95_upper": (
                            2 ** result[
                                "ci95_upper"
                            ]
                            if np.isfinite(
                                result[
                                    "ci95_upper"
                                ]
                            )
                            else np.nan
                        ),
                        "test_statistic": (
                            result[
                                "test_statistic"
                            ]
                        ),
                        "degrees_freedom": (
                            result[
                                "degrees_freedom"
                            ]
                        ),
                        "p_value": (
                            result["p_value"]
                        ),
                        "hedges_g": (
                            result["hedges_g"]
                        ),
                        "sensitivity_requirement": (
                            sensitivity
                        ),
                        "bh_family": (
                            f"{model_id}|"
                            f"{outcome_family}|"
                            f"{contrast_type}"
                        ),
                    }
                )

        # -----------------------------------------------------
        # Primary versus pooled recall contrast
        # -----------------------------------------------------

        if antigen != "BPV":
            primary = feature_group.loc[
                feature_group[
                    "previous_4vHPV_doses"
                ] == 0,
                "log2_change_authoritative",
            ]

            recall = feature_group.loc[
                feature_group[
                    "previous_4vHPV_doses"
                ] > 0,
                "log2_change_authoritative",
            ]

            result = two_group_welch_test(
                primary,
                recall,
            )

            primary_recall_rows.append(
                {
                    "model_id": (
                        "HPVVT-FJ-M04"
                    ),
                    "model_family": (
                        "Primary-versus-recall contrast"
                    ),
                    "contrast_type": (
                        "primary_minus_pooled_recall"
                    ),
                    **metadata,
                    "primary_participants": int(
                        result[
                            "n_first"
                        ]
                    ),
                    "recall_participants": int(
                        result[
                            "n_second"
                        ]
                    ),
                    "primary_mean_log2_change": (
                        result[
                            "mean_first"
                        ]
                    ),
                    "recall_mean_log2_change": (
                        result[
                            "mean_second"
                        ]
                    ),
                    "difference_log2": (
                        result[
                            "difference"
                        ]
                    ),
                    "ci95_lower_difference": (
                        result[
                            "ci95_lower"
                        ]
                    ),
                    "ci95_upper_difference": (
                        result[
                            "ci95_upper"
                        ]
                    ),
                    "ratio_of_geometric_mean_ratios": (
                        2 ** result[
                            "difference"
                        ]
                        if np.isfinite(
                            result[
                                "difference"
                            ]
                        )
                        else np.nan
                    ),
                    "ratio_ci95_lower": (
                        2 ** result[
                            "ci95_lower"
                        ]
                        if np.isfinite(
                            result[
                                "ci95_lower"
                            ]
                        )
                        else np.nan
                    ),
                    "ratio_ci95_upper": (
                        2 ** result[
                            "ci95_upper"
                        ]
                        if np.isfinite(
                            result[
                                "ci95_upper"
                            ]
                        )
                        else np.nan
                    ),
                    "test_statistic": (
                        result[
                            "test_statistic"
                        ]
                    ),
                    "degrees_freedom": (
                        result[
                            "degrees_freedom"
                        ]
                    ),
                    "p_value": (
                        result["p_value"]
                    ),
                    "hedges_g": (
                        result["hedges_g"]
                    ),
                    "sensitivity_requirement": (
                        sensitivity
                    ),
                    "bh_family": (
                        "HPVVT-FJ-M04|"
                        f"{outcome_family}|"
                        "primary_minus_pooled_recall"
                    ),
                }
            )

    within = apply_bh(
        pd.DataFrame(
            within_rows
        )
    )

    global_tests = apply_bh(
        pd.DataFrame(
            global_rows
        )
    )

    pairwise = apply_bh(
        pd.DataFrame(
            pairwise_rows
        )
    )

    primary_recall = apply_bh(
        pd.DataFrame(
            primary_recall_rows
        )
    )

    expected_counts = {
        "within_rows": 368,
        "global_rows": 184,
        "pairwise_rows": 552,
        "primary_recall_rows": 81,
    }

    observed_counts = {
        "within_rows": len(
            within
        ),
        "global_rows": len(
            global_tests
        ),
        "pairwise_rows": len(
            pairwise
        ),
        "primary_recall_rows": len(
            primary_recall
        ),
    }

    validation_failures: list[str] = []

    for key, expected in (
        expected_counts.items()
    ):
        observed = observed_counts[
            key
        ]

        if observed != expected:
            validation_failures.append(
                f"{key}: expected "
                f"{expected}, observed "
                f"{observed}"
            )

    for name, frame in [
        ("within", within),
        ("global", global_tests),
        ("pairwise", pairwise),
        (
            "primary_recall",
            primary_recall,
        ),
    ]:
        if frame["p_value"].isna().any():
            validation_failures.append(
                f"{name}: missing P values"
            )

        if frame["q_value"].isna().any():
            validation_failures.append(
                f"{name}: missing q values"
            )

    significance_summary = pd.concat(
        [
            within.assign(
                result_table="within_trajectory"
            ),
            global_tests.assign(
                result_table="global_dose_tests"
            ),
            pairwise.assign(
                result_table="pairwise_dose_contrasts"
            ),
            primary_recall.assign(
                result_table="primary_vs_recall"
            ),
        ],
        ignore_index=True,
        sort=False,
    )

    significance_summary = (
        significance_summary.groupby(
            [
                "result_table",
                "model_id",
                "model_family",
                "outcome_family",
            ],
            dropna=False,
        )
        .agg(
            tests=(
                "p_value",
                "size",
            ),
            nominal_p_lt_0_05=(
                "p_value",
                lambda values: int(
                    (
                        values
                        < 0.05
                    ).sum()
                ),
            ),
            fdr_q_lt_0_05=(
                "q_value",
                lambda values: int(
                    (
                        values
                        < 0.05
                    ).sum()
                ),
            ),
            minimum_p_value=(
                "p_value",
                "min",
            ),
            minimum_q_value=(
                "q_value",
                "min",
            ),
        )
        .reset_index()
    )

    decision = (
        "READY_FOR_PHASE2B2_MIXED_MODEL_AND_FLOOR_SENSITIVITY"
        if not validation_failures
        else "PHASE2B1_REPAIR_REQUIRED"
    )

    decision_frame = pd.DataFrame(
        [
            {
                "decision": decision,
                "unique_participants": int(
                    paired[
                        "participant_id"
                    ].nunique()
                ),
                "paired_feature_records": (
                    len(paired)
                ),
                **observed_counts,
                "within_fdr_significant": int(
                    (
                        within[
                            "q_value"
                        ]
                        < 0.05
                    ).sum()
                ),
                "global_fdr_significant": int(
                    (
                        global_tests[
                            "q_value"
                        ]
                        < 0.05
                    ).sum()
                ),
                "pairwise_fdr_significant": int(
                    (
                        pairwise[
                            "q_value"
                        ]
                        < 0.05
                    ).sum()
                ),
                "primary_recall_fdr_significant": int(
                    (
                        primary_recall[
                            "q_value"
                        ]
                        < 0.05
                    ).sum()
                ),
                "validation_failures": (
                    "; ".join(
                        validation_failures
                    )
                ),
            }
        ]
    )

    write_tsv(
        within,
        TABLE_DIR
        / "phase2B1_fiji_within_trajectory_tests.tsv",
    )

    write_tsv(
        global_tests,
        TABLE_DIR
        / "phase2B1_fiji_global_dose_tests.tsv",
    )

    write_tsv(
        pairwise,
        TABLE_DIR
        / "phase2B1_fiji_pairwise_dose_contrasts.tsv",
    )

    write_tsv(
        primary_recall,
        TABLE_DIR
        / "phase2B1_fiji_primary_vs_recall_tests.tsv",
    )

    write_tsv(
        significance_summary,
        TABLE_DIR
        / "phase2B1_fiji_significance_summary.tsv",
    )

    write_tsv(
        decision_frame,
        TABLE_DIR
        / "phase2B1_fiji_core_inference_decision.tsv",
    )

    report_path = (
        REPORT_DIR
        / "phase2B1_fiji_core_inferential_contrasts_report.md"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as report:
        report.write(
            "# Phase 2B1 Fiji core inferential contrasts\n\n"
        )

        report.write("## Decision\n\n")
        report.write(
            f"**{decision}**\n\n"
        )

        report.write("## Analytical implementation\n\n")
        report.write(
            "Because the Fiji dataset contains exactly two complete "
            "measurements per participant, participant-paired log2 "
            "change removes each participant-specific intercept. "
            "The primary induction and recall tests are therefore "
            "implemented on paired log2 changes. This is the direct "
            "two-time-point analogue of modeling a visit effect with "
            "a participant-specific intercept.\n\n"
        )

        report.write("## Tests completed\n\n")
        report.write(
            f"- Within-trajectory primary, recall and BPV-control "
            f"tests: {len(within)}\n"
        )
        report.write(
            f"- Global persistence and recall dose tests: "
            f"{len(global_tests)}\n"
        )
        report.write(
            f"- Pairwise dose contrasts: {len(pairwise)}\n"
        )
        report.write(
            f"- Primary-versus-recall contrasts: "
            f"{len(primary_recall)}\n\n"
        )

        report.write("## Inferential methods\n\n")
        report.write(
            "- Within-participant changes: two-sided one-sample "
            "t tests of mean log2 change against zero.\n"
        )
        report.write(
            "- Three-group dose heterogeneity: Welch heteroscedastic "
            "one-way analysis.\n"
        )
        report.write(
            "- Pairwise dose comparisons: Welch two-group contrasts.\n"
        )
        report.write(
            "- Primary-versus-recall comparisons: Welch contrasts "
            "between the dose-0 primary group and pooled previously "
            "vaccinated participants.\n"
        )
        report.write(
            "- Multiplicity control: Benjamini–Hochberg adjustment "
            "within the model and biological outcome families locked "
            "in Phase 1F.\n\n"
        )

        report.write("## FDR-significant results\n\n")
        report.write(
            f"- Within-trajectory tests: "
            f"{int((within['q_value'] < 0.05).sum())}\n"
        )
        report.write(
            f"- Global dose tests: "
            f"{int((global_tests['q_value'] < 0.05).sum())}\n"
        )
        report.write(
            f"- Pairwise dose contrasts: "
            f"{int((pairwise['q_value'] < 0.05).sum())}\n"
        )
        report.write(
            f"- Primary-versus-recall contrasts: "
            f"{int((primary_recall['q_value'] < 0.05).sum())}\n\n"
        )

        report.write("## Interpretation boundary\n\n")
        report.write(
            "These results establish statistical evidence for "
            "systems-serology changes but do not directly measure "
            "intracellular trafficking, antigen processing or APC "
            "signaling. Those mechanisms will be addressed through "
            "integration with HPV cellular datasets and mechanistic "
            "transcriptomic comparators.\n\n"
        )

        report.write("## Next phase\n\n")
        report.write(
            "Phase 2B2 will fit long-format participant-random-intercept "
            "models as confirmation and implement the prespecified "
            "above-floor and two-part sensitivity analyses for "
            "moderate- and high-floor outcomes.\n"
        )

        if validation_failures:
            report.write("\n## Validation failures\n\n")

            for failure in validation_failures:
                report.write(
                    f"- {failure}\n"
                )

    print("===== PHASE 2B1 COMPLETE =====")
    print(f"Decision: {decision}")
    print(
        "Within-trajectory tests: "
        f"{len(within)}"
    )
    print(
        "Global dose tests: "
        f"{len(global_tests)}"
    )
    print(
        "Pairwise dose contrasts: "
        f"{len(pairwise)}"
    )
    print(
        "Primary-versus-recall tests: "
        f"{len(primary_recall)}"
    )
    print(
        "FDR-significant within tests: "
        f"{int((within['q_value'] < 0.05).sum())}"
    )
    print(
        "FDR-significant global tests: "
        f"{int((global_tests['q_value'] < 0.05).sum())}"
    )
    print(
        "FDR-significant pairwise tests: "
        f"{int((pairwise['q_value'] < 0.05).sum())}"
    )
    print(
        "FDR-significant primary-versus-recall tests: "
        f"{int((primary_recall['q_value'] < 0.05).sum())}"
    )
    print(f"Report: {report_path}")

    if validation_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
