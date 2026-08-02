#!/usr/bin/env python3

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Iterable

try:
    import numpy as np
    import pandas as pd
    from scipy.stats import t as student_t
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

LONG_INPUT = (
    PROCESSED_DIR
    / "phase1E_fiji_long_immunization_context_corrected.tsv"
)

PAIRED_INPUT = (
    PROCESSED_DIR
    / "phase1E_fiji_paired_immunization_context_corrected.tsv"
)

FLOOR_REGISTRY = (
    TABLE_DIR
    / "phase1F_fiji_floor_severity_registry.tsv"
)

OUTCOME_REGISTRY = (
    TABLE_DIR
    / "phase1F_fiji_outcome_registry.tsv"
)

ANALYSIS_READY_LONG = (
    PROCESSED_DIR
    / "phase2A_fiji_log2_long_analysis_ready.tsv"
)

ANALYSIS_READY_PAIRED = (
    PROCESSED_DIR
    / "phase2A_fiji_paired_effects_analysis_ready.tsv"
)

SEVERITY_ORDER = {
    "none": 0,
    "low": 1,
    "moderate": 2,
    "high": 3,
}

ANTIGEN_ORDER = [
    "HPV16",
    "HPV18",
    "HPV31",
    "HPV33",
    "HPV45",
    "HPV52",
    "HPV58",
    "BPV",
]

FEATURE_ORDER = [
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
    "ADCP",
    "nAb",
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


def mean_ci(
    values: Iterable[float],
    confidence: float = 0.95,
) -> tuple[int, float, float, float, float, float]:
    array = np.asarray(
        list(values),
        dtype=float,
    )

    array = array[
        np.isfinite(array)
    ]

    n = int(len(array))

    if n == 0:
        return (
            0,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    mean = float(
        np.mean(array)
    )

    if n == 1:
        return (
            n,
            mean,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    sd = float(
        np.std(
            array,
            ddof=1,
        )
    )

    se = sd / math.sqrt(n)

    critical = float(
        student_t.ppf(
            0.5 + confidence / 2,
            df=n - 1,
        )
    )

    lower = mean - critical * se
    upper = mean + critical * se

    return (
        n,
        mean,
        sd,
        se,
        lower,
        upper,
    )


def welch_difference_ci(
    first: Iterable[float],
    second: Iterable[float],
    confidence: float = 0.95,
) -> tuple[
    int,
    int,
    float,
    float,
    float,
    float,
]:
    a = np.asarray(
        list(first),
        dtype=float,
    )

    b = np.asarray(
        list(second),
        dtype=float,
    )

    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    n_a = int(len(a))
    n_b = int(len(b))

    if n_a == 0 or n_b == 0:
        return (
            n_a,
            n_b,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    difference = mean_a - mean_b

    if n_a < 2 or n_b < 2:
        return (
            n_a,
            n_b,
            difference,
            np.nan,
            np.nan,
            np.nan,
        )

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
        component_a
        + component_b
    )

    if se == 0:
        return (
            n_a,
            n_b,
            difference,
            0.0,
            difference,
            difference,
        )

    numerator = (
        component_a
        + component_b
    ) ** 2

    denominator = (
        (
            component_a ** 2
        )
        / (n_a - 1)
        + (
            component_b ** 2
        )
        / (n_b - 1)
    )

    degrees_freedom = (
        numerator / denominator
        if denominator > 0
        else n_a + n_b - 2
    )

    critical = float(
        student_t.ppf(
            0.5 + confidence / 2,
            df=degrees_freedom,
        )
    )

    lower = (
        difference
        - critical * se
    )

    upper = (
        difference
        + critical * se
    )

    return (
        n_a,
        n_b,
        difference,
        se,
        lower,
        upper,
    )


def maximum_severity(
    severities: Iterable[str],
) -> str:
    cleaned = [
        str(value)
        for value in severities
        if pd.notna(value)
        and str(value) in SEVERITY_ORDER
    ]

    if not cleaned:
        return "unresolved"

    return max(
        cleaned,
        key=lambda value: (
            SEVERITY_ORDER[value]
        ),
    )


def trajectory_label(
    previous_doses: int,
) -> str:
    if int(previous_doses) == 0:
        return (
            "primary_2vHPV_induction"
        )

    return (
        f"recall_after_"
        f"{int(previous_doses)}_previous_4vHPV_doses"
    )


def response_transition(
    previous_doses: int,
) -> str:
    if int(previous_doses) == 0:
        return (
            "unvaccinated_baseline_to_"
            "primary_2vHPV_response"
        )

    return (
        "six_year_4vHPV_persistence_to_"
        "heterologous_2vHPV_recall"
    )


def categorical_sort(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()

    if "antigen_target" in result.columns:
        result["antigen_target"] = pd.Categorical(
            result["antigen_target"],
            categories=ANTIGEN_ORDER,
            ordered=True,
        )

    if "feature" in result.columns:
        result["feature"] = pd.Categorical(
            result["feature"],
            categories=FEATURE_ORDER,
            ordered=True,
        )

    sorting_columns = [
        column
        for column in [
            "previous_4vHPV_doses",
            "visit",
            "antigen_target",
            "feature",
        ]
        if column in result.columns
    ]

    if sorting_columns:
        result = result.sort_values(
            sorting_columns
        )

    if "antigen_target" in result.columns:
        result["antigen_target"] = (
            result[
                "antigen_target"
            ].astype("string")
        )

    if "feature" in result.columns:
        result["feature"] = (
            result[
                "feature"
            ].astype("string")
        )

    return result.reset_index(
        drop=True
    )


def build_feature_to_outcome(
    registry: pd.DataFrame,
) -> dict[str, str]:
    mapping: dict[str, str] = {}

    for _, row in registry.iterrows():
        family = str(
            row["outcome_family"]
        )

        features = [
            value.strip()
            for value in str(
                row["features"]
            ).split(";")
            if value.strip()
        ]

        for feature in features:
            if (
                feature in mapping
                and mapping[feature]
                != family
            ):
                raise ValueError(
                    "Feature assigned to multiple "
                    f"outcome families: {feature}"
                )

            mapping[feature] = family

    return mapping


def main() -> None:
    required_files = [
        LONG_INPUT,
        PAIRED_INPUT,
        FLOOR_REGISTRY,
        OUTCOME_REGISTRY,
    ]

    for required_file in required_files:
        if not required_file.exists():
            sys.exit(
                "ERROR: Required input "
                f"missing: {required_file}"
            )

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    long_df = pd.read_csv(
        LONG_INPUT,
        sep="\t",
        dtype={
            "participant_id": "string",
            "previous_4vHPV_doses": "Int64",
            "visit": "string",
            "antigen_target": "string",
            "feature": "string",
        },
    )

    paired_df = pd.read_csv(
        PAIRED_INPUT,
        sep="\t",
        dtype={
            "participant_id": "string",
            "previous_4vHPV_doses": "Int64",
            "antigen_target": "string",
            "feature": "string",
        },
    )

    floor_df = pd.read_csv(
        FLOOR_REGISTRY,
        sep="\t",
        dtype={
            "antigen_target": "string",
            "visit": "string",
            "feature": "string",
            "floor_severity": "string",
        },
    )

    outcome_df = pd.read_csv(
        OUTCOME_REGISTRY,
        sep="\t",
    )

    feature_to_outcome = (
        build_feature_to_outcome(
            outcome_df
        )
    )

    required_long_columns = {
        "participant_id",
        "previous_4vHPV_doses",
        "visit",
        "antigen_target",
        "antigen_class",
        "feature",
        "assay_family",
        "value",
        "immunization_context",
        "response_type",
    }

    required_paired_columns = {
        "participant_id",
        "previous_4vHPV_doses",
        "antigen_target",
        "antigen_class",
        "feature",
        "assay_family",
        "v1",
        "v2",
        "paired_complete",
        "prior_4vHPV_exposure_status",
        "response_transition",
    }

    missing_long = (
        required_long_columns
        - set(long_df.columns)
    )

    missing_paired = (
        required_paired_columns
        - set(paired_df.columns)
    )

    if missing_long:
        sys.exit(
            "ERROR: Missing long-table "
            "columns: "
            + ", ".join(
                sorted(missing_long)
            )
        )

    if missing_paired:
        sys.exit(
            "ERROR: Missing paired-table "
            "columns: "
            + ", ".join(
                sorted(missing_paired)
            )
        )

    long_df["value_numeric"] = (
        pd.to_numeric(
            long_df["value"],
            errors="coerce",
        )
    )

    paired_df["v1_numeric"] = (
        pd.to_numeric(
            paired_df["v1"],
            errors="coerce",
        )
    )

    paired_df["v2_numeric"] = (
        pd.to_numeric(
            paired_df["v2"],
            errors="coerce",
        )
    )

    if long_df[
        "value_numeric"
    ].isna().any():
        sys.exit(
            "ERROR: Nonnumeric values "
            "remain in the long table."
        )

    if (
        paired_df[
            [
                "v1_numeric",
                "v2_numeric",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        sys.exit(
            "ERROR: Missing or nonnumeric "
            "paired values remain."
        )

    if (
        long_df["value_numeric"]
        <= 0
    ).any():
        sys.exit(
            "ERROR: Nonpositive values "
            "prevent raw log2 transformation."
        )

    if (
        paired_df["v1_numeric"]
        <= 0
    ).any() or (
        paired_df["v2_numeric"]
        <= 0
    ).any():
        sys.exit(
            "ERROR: Nonpositive paired "
            "values prevent log2 transformation."
        )

    long_df["log2_value"] = np.log2(
        long_df["value_numeric"]
    )

    paired_df["log2_v1_authoritative"] = (
        np.log2(
            paired_df["v1_numeric"]
        )
    )

    paired_df["log2_v2_authoritative"] = (
        np.log2(
            paired_df["v2_numeric"]
        )
    )

    paired_df[
        "log2_change_authoritative"
    ] = (
        paired_df[
            "log2_v2_authoritative"
        ]
        - paired_df[
            "log2_v1_authoritative"
        ]
    )

    paired_df[
        "geometric_mean_ratio_individual"
    ] = 2 ** paired_df[
        "log2_change_authoritative"
    ]

    long_df["outcome_family"] = (
        long_df["feature"]
        .map(feature_to_outcome)
    )

    paired_df["outcome_family"] = (
        paired_df["feature"]
        .map(feature_to_outcome)
    )

    unmapped_features = sorted(
        set(
            long_df.loc[
                long_df[
                    "outcome_family"
                ].isna(),
                "feature",
            ].astype(str)
        )
        | set(
            paired_df.loc[
                paired_df[
                    "outcome_family"
                ].isna(),
                "feature",
            ].astype(str)
        )
    )

    if unmapped_features:
        sys.exit(
            "ERROR: Unmapped outcome "
            "features: "
            + ", ".join(
                unmapped_features
            )
        )

    floor_columns = [
        "antigen_target",
        "visit",
        "feature",
        "minimum",
        "minimum_fraction",
        "floor_severity",
        "discrete_titer_or_ordinal_assay",
    ]

    floor_compact = (
        floor_df[
            floor_columns
        ]
        .drop_duplicates(
            [
                "antigen_target",
                "visit",
                "feature",
            ]
        )
    )

    long_df = long_df.merge(
        floor_compact,
        on=[
            "antigen_target",
            "visit",
            "feature",
        ],
        how="left",
        validate="many_to_one",
    )

    if long_df[
        "floor_severity"
    ].isna().any():
        unresolved = (
            long_df.loc[
                long_df[
                    "floor_severity"
                ].isna(),
                [
                    "antigen_target",
                    "visit",
                    "feature",
                ],
            ]
            .drop_duplicates()
        )

        sys.exit(
            "ERROR: Missing floor "
            "classification:\n"
            + unresolved.to_string(
                index=False
            )
        )

    floor_v1 = (
        floor_compact[
            floor_compact[
                "visit"
            ] == "v1"
        ]
        .drop(
            columns=["visit"]
        )
        .rename(
            columns={
                "minimum": "v1_floor_value",
                "minimum_fraction": (
                    "v1_floor_fraction"
                ),
                "floor_severity": (
                    "v1_floor_severity"
                ),
                "discrete_titer_or_ordinal_assay": (
                    "v1_discrete_titer_or_ordinal"
                ),
            }
        )
    )

    floor_v2 = (
        floor_compact[
            floor_compact[
                "visit"
            ] == "v2"
        ]
        .drop(
            columns=["visit"]
        )
        .rename(
            columns={
                "minimum": "v2_floor_value",
                "minimum_fraction": (
                    "v2_floor_fraction"
                ),
                "floor_severity": (
                    "v2_floor_severity"
                ),
                "discrete_titer_or_ordinal_assay": (
                    "v2_discrete_titer_or_ordinal"
                ),
            }
        )
    )

    paired_df = paired_df.merge(
        floor_v1,
        on=[
            "antigen_target",
            "feature",
        ],
        how="left",
        validate="many_to_one",
    )

    paired_df = paired_df.merge(
        floor_v2,
        on=[
            "antigen_target",
            "feature",
        ],
        how="left",
        validate="many_to_one",
    )

    paired_df[
        "paired_floor_severity"
    ] = [
        maximum_severity(
            [first, second]
        )
        for first, second in zip(
            paired_df[
                "v1_floor_severity"
            ],
            paired_df[
                "v2_floor_severity"
            ],
        )
    ]

    paired_df[
        "trajectory"
    ] = paired_df[
        "previous_4vHPV_doses"
    ].apply(trajectory_label)

    write_tsv(
        long_df,
        ANALYSIS_READY_LONG,
    )

    write_tsv(
        paired_df,
        ANALYSIS_READY_PAIRED,
    )

    # ---------------------------------------------------------
    # Visit-level descriptive landscape
    # ---------------------------------------------------------

    visit_group_columns = [
        "previous_4vHPV_doses",
        "visit",
        "prior_4vHPV_exposure_status",
        "immunization_context",
        "response_type",
        "antigen_target",
        "antigen_class",
        "feature",
        "assay_family",
        "outcome_family",
        "floor_severity",
        "minimum_fraction",
    ]

    visit_rows: list[
        dict[str, object]
    ] = []

    for keys, group in long_df.groupby(
        visit_group_columns,
        dropna=False,
        observed=True,
    ):
        key_map = dict(
            zip(
                visit_group_columns,
                keys,
            )
        )

        (
            n,
            mean_log2,
            sd_log2,
            se_log2,
            ci_lower,
            ci_upper,
        ) = mean_ci(
            group["log2_value"]
        )

        raw_values = group[
            "value_numeric"
        ]

        visit_rows.append(
            {
                **key_map,
                "participants": int(
                    group[
                        "participant_id"
                    ].nunique()
                ),
                "observations": n,
                "mean_log2_value": (
                    mean_log2
                ),
                "sd_log2_value": (
                    sd_log2
                ),
                "se_log2_value": (
                    se_log2
                ),
                "ci95_lower_log2": (
                    ci_lower
                ),
                "ci95_upper_log2": (
                    ci_upper
                ),
                "geometric_mean": (
                    2 ** mean_log2
                ),
                "geometric_mean_ci95_lower": (
                    2 ** ci_lower
                    if np.isfinite(
                        ci_lower
                    )
                    else np.nan
                ),
                "geometric_mean_ci95_upper": (
                    2 ** ci_upper
                    if np.isfinite(
                        ci_upper
                    )
                    else np.nan
                ),
                "raw_median": float(
                    raw_values.median()
                ),
                "raw_q1": float(
                    raw_values.quantile(
                        0.25
                    )
                ),
                "raw_q3": float(
                    raw_values.quantile(
                        0.75
                    )
                ),
                "raw_minimum": float(
                    raw_values.min()
                ),
                "raw_maximum": float(
                    raw_values.max()
                ),
            }
        )

    visit_summary = (
        categorical_sort(
            pd.DataFrame(
                visit_rows
            )
        )
    )

    # ---------------------------------------------------------
    # Paired primary/recall effect landscape
    # ---------------------------------------------------------

    paired_group_columns = [
        "previous_4vHPV_doses",
        "prior_4vHPV_exposure_status",
        "response_transition",
        "trajectory",
        "antigen_target",
        "antigen_class",
        "feature",
        "assay_family",
        "outcome_family",
        "paired_floor_severity",
        "v1_floor_fraction",
        "v2_floor_fraction",
    ]

    paired_rows: list[
        dict[str, object]
    ] = []

    for keys, group in paired_df.groupby(
        paired_group_columns,
        dropna=False,
        observed=True,
    ):
        key_map = dict(
            zip(
                paired_group_columns,
                keys,
            )
        )

        changes = group[
            "log2_change_authoritative"
        ]

        (
            n,
            mean_change,
            sd_change,
            se_change,
            ci_lower,
            ci_upper,
        ) = mean_ci(
            changes
        )

        paired_rows.append(
            {
                **key_map,
                "participants": int(
                    group[
                        "participant_id"
                    ].nunique()
                ),
                "paired_observations": n,
                "mean_log2_change": (
                    mean_change
                ),
                "sd_log2_change": (
                    sd_change
                ),
                "se_log2_change": (
                    se_change
                ),
                "ci95_lower_log2_change": (
                    ci_lower
                ),
                "ci95_upper_log2_change": (
                    ci_upper
                ),
                "geometric_mean_ratio": (
                    2 ** mean_change
                ),
                "geometric_mean_ratio_ci95_lower": (
                    2 ** ci_lower
                    if np.isfinite(
                        ci_lower
                    )
                    else np.nan
                ),
                "geometric_mean_ratio_ci95_upper": (
                    2 ** ci_upper
                    if np.isfinite(
                        ci_upper
                    )
                    else np.nan
                ),
                "median_log2_change": float(
                    changes.median()
                ),
                "q1_log2_change": float(
                    changes.quantile(
                        0.25
                    )
                ),
                "q3_log2_change": float(
                    changes.quantile(
                        0.75
                    )
                ),
                "positive_change_fraction": float(
                    (
                        changes > 0
                    ).mean()
                ),
                "negative_change_fraction": float(
                    (
                        changes < 0
                    ).mean()
                ),
                "no_change_fraction": float(
                    (
                        changes == 0
                    ).mean()
                ),
                "v1_geometric_mean": float(
                    2 ** group[
                        "log2_v1_authoritative"
                    ].mean()
                ),
                "v2_geometric_mean": float(
                    2 ** group[
                        "log2_v2_authoritative"
                    ].mean()
                ),
            }
        )

    paired_summary = (
        categorical_sort(
            pd.DataFrame(
                paired_rows
            )
        )
    )

    # ---------------------------------------------------------
    # Primary versus pooled recall descriptive contrast
    # ---------------------------------------------------------

    comparison_rows: list[
        dict[str, object]
    ] = []

    comparison_group_columns = [
        "antigen_target",
        "antigen_class",
        "feature",
        "assay_family",
        "outcome_family",
    ]

    for keys, group in paired_df.groupby(
        comparison_group_columns,
        dropna=False,
        observed=True,
    ):
        key_map = dict(
            zip(
                comparison_group_columns,
                keys,
            )
        )

        primary = group.loc[
            group[
                "previous_4vHPV_doses"
            ] == 0,
            "log2_change_authoritative",
        ]

        recall = group.loc[
            group[
                "previous_4vHPV_doses"
            ] > 0,
            "log2_change_authoritative",
        ]

        (
            n_primary,
            n_recall,
            difference,
            difference_se,
            difference_lower,
            difference_upper,
        ) = welch_difference_ci(
            primary,
            recall,
        )

        comparison_rows.append(
            {
                **key_map,
                "primary_participants": (
                    n_primary
                ),
                "recall_participants": (
                    n_recall
                ),
                "primary_mean_log2_change": float(
                    primary.mean()
                ),
                "recall_mean_log2_change": float(
                    recall.mean()
                ),
                "primary_geometric_mean_ratio": float(
                    2 ** primary.mean()
                ),
                "recall_geometric_mean_ratio": float(
                    2 ** recall.mean()
                ),
                "primary_minus_recall_log2_change": (
                    difference
                ),
                "difference_standard_error": (
                    difference_se
                ),
                "ci95_lower_primary_minus_recall": (
                    difference_lower
                ),
                "ci95_upper_primary_minus_recall": (
                    difference_upper
                ),
                "ratio_of_geometric_mean_ratios": (
                    2 ** difference
                ),
                "ratio_ci95_lower": (
                    2 ** difference_lower
                    if np.isfinite(
                        difference_lower
                    )
                    else np.nan
                ),
                "ratio_ci95_upper": (
                    2 ** difference_upper
                    if np.isfinite(
                        difference_upper
                    )
                    else np.nan
                ),
                "maximum_floor_severity": (
                    maximum_severity(
                        group[
                            "paired_floor_severity"
                        ]
                    )
                ),
            }
        )

    primary_recall_summary = (
        categorical_sort(
            pd.DataFrame(
                comparison_rows
            )
        )
    )

    # ---------------------------------------------------------
    # BPV-calibrated HPV-specific paired changes
    # ---------------------------------------------------------

    bpv_long = long_df.loc[
        long_df[
            "antigen_target"
        ] == "BPV",
        [
            "participant_id",
            "previous_4vHPV_doses",
            "visit",
            "feature",
            "log2_value",
            "floor_severity",
            "minimum_fraction",
        ],
    ].copy()

    bpv_long = bpv_long.rename(
        columns={
            "log2_value": (
                "bpv_log2_value"
            ),
            "floor_severity": (
                "bpv_floor_severity"
            ),
            "minimum_fraction": (
                "bpv_floor_fraction"
            ),
        }
    )

    hpv_long = long_df.loc[
        long_df[
            "antigen_target"
        ].str.startswith(
            "HPV"
        ),
        [
            "participant_id",
            "previous_4vHPV_doses",
            "visit",
            "prior_4vHPV_exposure_status",
            "immunization_context",
            "response_type",
            "antigen_target",
            "antigen_class",
            "feature",
            "assay_family",
            "outcome_family",
            "log2_value",
            "floor_severity",
            "minimum_fraction",
        ],
    ].copy()

    calibrated_long = hpv_long.merge(
        bpv_long,
        on=[
            "participant_id",
            "previous_4vHPV_doses",
            "visit",
            "feature",
        ],
        how="inner",
        validate="many_to_one",
    )

    calibrated_long[
        "hpv_minus_bpv_log2"
    ] = (
        calibrated_long[
            "log2_value"
        ]
        - calibrated_long[
            "bpv_log2_value"
        ]
    )

    calibrated_long[
        "combined_floor_severity"
    ] = [
        maximum_severity(
            [hpv, bpv]
        )
        for hpv, bpv in zip(
            calibrated_long[
                "floor_severity"
            ],
            calibrated_long[
                "bpv_floor_severity"
            ],
        )
    ]

    calibration_index = [
        "participant_id",
        "previous_4vHPV_doses",
        "prior_4vHPV_exposure_status",
        "antigen_target",
        "antigen_class",
        "feature",
        "assay_family",
        "outcome_family",
    ]

    calibrated_paired = (
        calibrated_long.pivot_table(
            index=calibration_index,
            columns="visit",
            values="hpv_minus_bpv_log2",
            aggfunc="first",
        )
        .reset_index()
    )

    calibrated_paired.columns.name = None

    calibrated_floor = (
        calibrated_long.groupby(
            calibration_index,
            dropna=False,
            observed=True,
        )
        .agg(
            calibrated_floor_severity=(
                "combined_floor_severity",
                maximum_severity,
            ),
            maximum_hpv_floor_fraction=(
                "minimum_fraction",
                "max",
            ),
            maximum_bpv_floor_fraction=(
                "bpv_floor_fraction",
                "max",
            ),
        )
        .reset_index()
    )

    calibrated_paired = (
        calibrated_paired.merge(
            calibrated_floor,
            on=calibration_index,
            how="left",
            validate="one_to_one",
        )
    )

    if "v1" not in calibrated_paired.columns:
        calibrated_paired["v1"] = np.nan

    if "v2" not in calibrated_paired.columns:
        calibrated_paired["v2"] = np.nan

    calibrated_paired[
        "bpv_calibrated_log2_change"
    ] = (
        calibrated_paired["v2"]
        - calibrated_paired["v1"]
    )

    calibrated_paired[
        "response_transition"
    ] = calibrated_paired[
        "previous_4vHPV_doses"
    ].apply(response_transition)

    calibrated_rows: list[
        dict[str, object]
    ] = []

    calibrated_group_columns = [
        "previous_4vHPV_doses",
        "prior_4vHPV_exposure_status",
        "response_transition",
        "antigen_target",
        "antigen_class",
        "feature",
        "assay_family",
        "outcome_family",
        "calibrated_floor_severity",
    ]

    for keys, group in calibrated_paired.groupby(
        calibrated_group_columns,
        dropna=False,
        observed=True,
    ):
        key_map = dict(
            zip(
                calibrated_group_columns,
                keys,
            )
        )

        changes = group[
            "bpv_calibrated_log2_change"
        ]

        (
            n,
            mean_change,
            sd_change,
            se_change,
            ci_lower,
            ci_upper,
        ) = mean_ci(
            changes
        )

        calibrated_rows.append(
            {
                **key_map,
                "participants": int(
                    group[
                        "participant_id"
                    ].nunique()
                ),
                "paired_observations": n,
                "mean_bpv_calibrated_log2_change": (
                    mean_change
                ),
                "sd_bpv_calibrated_log2_change": (
                    sd_change
                ),
                "se_bpv_calibrated_log2_change": (
                    se_change
                ),
                "ci95_lower_bpv_calibrated": (
                    ci_lower
                ),
                "ci95_upper_bpv_calibrated": (
                    ci_upper
                ),
                "bpv_calibrated_geometric_ratio": (
                    2 ** mean_change
                ),
                "bpv_calibrated_ratio_ci95_lower": (
                    2 ** ci_lower
                    if np.isfinite(
                        ci_lower
                    )
                    else np.nan
                ),
                "bpv_calibrated_ratio_ci95_upper": (
                    2 ** ci_upper
                    if np.isfinite(
                        ci_upper
                    )
                    else np.nan
                ),
                "positive_calibrated_change_fraction": float(
                    (
                        changes > 0
                    ).mean()
                ),
            }
        )

    calibrated_summary = (
        categorical_sort(
            pd.DataFrame(
                calibrated_rows
            )
        )
    )

    # ---------------------------------------------------------
    # Heatmap-ready matrices
    # ---------------------------------------------------------

    matrix_index = [
        "antigen_target",
        "antigen_class",
        "outcome_family",
        "assay_family",
        "feature",
    ]

    primary_matrix = (
        paired_summary.loc[
            paired_summary[
                "previous_4vHPV_doses"
            ] == 0
        ]
        .pivot_table(
            index=matrix_index,
            values="mean_log2_change",
            aggfunc="first",
        )
        .rename(
            columns={
                "mean_log2_change": (
                    "dose0_primary_mean_log2_change"
                )
            }
        )
        .reset_index()
    )

    persistence_matrix = (
        visit_summary.loc[
            (
                visit_summary[
                    "visit"
                ] == "v1"
            )
            & (
                visit_summary[
                    "previous_4vHPV_doses"
                ] > 0
            )
        ]
        .pivot_table(
            index=matrix_index,
            columns="previous_4vHPV_doses",
            values="mean_log2_value",
            aggfunc="first",
        )
        .rename(
            columns={
                1: "dose1_persistence_mean_log2",
                2: "dose2_persistence_mean_log2",
                3: "dose3_persistence_mean_log2",
            }
        )
        .reset_index()
    )

    recall_matrix = (
        paired_summary.loc[
            paired_summary[
                "previous_4vHPV_doses"
            ] > 0
        ]
        .pivot_table(
            index=matrix_index,
            columns="previous_4vHPV_doses",
            values="mean_log2_change",
            aggfunc="first",
        )
        .rename(
            columns={
                1: "dose1_recall_mean_log2_change",
                2: "dose2_recall_mean_log2_change",
                3: "dose3_recall_mean_log2_change",
            }
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # Headline effect table
    # ---------------------------------------------------------

    headline_source = (
        paired_summary.loc[
            paired_summary[
                "antigen_target"
            ] != "BPV"
        ]
        .copy()
    )

    headline_source[
        "absolute_mean_log2_change"
    ] = headline_source[
        "mean_log2_change"
    ].abs()

    headline_source[
        "floor_interpretation"
    ] = headline_source[
        "paired_floor_severity"
    ].map(
        {
            "none": (
                "standard_continuous_interpretation"
            ),
            "low": (
                "continuous_interpretation_with_floor_summary"
            ),
            "moderate": (
                "requires_above_floor_sensitivity_analysis"
            ),
            "high": (
                "requires_two_part_sensitivity_analysis"
            ),
        }
    )

    headline_rows: list[
        pd.DataFrame
    ] = []

    for trajectory, group in (
        headline_source.groupby(
            "trajectory",
            observed=True,
        )
    ):
        positive = (
            group.sort_values(
                "mean_log2_change",
                ascending=False,
            )
            .head(12)
            .copy()
        )

        positive["direction"] = (
            "positive"
        )

        negative = (
            group.sort_values(
                "mean_log2_change",
                ascending=True,
            )
            .head(12)
            .copy()
        )

        negative["direction"] = (
            "negative"
        )

        headline_rows.extend(
            [
                positive,
                negative,
            ]
        )

    headline_effects = pd.concat(
        headline_rows,
        ignore_index=True,
    )

    headline_columns = [
        "trajectory",
        "previous_4vHPV_doses",
        "direction",
        "antigen_target",
        "antigen_class",
        "feature",
        "assay_family",
        "outcome_family",
        "participants",
        "mean_log2_change",
        "ci95_lower_log2_change",
        "ci95_upper_log2_change",
        "geometric_mean_ratio",
        "positive_change_fraction",
        "paired_floor_severity",
        "floor_interpretation",
    ]

    headline_effects = (
        headline_effects[
            headline_columns
        ]
    )

    # ---------------------------------------------------------
    # Validation decision
    # ---------------------------------------------------------

    validation_failures: list[str] = []

    unique_participants = int(
        long_df[
            "participant_id"
        ].nunique()
    )

    if unique_participants != 80:
        validation_failures.append(
            "Expected 80 unique participants, "
            f"observed {unique_participants}."
        )

    if len(paired_df) != 7360:
        validation_failures.append(
            "Expected 7360 paired feature records, "
            f"observed {len(paired_df)}."
        )

    if long_df[
        "floor_severity"
    ].isna().any():
        validation_failures.append(
            "Missing visit-specific floor severity."
        )

    if paired_df[
        "paired_floor_severity"
    ].eq(
        "unresolved"
    ).any():
        validation_failures.append(
            "Unresolved paired floor severity."
        )

    if len(calibrated_summary) == 0:
        validation_failures.append(
            "No BPV-calibrated summaries generated."
        )

    decision = (
        "READY_FOR_PHASE2B_INFERENTIAL_MODELING"
        if not validation_failures
        else "PHASE2A_REPAIR_REQUIRED"
    )

    decision_frame = pd.DataFrame(
        [
            {
                "decision": decision,
                "unique_participants": (
                    unique_participants
                ),
                "long_observations": (
                    len(long_df)
                ),
                "paired_feature_records": (
                    len(paired_df)
                ),
                "visit_summary_rows": (
                    len(visit_summary)
                ),
                "paired_effect_summary_rows": (
                    len(paired_summary)
                ),
                "primary_recall_contrast_rows": (
                    len(
                        primary_recall_summary
                    )
                ),
                "bpv_calibrated_summary_rows": (
                    len(
                        calibrated_summary
                    )
                ),
                "headline_effect_rows": (
                    len(headline_effects)
                ),
                "validation_failures": (
                    "; ".join(
                        validation_failures
                    )
                ),
                "inferential_tests_performed": (
                    "no"
                ),
            }
        ]
    )

    # ---------------------------------------------------------
    # Write outputs
    # ---------------------------------------------------------

    write_tsv(
        visit_summary,
        TABLE_DIR
        / "phase2A_fiji_visit_descriptive_landscape.tsv",
    )

    write_tsv(
        paired_summary,
        TABLE_DIR
        / "phase2A_fiji_paired_effect_landscape.tsv",
    )

    write_tsv(
        primary_recall_summary,
        TABLE_DIR
        / "phase2A_fiji_primary_vs_recall_descriptive_contrasts.tsv",
    )

    write_tsv(
        calibrated_summary,
        TABLE_DIR
        / "phase2A_fiji_bpv_calibrated_effect_landscape.tsv",
    )

    write_tsv(
        primary_matrix,
        TABLE_DIR
        / "phase2A_fiji_primary_induction_matrix.tsv",
    )

    write_tsv(
        persistence_matrix,
        TABLE_DIR
        / "phase2A_fiji_persistence_matrix.tsv",
    )

    write_tsv(
        recall_matrix,
        TABLE_DIR
        / "phase2A_fiji_recall_matrix.tsv",
    )

    write_tsv(
        headline_effects,
        TABLE_DIR
        / "phase2A_fiji_headline_effects.tsv",
    )

    write_tsv(
        decision_frame,
        TABLE_DIR
        / "phase2A_fiji_descriptive_landscape_decision.tsv",
    )

    report_path = (
        REPORT_DIR
        / "phase2A_fiji_descriptive_immune_landscape_report.md"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as report:
        report.write(
            "# Phase 2A Fiji descriptive immune landscape\n\n"
        )

        report.write("## Decision\n\n")
        report.write(
            f"**{decision}**\n\n"
        )

        report.write("## Analytical scope\n\n")
        report.write(
            "This phase reconstructs the systems-serology "
            "landscape before inferential mixed-effects modeling. "
            "It summarizes original visit states, participant-paired "
            "log2 changes, geometric-mean ratios, primary-versus-recall "
            "effect differences and BPV-calibrated HPV-specific changes.\n\n"
        )

        report.write(
            "No hypothesis-test P values or false-discovery-rate "
            "decisions are generated in this phase. Confidence intervals "
            "describe effect-size precision and do not replace the "
            "prespecified Phase 2B models.\n\n"
        )

        report.write("## Validated structure\n\n")
        report.write(
            f"- Unique participants: {unique_participants}\n"
        )
        report.write(
            f"- Long feature observations: {len(long_df)}\n"
        )
        report.write(
            f"- Complete paired feature records: {len(paired_df)}\n"
        )
        report.write(
            f"- Visit-level descriptive rows: {len(visit_summary)}\n"
        )
        report.write(
            f"- Dose–antigen–feature paired-effect rows: "
            f"{len(paired_summary)}\n"
        )
        report.write(
            f"- Primary-versus-recall descriptive contrasts: "
            f"{len(primary_recall_summary)}\n"
        )
        report.write(
            f"- BPV-calibrated effect rows: "
            f"{len(calibrated_summary)}\n\n"
        )

        report.write("## Authoritative effect scale\n\n")
        report.write(
            "All continuous assay values are analyzed as "
            "`log2(raw positive value)`. The paired response is:\n\n"
        )
        report.write(
            "`log2(Visit 2) - log2(Visit 1)`\n\n"
        )
        report.write(
            "The corresponding geometric-mean ratio is "
            "`2^(mean log2 change)`. Ratios above one indicate "
            "higher Visit 2 responses; ratios below one indicate "
            "lower Visit 2 responses.\n\n"
        )

        report.write("## Biological trajectories\n\n")
        report.write(
            "- Dose 0: unvaccinated baseline to primary 2vHPV "
            "induction.\n"
        )
        report.write(
            "- Doses 1–3: six-year 4vHPV persistence to "
            "heterologous 2vHPV recall.\n"
        )
        report.write(
            "- Cross-reactive HPV31/33/45/52/58 responses are "
            "retained separately from HPV16/18 vaccine-target "
            "responses.\n"
        )
        report.write(
            "- BPV-calibrated effects estimate HPV-specific "
            "changes after subtracting contemporaneous "
            "heterologous-control movement on the log2 scale.\n\n"
        )

        report.write("## Generated matrices\n\n")
        report.write(
            "- Primary-induction effect matrix for dose 0.\n"
        )
        report.write(
            "- Six-year persistence matrix for previous "
            "dose groups 1–3.\n"
        )
        report.write(
            "- Heterologous recall matrix for previous "
            "dose groups 1–3.\n"
        )
        report.write(
            "- Primary-versus-recall descriptive contrast table.\n"
        )
        report.write(
            "- BPV-calibrated HPV-specific change table.\n\n"
        )

        report.write("## Assay-floor interpretation\n\n")
        report.write(
            "Every effect carries the maximum floor severity "
            "observed across its Visit 1 and Visit 2 distributions. "
            "Moderate-floor results require an above-floor "
            "sensitivity model, and high-floor results require the "
            "two-part modeling strategy locked in Phase 1F.\n\n"
        )

        if validation_failures:
            report.write("## Validation failures\n\n")

            for failure in validation_failures:
                report.write(
                    f"- {failure}\n"
                )

            report.write("\n")

        report.write("## Next phase\n\n")
        report.write(
            "Phase 2B will fit the prespecified inferential models "
            "for primary induction, six-year persistence, "
            "heterologous recall, primary-versus-recall contrasts, "
            "cross-reactive breadth, BPV controls and antibody "
            "functional coupling. Benjamini–Hochberg adjustment will "
            "be applied within the biological families locked in "
            "Phase 1F.\n"
        )

    print("===== PHASE 2A COMPLETE =====")
    print(f"Decision: {decision}")
    print(
        "Unique participants: "
        f"{unique_participants}"
    )
    print(
        "Visit summary rows: "
        f"{len(visit_summary)}"
    )
    print(
        "Paired effect rows: "
        f"{len(paired_summary)}"
    )
    print(
        "Primary-versus-recall rows: "
        f"{len(primary_recall_summary)}"
    )
    print(
        "BPV-calibrated rows: "
        f"{len(calibrated_summary)}"
    )
    print(
        "Headline effect rows: "
        f"{len(headline_effects)}"
    )
    print(f"Report: {report_path}")

    if validation_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
