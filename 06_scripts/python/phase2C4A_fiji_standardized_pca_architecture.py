#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import (
    kruskal,
    mannwhitneyu,
    ttest_ind,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


ROOT = Path(
    "/mnt/d/HPV_Vaccine_Trafficome_Project"
)

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

RAW_INPUT = (
    PROCESSED
    / "phase2C1_fiji_raw_log2_change_matrix.tsv"
)

BPV_INPUT = (
    PROCESSED
    / "phase2C1_fiji_bpv_calibrated_log2_change_matrix.tsv"
)

METADATA_INPUT = (
    PROCESSED
    / "phase2C1_fiji_participant_metadata.tsv"
)

C3C_DECISION_INPUT = (
    TABLES
    / "phase2C3C_fiji_functional_coupling_synthesis_decision.tsv"
)

EXPECTED_C3C_DECISION = (
    "READY_FOR_PHASE2C3_COMMIT_AND_PHASE2C4_IMMUNE_STATE_STRUCTURE"
)

REPRESENTATIONS = {
    "raw_log2_change": RAW_INPUT,
    "bpv_calibrated_log2_change": BPV_INPUT,
}

ANTIGENS = [
    "HPV16",
    "HPV18",
    "HPV31",
    "HPV33",
    "HPV45",
    "HPV52",
    "HPV58",
    "BPV",
]

TOP_COMPONENTS = 10

CENTROID_STRATA = [
    "primary_dose0",
    "recall_all_doses",
    "recall_dose1",
    "recall_dose2",
    "recall_dose3",
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


def parse_feature_name(
    column: str,
) -> tuple[str, str]:
    text = str(
        column
    )

    separators = [
        "__",
        "::",
        "|",
        "_",
        "-",
        ".",
    ]

    for antigen in ANTIGENS:
        if text == antigen:
            return (
                antigen,
                "unresolved",
            )

        if text.startswith(
            antigen
        ):
            remainder = text[
                len(antigen):
            ]

            for separator in separators:
                remainder = remainder.lstrip(
                    separator
                )

            if remainder:
                return (
                    antigen,
                    remainder,
                )

        if text.endswith(
            antigen
        ):
            remainder = text[
                : -len(antigen)
            ]

            for separator in separators:
                remainder = remainder.rstrip(
                    separator
                )

            if remainder:
                return (
                    antigen,
                    remainder,
                )

    for separator in separators:
        if separator in text:
            parts = [
                part
                for part in text.split(
                    separator
                )
                if part
            ]

            if len(parts) >= 2:
                for part in parts:
                    if part in ANTIGENS:
                        other = [
                            value
                            for value in parts
                            if value != part
                        ]

                        return (
                            part,
                            separator.join(
                                other
                            ),
                        )

    return (
        "unresolved",
        text,
    )


def feature_family(
    assay_feature: str,
) -> str:
    feature = str(
        assay_feature
    )

    if feature in {
        "IgG",
        "IgM",
        "IgA1",
        "IgA2",
    }:
        return (
            "binding_antibody_abundance"
        )

    if feature in {
        "IgG1",
        "IgG2",
        "IgG3",
        "IgG4",
    }:
        return (
            "igg_subclass_architecture"
        )

    if feature in {
        "FcgR2A",
        "FcgR2B",
        "FcgR3A",
    }:
        return (
            "fc_receptor_communication"
        )

    if feature == "ADCP":
        return "phagocytic_function"

    if feature == "nAb":
        return "neutralization"

    return "unresolved"


def stratum_mask(
    frame: pd.DataFrame,
    stratum: str,
) -> pd.Series:
    doses = pd.to_numeric(
        frame[
            "previous_4vHPV_doses"
        ],
        errors="coerce",
    )

    if stratum == "primary_dose0":
        return doses == 0

    if stratum == "recall_all_doses":
        return doses > 0

    if stratum == "recall_dose1":
        return doses == 1

    if stratum == "recall_dose2":
        return doses == 2

    if stratum == "recall_dose3":
        return doses == 3

    raise ValueError(
        f"Unknown stratum: {stratum}"
    )


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

    n_b = len(
        group_b
    )

    n_a = len(
        group_a
    )

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
            n_b
            - 1
        )
        * variance_b
        + (
            n_a
            - 1
        )
        * variance_a
    ) / denominator_df

    if (
        not np.isfinite(
            pooled_variance
        )
        or pooled_variance <= 0
    ):
        return np.nan

    cohens_d = (
        np.mean(
            group_b
        )
        - np.mean(
            group_a
        )
    ) / np.sqrt(
        pooled_variance
    )

    correction = (
        1
        - 3
        / (
            4
            * denominator_df
            - 1
        )
    )

    return float(
        correction
        * cohens_d
    )


def add_fdr(
    frame: pd.DataFrame,
    group_columns: list[str],
    p_column: str,
) -> pd.DataFrame:
    output = frame.copy()

    output[
        "bh_q_value"
    ] = np.nan

    groups = output.groupby(
        group_columns,
        observed=True,
        dropna=False,
    ).groups

    for _, indices in groups.items():
        index_list = list(
            indices
        )

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
            p_values.loc[
                valid
            ],
            method="fdr_bh",
        )[1]

        output.loc[
            p_values.loc[
                valid
            ].index,
            "bh_q_value",
        ] = adjusted

    output[
        "fdr_significant"
    ] = (
        output[
            "bh_q_value"
        ]
        < 0.05
    )

    return output


def primary_vs_recall_tests(
    score_wide: pd.DataFrame,
    representation: str,
) -> pd.DataFrame:
    rows: list[
        dict[str, object]
    ] = []

    primary = score_wide[
        pd.to_numeric(
            score_wide[
                "previous_4vHPV_doses"
            ],
            errors="coerce",
        )
        == 0
    ]

    recall = score_wide[
        pd.to_numeric(
            score_wide[
                "previous_4vHPV_doses"
            ],
            errors="coerce",
        )
        > 0
    ]

    for component_number in range(
        1,
        TOP_COMPONENTS + 1,
    ):
        component = (
            f"PC{component_number}"
        )

        group_a = pd.to_numeric(
            primary[
                component
            ],
            errors="coerce",
        ).dropna().to_numpy(
            dtype=float
        )

        group_b = pd.to_numeric(
            recall[
                component
            ],
            errors="coerce",
        ).dropna().to_numpy(
            dtype=float
        )

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
            2
            * float(
                mann.statistic
            )
            / (
                len(group_b)
                * len(group_a)
            )
            - 1
        )

        rows.append(
            {
                "matrix_representation": (
                    representation
                ),
                "principal_component": component,
                "component_number": (
                    component_number
                ),
                "primary_participants": len(
                    group_a
                ),
                "recall_participants": len(
                    group_b
                ),
                "primary_mean": float(
                    np.mean(
                        group_a
                    )
                ),
                "recall_mean": float(
                    np.mean(
                        group_b
                    )
                ),
                "mean_difference_recall_minus_primary": float(
                    np.mean(
                        group_b
                    )
                    - np.mean(
                        group_a
                    )
                ),
                "primary_median": float(
                    np.median(
                        group_a
                    )
                ),
                "recall_median": float(
                    np.median(
                        group_b
                    )
                ),
                "median_difference_recall_minus_primary": float(
                    np.median(
                        group_b
                    )
                    - np.median(
                        group_a
                    )
                ),
                "mann_whitney_u": float(
                    mann.statistic
                ),
                "mann_whitney_p_value": float(
                    mann.pvalue
                ),
                "rank_biserial_recall_minus_primary": float(
                    rank_biserial
                ),
                "welch_t_statistic": float(
                    welch.statistic
                ),
                "welch_p_value": float(
                    welch.pvalue
                ),
                "hedges_g_recall_minus_primary": (
                    hedges_g(
                        group_b,
                        group_a,
                    )
                ),
            }
        )

    tests = pd.DataFrame(
        rows
    )

    return add_fdr(
        tests,
        group_columns=[
            "matrix_representation",
        ],
        p_column=(
            "mann_whitney_p_value"
        ),
    )


def recall_dose_global_tests(
    score_wide: pd.DataFrame,
    representation: str,
) -> pd.DataFrame:
    rows: list[
        dict[str, object]
    ] = []

    recall = score_wide[
        pd.to_numeric(
            score_wide[
                "previous_4vHPV_doses"
            ],
            errors="coerce",
        )
        > 0
    ]

    for component_number in range(
        1,
        TOP_COMPONENTS + 1,
    ):
        component = (
            f"PC{component_number}"
        )

        groups = [
            pd.to_numeric(
                recall.loc[
                    pd.to_numeric(
                        recall[
                            "previous_4vHPV_doses"
                        ],
                        errors="coerce",
                    )
                    == dose,
                    component,
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

        result = kruskal(
            *groups,
            nan_policy="omit",
        )

        total_n = sum(
            len(
                group
            )
            for group in groups
        )

        group_count = 3

        epsilon_squared = (
            (
                float(
                    result.statistic
                )
                - group_count
                + 1
            )
            / (
                total_n
                - group_count
            )
            if total_n
            > group_count
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
                "matrix_representation": (
                    representation
                ),
                "principal_component": component,
                "component_number": (
                    component_number
                ),
                "dose1_participants": len(
                    groups[0]
                ),
                "dose2_participants": len(
                    groups[1]
                ),
                "dose3_participants": len(
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
                "kruskal_wallis_h": float(
                    result.statistic
                ),
                "kruskal_wallis_p_value": float(
                    result.pvalue
                ),
                "epsilon_squared": (
                    epsilon_squared
                ),
            }
        )

    tests = pd.DataFrame(
        rows
    )

    return add_fdr(
        tests,
        group_columns=[
            "matrix_representation",
        ],
        p_column=(
            "kruskal_wallis_p_value"
        ),
    )


def fit_representation(
    representation: str,
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    feature_columns = [
        column
        for column in matrix.columns
        if column
        != "participant_id"
    ]

    numeric = matrix[
        feature_columns
    ].apply(
        pd.to_numeric,
        errors="raise",
    )

    scaler = StandardScaler(
        with_mean=True,
        with_std=True,
    )

    standardized = scaler.fit_transform(
        numeric.to_numpy(
            dtype=float
        )
    )

    n_components = min(
        len(
            matrix
        )
        - 1,
        len(
            feature_columns
        ),
    )

    pca = PCA(
        n_components=n_components,
        svd_solver="full",
    )

    score_array = pca.fit_transform(
        standardized
    )

    component_names = [
        f"PC{index}"
        for index in range(
            1,
            n_components + 1,
        )
    ]

    score_wide = pd.DataFrame(
        score_array,
        columns=component_names,
    )

    score_wide.insert(
        0,
        "participant_id",
        matrix[
            "participant_id"
        ].astype(
            "string"
        ).to_numpy(),
    )

    score_wide = score_wide.merge(
        metadata,
        on="participant_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    score_wide[
        "matrix_representation"
    ] = representation

    score_long = score_wide.melt(
        id_vars=[
            "participant_id",
            "previous_4vHPV_doses",
            "prior_4vHPV_exposure_status",
            "trajectory",
            "v1_biological_context",
            "v2_biological_context",
            "analysis_context",
            "matrix_representation",
        ],
        value_vars=component_names,
        var_name="principal_component",
        value_name="principal_component_score",
    )

    score_long[
        "component_number"
    ] = (
        score_long[
            "principal_component"
        ]
        .str.replace(
            "PC",
            "",
            regex=False,
        )
        .astype(int)
    )

    cumulative = np.cumsum(
        pca.explained_variance_ratio_
    )

    variance = pd.DataFrame(
        {
            "matrix_representation": (
                representation
            ),
            "principal_component": (
                component_names
            ),
            "component_number": np.arange(
                1,
                n_components + 1,
            ),
            "explained_variance": (
                pca.explained_variance_
            ),
            "explained_variance_ratio": (
                pca.explained_variance_ratio_
            ),
            "explained_variance_percent": (
                100
                * pca.explained_variance_ratio_
            ),
            "cumulative_variance_ratio": (
                cumulative
            ),
            "cumulative_variance_percent": (
                100
                * cumulative
            ),
        }
    )

    annotation_rows = []

    for column in feature_columns:
        antigen, assay_feature = (
            parse_feature_name(
                column
            )
        )

        annotation_rows.append(
            {
                "feature_column": column,
                "antigen_target": antigen,
                "assay_feature": assay_feature,
                "feature_family": (
                    feature_family(
                        assay_feature
                    )
                ),
            }
        )

    annotation = pd.DataFrame(
        annotation_rows
    )

    loading_rows: list[
        dict[str, object]
    ] = []

    for component_index in range(
        TOP_COMPONENTS
    ):
        component = (
            f"PC{component_index + 1}"
        )

        loadings = pca.components_[
            component_index,
            :
        ]

        for feature_index, column in enumerate(
            feature_columns
        ):
            loading = float(
                loadings[
                    feature_index
                ]
            )

            loading_rows.append(
                {
                    "matrix_representation": (
                        representation
                    ),
                    "principal_component": component,
                    "component_number": (
                        component_index
                        + 1
                    ),
                    "feature_column": column,
                    "loading": loading,
                    "absolute_loading": abs(
                        loading
                    ),
                    "squared_loading": (
                        loading ** 2
                    ),
                    "contribution_percent": (
                        100
                        * loading ** 2
                    ),
                }
            )

    loadings = pd.DataFrame(
        loading_rows
    ).merge(
        annotation,
        on="feature_column",
        how="left",
        validate="many_to_one",
    )

    extreme_rows: list[
        dict[str, object]
    ] = []

    for component, subset in loadings.groupby(
        "principal_component",
        observed=True,
    ):
        component_number = int(
            subset[
                "component_number"
            ].iloc[0]
        )

        highest = subset.sort_values(
            "loading",
            ascending=False,
        ).head(10)

        lowest = subset.sort_values(
            "loading",
            ascending=True,
        ).head(10)

        for direction, selected in [
            (
                "highest_loading",
                highest,
            ),
            (
                "lowest_loading",
                lowest,
            ),
        ]:
            for rank, row in enumerate(
                selected.itertuples(
                    index=False
                ),
                start=1,
            ):
                extreme_rows.append(
                    {
                        "matrix_representation": (
                            representation
                        ),
                        "principal_component": (
                            component
                        ),
                        "component_number": (
                            component_number
                        ),
                        "loading_direction": (
                            direction
                        ),
                        "loading_rank": rank,
                        "feature_column": (
                            row.feature_column
                        ),
                        "antigen_target": (
                            row.antigen_target
                        ),
                        "assay_feature": (
                            row.assay_feature
                        ),
                        "feature_family": (
                            row.feature_family
                        ),
                        "loading": (
                            row.loading
                        ),
                        "absolute_loading": (
                            row.absolute_loading
                        ),
                        "contribution_percent": (
                            row.contribution_percent
                        ),
                    }
                )

    extremes = pd.DataFrame(
        extreme_rows
    )

    contribution_rows: list[
        dict[str, object]
    ] = []

    for component, subset in loadings.groupby(
        "principal_component",
        observed=True,
    ):
        component_number = int(
            subset[
                "component_number"
            ].iloc[0]
        )

        for annotation_level, column in [
            (
                "antigen_target",
                "antigen_target",
            ),
            (
                "feature_family",
                "feature_family",
            ),
        ]:
            grouped = (
                subset.groupby(
                    column,
                    observed=True,
                    dropna=False,
                )[
                    "contribution_percent"
                ]
                .sum()
                .reset_index()
            )

            for row in grouped.itertuples(
                index=False
            ):
                contribution_rows.append(
                    {
                        "matrix_representation": (
                            representation
                        ),
                        "principal_component": (
                            component
                        ),
                        "component_number": (
                            component_number
                        ),
                        "annotation_level": (
                            annotation_level
                        ),
                        "annotation_value": (
                            getattr(
                                row,
                                column,
                            )
                        ),
                        "contribution_percent": (
                            row.contribution_percent
                        ),
                    }
                )

    contributions = pd.DataFrame(
        contribution_rows
    )

    centroid_rows: list[
        dict[str, object]
    ] = []

    for stratum in CENTROID_STRATA:
        subset = score_wide.loc[
            stratum_mask(
                score_wide,
                stratum,
            )
        ]

        for component_number in range(
            1,
            TOP_COMPONENTS + 1,
        ):
            component = (
                f"PC{component_number}"
            )

            centroid_rows.append(
                {
                    "matrix_representation": (
                        representation
                    ),
                    "analysis_stratum": (
                        stratum
                    ),
                    "participants": len(
                        subset
                    ),
                    "principal_component": (
                        component
                    ),
                    "component_number": (
                        component_number
                    ),
                    "centroid_score": float(
                        pd.to_numeric(
                            subset[
                                component
                            ],
                            errors="coerce",
                        ).mean()
                    ),
                    "median_score": float(
                        pd.to_numeric(
                            subset[
                                component
                            ],
                            errors="coerce",
                        ).median()
                    ),
                    "score_standard_deviation": float(
                        pd.to_numeric(
                            subset[
                                component
                            ],
                            errors="coerce",
                        ).std(
                            ddof=1
                        )
                    ),
                }
            )

    centroids = pd.DataFrame(
        centroid_rows
    )

    context_tests = (
        primary_vs_recall_tests(
            score_wide,
            representation,
        )
    )

    dose_tests = (
        recall_dose_global_tests(
            score_wide,
            representation,
        )
    )

    return {
        "score_wide": score_wide,
        "score_long": score_long,
        "variance": variance,
        "loadings": loadings,
        "extremes": extremes,
        "contributions": contributions,
        "centroids": centroids,
        "context_tests": context_tests,
        "dose_tests": dose_tests,
    }


def main() -> None:
    for path in [
        RAW_INPUT,
        BPV_INPUT,
        METADATA_INPUT,
        C3C_DECISION_INPUT,
    ]:
        if not path.exists():
            sys.exit(
                f"ERROR: Required input missing: {path}"
            )

    decision = pd.read_csv(
        C3C_DECISION_INPUT,
        sep="\t",
    )

    observed_decision = str(
        decision.loc[
            0,
            "decision",
        ]
    )

    if observed_decision != EXPECTED_C3C_DECISION:
        sys.exit(
            "ERROR: Phase 2C3C decision is "
            f"{observed_decision}; expected "
            f"{EXPECTED_C3C_DECISION}."
        )

    metadata = pd.read_csv(
        METADATA_INPUT,
        sep="\t",
        dtype={
            "participant_id": "string",
        },
    )

    require_columns(
        metadata,
        {
            "participant_id",
            "previous_4vHPV_doses",
            "prior_4vHPV_exposure_status",
            "trajectory",
            "v1_biological_context",
            "v2_biological_context",
            "analysis_context",
        },
        "Participant metadata",
    )

    if len(metadata) != 80:
        sys.exit(
            "ERROR: Expected 80 participant metadata rows."
        )

    if metadata[
        "participant_id"
    ].duplicated().any():
        sys.exit(
            "ERROR: Duplicate participant IDs in metadata."
        )

    outputs = []

    failures: list[str] = []

    for representation, path in REPRESENTATIONS.items():
        matrix = pd.read_csv(
            path,
            sep="\t",
            dtype={
                "participant_id": "string",
            },
        )

        require_columns(
            matrix,
            {
                "participant_id",
            },
            f"{representation} matrix",
        )

        if len(matrix) != 80:
            failures.append(
                f"{representation}: expected 80 rows, "
                f"observed {len(matrix)}."
            )

        if matrix[
            "participant_id"
        ].duplicated().any():
            failures.append(
                f"{representation}: duplicate participant IDs."
            )

        if (
            matrix[
                "participant_id"
            ].astype(str).tolist()
            != metadata[
                "participant_id"
            ].astype(str).tolist()
        ):
            failures.append(
                f"{representation}: participant order "
                "does not match metadata."
            )

        numeric = matrix.drop(
            columns=[
                "participant_id",
            ]
        ).apply(
            pd.to_numeric,
            errors="coerce",
        )

        missing_values = int(
            numeric.isna()
            .sum()
            .sum()
        )

        if missing_values:
            failures.append(
                f"{representation}: {missing_values} "
                "missing numeric values."
            )

        zero_variance = int(
            (
                numeric.std(
                    axis=0,
                    ddof=1,
                )
                <= 1e-15
            ).sum()
        )

        if zero_variance:
            failures.append(
                f"{representation}: {zero_variance} "
                "zero-variance features."
            )

        outputs.append(
            fit_representation(
                representation,
                matrix,
                metadata,
            )
        )

    variance = pd.concat(
        [
            output["variance"]
            for output in outputs
        ],
        ignore_index=True,
    )

    score_long = pd.concat(
        [
            output["score_long"]
            for output in outputs
        ],
        ignore_index=True,
    )

    loadings = pd.concat(
        [
            output["loadings"]
            for output in outputs
        ],
        ignore_index=True,
    )

    extremes = pd.concat(
        [
            output["extremes"]
            for output in outputs
        ],
        ignore_index=True,
    )

    contributions = pd.concat(
        [
            output["contributions"]
            for output in outputs
        ],
        ignore_index=True,
    )

    centroids = pd.concat(
        [
            output["centroids"]
            for output in outputs
        ],
        ignore_index=True,
    )

    context_tests = pd.concat(
        [
            output["context_tests"]
            for output in outputs
        ],
        ignore_index=True,
    )

    dose_tests = pd.concat(
        [
            output["dose_tests"]
            for output in outputs
        ],
        ignore_index=True,
    )

    expected_counts = {
        "variance": 156,
        "score_long": 12480,
        "loadings": 1690,
        "extremes": 400,
        "contributions": 230,
        "centroids": 100,
        "context_tests": 20,
        "dose_tests": 20,
    }

    observed_counts = {
        "variance": len(
            variance
        ),
        "score_long": len(
            score_long
        ),
        "loadings": len(
            loadings
        ),
        "extremes": len(
            extremes
        ),
        "contributions": len(
            contributions
        ),
        "centroids": len(
            centroids
        ),
        "context_tests": len(
            context_tests
        ),
        "dose_tests": len(
            dose_tests
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

    if score_long[
        "principal_component_score"
    ].isna().any():
        failures.append(
            "Missing PCA scores detected."
        )

    if loadings[
        "loading"
    ].isna().any():
        failures.append(
            "Missing PCA loadings detected."
        )

    for label, tests in [
        (
            "context_tests",
            context_tests,
        ),
        (
            "dose_tests",
            dose_tests,
        ),
    ]:
        q_values = pd.to_numeric(
            tests[
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

    unresolved_features = int(
        (
            loadings[
                "antigen_target"
            ]
            == "unresolved"
        ).sum()
    )

    if unresolved_features:
        failures.append(
            f"{unresolved_features} loading rows have "
            "unresolved antigen annotation."
        )

    raw_pc80 = int(
        variance.loc[
            (
                variance[
                    "matrix_representation"
                ]
                == "raw_log2_change"
            )
            & (
                variance[
                    "cumulative_variance_ratio"
                ]
                >= 0.80
            ),
            "component_number",
        ].min()
    )

    bpv_pc80 = int(
        variance.loc[
            (
                variance[
                    "matrix_representation"
                ]
                == "bpv_calibrated_log2_change"
            )
            & (
                variance[
                    "cumulative_variance_ratio"
                ]
                >= 0.80
            ),
            "component_number",
        ].min()
    )

    context_fdr = int(
        context_tests[
            "fdr_significant"
        ].sum()
    )

    dose_fdr = int(
        dose_tests[
            "fdr_significant"
        ].sum()
    )

    decision_value = (
        "READY_FOR_PHASE2C4B_PCA_STABILITY_AND_CLUSTERING"
        if not failures
        else "PHASE2C4A_REPAIR_REQUIRED"
    )

    score_output = (
        PROCESSED
        / "phase2C4A_fiji_pca_scores_long.tsv"
    )

    variance_output = (
        TABLES
        / "phase2C4A_fiji_pca_variance.tsv"
    )

    loading_output = (
        TABLES
        / "phase2C4A_fiji_pca_loadings_top10.tsv"
    )

    extreme_output = (
        TABLES
        / "phase2C4A_fiji_pca_loading_extremes.tsv"
    )

    contribution_output = (
        TABLES
        / "phase2C4A_fiji_pca_axis_contributions.tsv"
    )

    centroid_output = (
        TABLES
        / "phase2C4A_fiji_pca_centroids.tsv"
    )

    context_output = (
        TABLES
        / "phase2C4A_fiji_primary_vs_recall_pc_tests.tsv"
    )

    dose_output = (
        TABLES
        / "phase2C4A_fiji_recall_dose_pc_global_tests.tsv"
    )

    decision_output = (
        TABLES
        / "phase2C4A_fiji_pca_architecture_decision.tsv"
    )

    write_tsv(
        score_long,
        score_output,
    )

    write_tsv(
        variance,
        variance_output,
    )

    write_tsv(
        loadings,
        loading_output,
    )

    write_tsv(
        extremes,
        extreme_output,
    )

    write_tsv(
        contributions,
        contribution_output,
    )

    write_tsv(
        centroids,
        centroid_output,
    )

    write_tsv(
        context_tests,
        context_output,
    )

    write_tsv(
        dose_tests,
        dose_output,
    )

    decision_frame = pd.DataFrame(
        [
            {
                "decision": decision_value,
                "raw_features": 92,
                "bpv_calibrated_features": 77,
                "raw_principal_components": 79,
                "bpv_calibrated_principal_components": 77,
                "variance_rows": len(
                    variance
                ),
                "pca_score_rows": len(
                    score_long
                ),
                "top10_loading_rows": len(
                    loadings
                ),
                "loading_extreme_rows": len(
                    extremes
                ),
                "axis_contribution_rows": len(
                    contributions
                ),
                "centroid_rows": len(
                    centroids
                ),
                "primary_vs_recall_pc_tests": len(
                    context_tests
                ),
                "primary_vs_recall_fdr_findings": (
                    context_fdr
                ),
                "recall_dose_pc_tests": len(
                    dose_tests
                ),
                "recall_dose_fdr_findings": (
                    dose_fdr
                ),
                "raw_components_for_80_percent_variance": (
                    raw_pc80
                ),
                "bpv_components_for_80_percent_variance": (
                    bpv_pc80
                ),
                "unresolved_feature_annotations": (
                    unresolved_features
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
        / "phase2C4A_fiji_standardized_pca_architecture_report.md"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as report:
        report.write(
            "# Phase 2C4A Fiji standardized PCA architecture\n\n"
        )

        report.write("## Decision\n\n")
        report.write(
            f"**{decision_value}**\n\n"
        )

        report.write(
            "- Participants: 80\n"
        )
        report.write(
            "- Raw features: 92\n"
        )
        report.write(
            "- BPV-calibrated features: 77\n"
        )
        report.write(
            "- Features were standardized to zero mean and unit "
            "variance separately within each representation.\n"
        )
        report.write(
            f"- Raw components required for 80% variance: "
            f"{raw_pc80}\n"
        )
        report.write(
            f"- BPV-calibrated components required for 80% variance: "
            f"{bpv_pc80}\n"
        )
        report.write(
            f"- Primary-versus-recall FDR-significant PCs among "
            f"PC1–PC10: {context_fdr}\n"
        )
        report.write(
            f"- Recall-dose FDR-significant PCs among "
            f"PC1–PC10: {dose_fdr}\n\n"
        )

        report.write(
            "The PCA models are unsupervised and were fit without "
            "immunization-context or dose-group labels. Group labels "
            "were introduced only after PCA to describe score centroids "
            "and test PC-score differences.\n\n"
        )

        report.write(
            "Loadings for PC1–PC10 were annotated by antigen and "
            "systems-serology feature family. PCA component signs are "
            "mathematically arbitrary; biological interpretation should "
            "therefore focus on relative loading architecture, absolute "
            "contribution and internally consistent score direction "
            "within the fitted model.\n\n"
        )

        report.write(
            "Phase 2C4B should evaluate leave-one-feature-family-out "
            "stability, bootstrap component reproducibility, centroid "
            "separation and unsupervised clustering across plausible "
            "cluster numbers. Raw and BPV-calibrated representations "
            "must remain separate throughout that analysis.\n"
        )

    print(
        "===== PHASE 2C4A COMPLETE ====="
    )

    print(
        f"Decision: {decision_value}"
    )

    print(
        f"Variance rows: {len(variance)}"
    )

    print(
        f"PCA score rows: {len(score_long)}"
    )

    print(
        f"Top-10 loading rows: {len(loadings)}"
    )

    print(
        f"Loading-extreme rows: {len(extremes)}"
    )

    print(
        "Axis-contribution rows: "
        f"{len(contributions)}"
    )

    print(
        f"Centroid rows: {len(centroids)}"
    )

    print(
        "Primary-vs-recall PC tests: "
        f"{len(context_tests)}"
    )

    print(
        "Primary-vs-recall FDR findings: "
        f"{context_fdr}"
    )

    print(
        f"Recall-dose PC tests: {len(dose_tests)}"
    )

    print(
        "Recall-dose FDR findings: "
        f"{dose_fdr}"
    )

    print(
        "Raw PCs for 80% variance: "
        f"{raw_pc80}"
    )

    print(
        "BPV-calibrated PCs for 80% variance: "
        f"{bpv_pc80}"
    )

    print(
        f"Report: {report_path}"
    )

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
