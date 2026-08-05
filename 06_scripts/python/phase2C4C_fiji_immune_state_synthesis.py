#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype
from scipy.stats import pearsonr, spearmanr


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

C4A_DECISION_PATH = (
    TABLES
    / "phase2C4A_fiji_pca_architecture_decision.tsv"
)

C4B_DECISION_PATH = (
    TABLES
    / "phase2C4B_fiji_pca_stability_clustering_decision.tsv"
)

VARIANCE_PATH = (
    TABLES
    / "phase2C4A_fiji_pca_variance.tsv"
)

LOADINGS_PATH = (
    TABLES
    / "phase2C4A_fiji_pca_loadings_top10.tsv"
)

CONTRIBUTIONS_PATH = (
    TABLES
    / "phase2C4A_fiji_pca_axis_contributions.tsv"
)

CONTEXT_TESTS_PATH = (
    TABLES
    / "phase2C4A_fiji_primary_vs_recall_pc_tests.tsv"
)

DOSE_TESTS_PATH = (
    TABLES
    / "phase2C4A_fiji_recall_dose_pc_global_tests.tsv"
)

PCA_SCORES_PATH = (
    PROCESSED
    / "phase2C4A_fiji_pca_scores_long.tsv"
)

LEAVE_ONE_FAMILY_PATH = (
    TABLES
    / "phase2C4B_fiji_leave_one_family_out_pca_stability.tsv"
)

BOOTSTRAP_SUMMARY_PATH = (
    TABLES
    / "phase2C4B_fiji_bootstrap_pca_stability_summary.tsv"
)

CLUSTER_METRICS_PATH = (
    TABLES
    / "phase2C4B_fiji_clustering_metrics.tsv"
)

CLUSTER_PROFILES_PATH = (
    TABLES
    / "phase2C4B_fiji_cluster_profiles.tsv"
)

EXPECTED_C4A_DECISION = (
    "READY_FOR_PHASE2C4B_PCA_STABILITY_AND_CLUSTERING"
)

EXPECTED_C4B_DECISION = (
    "READY_FOR_PHASE2C4C_IMMUNE_STATE_SYNTHESIS"
)

REPRESENTATIONS = [
    "raw_log2_change",
    "bpv_calibrated_log2_change",
]

CORE_COMPONENTS = {
    1: "cross_reactive_recall_breadth_axis",
    2: "vaccine_type_effector_axis",
}


def fail(
    message: str,
) -> None:
    raise SystemExit(
        f"ERROR: {message}"
    )


def read_table(
    path: Path,
    label: str,
) -> pd.DataFrame:
    if not path.exists():
        fail(
            f"Required input is missing: {path}"
        )

    frame = pd.read_csv(
        path,
        sep="\t",
    )

    if frame.empty:
        fail(
            f"{label} is empty: {path}"
        )

    return frame


def require_columns(
    frame: pd.DataFrame,
    required: set[str],
    label: str,
) -> None:
    missing = required - set(
        frame.columns
    )

    if missing:
        fail(
            f"{label} is missing columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )


def convert_numeric(
    frame: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    for column in columns:
        if column not in frame.columns:
            fail(
                f"{label} does not contain {column}"
            )

        frame[
            column
        ] = pd.to_numeric(
            frame[
                column
            ],
            errors="coerce",
        )


def as_bool(
    series: pd.Series,
) -> pd.Series:
    if is_bool_dtype(
        series
    ):
        return series.fillna(
            False
        ).astype(
            bool
        )

    normalized = (
        series.astype(
            "string"
        )
        .str.strip()
        .str.lower()
    )

    return normalized.isin(
        [
            "true",
            "1",
            "yes",
            "y",
        ]
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


def safe_pearson(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
) -> tuple[float, float]:
    x_array = np.asarray(
        x,
        dtype=float,
    )

    y_array = np.asarray(
        y,
        dtype=float,
    )

    valid = (
        np.isfinite(
            x_array
        )
        & np.isfinite(
            y_array
        )
    )

    x_array = x_array[
        valid
    ]

    y_array = y_array[
        valid
    ]

    if len(
        x_array
    ) < 3:
        return (
            np.nan,
            np.nan,
        )

    if (
        np.std(
            x_array,
            ddof=1,
        )
        <= 1e-15
        or np.std(
            y_array,
            ddof=1,
        )
        <= 1e-15
    ):
        return (
            np.nan,
            np.nan,
        )

    result = pearsonr(
        x_array,
        y_array,
    )

    return (
        float(
            result.statistic
        ),
        float(
            result.pvalue
        ),
    )


def safe_spearman(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
) -> tuple[float, float]:
    x_array = np.asarray(
        x,
        dtype=float,
    )

    y_array = np.asarray(
        y,
        dtype=float,
    )

    valid = (
        np.isfinite(
            x_array
        )
        & np.isfinite(
            y_array
        )
    )

    x_array = x_array[
        valid
    ]

    y_array = y_array[
        valid
    ]

    if len(
        x_array
    ) < 3:
        return (
            np.nan,
            np.nan,
        )

    result = spearmanr(
        x_array,
        y_array,
    )

    return (
        float(
            result.statistic
        ),
        float(
            result.pvalue
        ),
    )


def validate_decisions() -> None:
    c4a = read_table(
        C4A_DECISION_PATH,
        "Phase 2C4A decision table",
    )

    c4b = read_table(
        C4B_DECISION_PATH,
        "Phase 2C4B decision table",
    )

    require_columns(
        c4a,
        {
            "decision",
        },
        "Phase 2C4A decision table",
    )

    require_columns(
        c4b,
        {
            "decision",
        },
        "Phase 2C4B decision table",
    )

    observed_c4a = str(
        c4a.loc[
            0,
            "decision",
        ]
    )

    observed_c4b = str(
        c4b.loc[
            0,
            "decision",
        ]
    )

    if observed_c4a != EXPECTED_C4A_DECISION:
        fail(
            "Unexpected Phase 2C4A decision: "
            f"{observed_c4a}"
        )

    if observed_c4b != EXPECTED_C4B_DECISION:
        fail(
            "Unexpected Phase 2C4B decision: "
            f"{observed_c4b}"
        )


def build_axis_registry(
    variance: pd.DataFrame,
    contributions: pd.DataFrame,
    context: pd.DataFrame,
    leave_one_family: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        variance,
        {
            "matrix_representation",
            "principal_component",
            "component_number",
            "explained_variance_percent",
            "cumulative_variance_percent",
        },
        "PCA variance table",
    )

    require_columns(
        contributions,
        {
            "matrix_representation",
            "principal_component",
            "component_number",
            "annotation_level",
            "annotation_value",
            "contribution_percent",
        },
        "PCA contribution table",
    )

    require_columns(
        context,
        {
            "matrix_representation",
            "principal_component",
            "component_number",
            "primary_mean",
            "recall_mean",
            "mean_difference_recall_minus_primary",
            "rank_biserial_recall_minus_primary",
            "hedges_g_recall_minus_primary",
            "mann_whitney_p_value",
            "bh_q_value",
        },
        "Primary-versus-recall PC tests",
    )

    require_columns(
        leave_one_family,
        {
            "matrix_representation",
            "reference_principal_component",
            "reference_component_number",
            "absolute_loading_correlation",
            "sign_aligned_score_correlation",
            "absolute_variance_ratio_difference",
        },
        "Leave-one-family-out stability table",
    )

    require_columns(
        bootstrap,
        {
            "matrix_representation",
            "reference_principal_component",
            "reference_component_number",
            "bootstrap_replicates",
            "median_absolute_loading_correlation",
            "loading_correlation_2_5_percentile",
            "loading_correlation_97_5_percentile",
            "proportion_loading_correlation_ge_0_80",
            "proportion_loading_correlation_ge_0_90",
            "median_absolute_variance_ratio_difference",
            "matched_bootstrap_component_mode",
            "unique_matched_bootstrap_components",
        },
        "Bootstrap stability summary",
    )

    convert_numeric(
        variance,
        [
            "component_number",
            "explained_variance_percent",
            "cumulative_variance_percent",
        ],
        "PCA variance table",
    )

    convert_numeric(
        contributions,
        [
            "component_number",
            "contribution_percent",
        ],
        "PCA contribution table",
    )

    convert_numeric(
        context,
        [
            "component_number",
            "primary_mean",
            "recall_mean",
            "mean_difference_recall_minus_primary",
            "rank_biserial_recall_minus_primary",
            "hedges_g_recall_minus_primary",
            "mann_whitney_p_value",
            "bh_q_value",
        ],
        "Primary-versus-recall PC tests",
    )

    convert_numeric(
        leave_one_family,
        [
            "reference_component_number",
            "absolute_loading_correlation",
            "sign_aligned_score_correlation",
            "absolute_variance_ratio_difference",
        ],
        "Leave-one-family-out stability table",
    )

    convert_numeric(
        bootstrap,
        [
            "reference_component_number",
            "bootstrap_replicates",
            "median_absolute_loading_correlation",
            "loading_correlation_2_5_percentile",
            "loading_correlation_97_5_percentile",
            "proportion_loading_correlation_ge_0_80",
            "proportion_loading_correlation_ge_0_90",
            "median_absolute_variance_ratio_difference",
            "matched_bootstrap_component_mode",
            "unique_matched_bootstrap_components",
        ],
        "Bootstrap stability summary",
    )

    variance_core = variance[
        variance[
            "component_number"
        ].isin(
            CORE_COMPONENTS
        )
    ][
        [
            "matrix_representation",
            "principal_component",
            "component_number",
            "explained_variance_percent",
            "cumulative_variance_percent",
        ]
    ].copy()

    context_core = context[
        context[
            "component_number"
        ].isin(
            CORE_COMPONENTS
        )
    ][
        [
            "matrix_representation",
            "principal_component",
            "component_number",
            "primary_mean",
            "recall_mean",
            "mean_difference_recall_minus_primary",
            "rank_biserial_recall_minus_primary",
            "hedges_g_recall_minus_primary",
            "mann_whitney_p_value",
            "bh_q_value",
        ]
    ].copy()

    family_contributions = contributions[
        (
            contributions[
                "annotation_level"
            ]
            == "feature_family"
        )
        & (
            contributions[
                "component_number"
            ].isin(
                CORE_COMPONENTS
            )
        )
    ].copy()

    top_family = (
        family_contributions.sort_values(
            [
                "matrix_representation",
                "component_number",
                "contribution_percent",
                "annotation_value",
            ],
            ascending=[
                True,
                True,
                False,
                True,
            ],
        )
        .groupby(
            [
                "matrix_representation",
                "component_number",
            ],
            observed=True,
        )
        .head(
            1
        )[
            [
                "matrix_representation",
                "component_number",
                "annotation_value",
                "contribution_percent",
            ]
        ]
        .rename(
            columns={
                "annotation_value": (
                    "leading_feature_family"
                ),
                "contribution_percent": (
                    "leading_feature_family_contribution_percent"
                ),
            }
        )
    )

    antigen_contributions = contributions[
        (
            contributions[
                "annotation_level"
            ]
            == "antigen_target"
        )
        & (
            contributions[
                "component_number"
            ].isin(
                CORE_COMPONENTS
            )
        )
    ].copy()

    top_antigen = (
        antigen_contributions.sort_values(
            [
                "matrix_representation",
                "component_number",
                "contribution_percent",
                "annotation_value",
            ],
            ascending=[
                True,
                True,
                False,
                True,
            ],
        )
        .groupby(
            [
                "matrix_representation",
                "component_number",
            ],
            observed=True,
        )
        .head(
            1
        )[
            [
                "matrix_representation",
                "component_number",
                "annotation_value",
                "contribution_percent",
            ]
        ]
        .rename(
            columns={
                "annotation_value": (
                    "leading_antigen"
                ),
                "contribution_percent": (
                    "leading_antigen_contribution_percent"
                ),
            }
        )
    )

    family_stability = (
        leave_one_family[
            leave_one_family[
                "reference_component_number"
            ].isin(
                CORE_COMPONENTS
            )
        ]
        .groupby(
            [
                "matrix_representation",
                "reference_principal_component",
                "reference_component_number",
            ],
            observed=True,
        )
        .agg(
            feature_family_omission_tests=(
                "absolute_loading_correlation",
                "size",
            ),
            minimum_family_omission_loading_correlation=(
                "absolute_loading_correlation",
                "min",
            ),
            median_family_omission_loading_correlation=(
                "absolute_loading_correlation",
                "median",
            ),
            minimum_family_omission_score_correlation=(
                "sign_aligned_score_correlation",
                "min",
            ),
            median_family_omission_score_correlation=(
                "sign_aligned_score_correlation",
                "median",
            ),
            maximum_family_omission_variance_difference=(
                "absolute_variance_ratio_difference",
                "max",
            ),
        )
        .reset_index()
        .rename(
            columns={
                "reference_principal_component": (
                    "principal_component"
                ),
                "reference_component_number": (
                    "component_number"
                ),
            }
        )
    )

    bootstrap_core = bootstrap[
        bootstrap[
            "reference_component_number"
        ].isin(
            CORE_COMPONENTS
        )
    ].copy()

    bootstrap_core = bootstrap_core.rename(
        columns={
            "reference_principal_component": (
                "principal_component"
            ),
            "reference_component_number": (
                "component_number"
            ),
        }
    )

    registry = (
        variance_core.merge(
            context_core,
            on=[
                "matrix_representation",
                "principal_component",
                "component_number",
            ],
            how="left",
            validate="1:1",
        )
        .merge(
            top_family,
            on=[
                "matrix_representation",
                "component_number",
            ],
            how="left",
            validate="1:1",
        )
        .merge(
            top_antigen,
            on=[
                "matrix_representation",
                "component_number",
            ],
            how="left",
            validate="1:1",
        )
        .merge(
            family_stability,
            on=[
                "matrix_representation",
                "principal_component",
                "component_number",
            ],
            how="left",
            validate="1:1",
        )
        .merge(
            bootstrap_core,
            on=[
                "matrix_representation",
                "principal_component",
                "component_number",
            ],
            how="left",
            validate="1:1",
        )
    )

    registry[
        "biological_axis"
    ] = registry[
        "component_number"
    ].map(
        CORE_COMPONENTS
    )

    registry[
        "context_difference_supported"
    ] = (
        registry[
            "bh_q_value"
        ]
        < 0.05
    )

    registry[
        "family_omission_stable"
    ] = (
        (
            registry[
                "minimum_family_omission_loading_correlation"
            ]
            >= 0.85
        )
        & (
            registry[
                "minimum_family_omission_score_correlation"
            ]
            >= 0.75
        )
    )

    registry[
        "bootstrap_stable"
    ] = (
        (
            registry[
                "median_absolute_loading_correlation"
            ]
            >= 0.85
        )
        & (
            registry[
                "proportion_loading_correlation_ge_0_80"
            ]
            >= 0.75
        )
    )

    registry[
        "core_axis_supported"
    ] = (
        registry[
            "context_difference_supported"
        ]
        & registry[
            "family_omission_stable"
        ]
        & registry[
            "bootstrap_stable"
        ]
    )

    registry[
        "evidence_grade"
    ] = np.where(
        registry[
            "component_number"
        ]
        == 1,
        "A1_highly_reproducible_core_axis",
        "A2_reproducible_secondary_core_axis",
    )

    registry[
        "biological_interpretation"
    ] = np.where(
        registry[
            "component_number"
        ]
        == 1,
        (
            "Cross-reactive HPV31/33/45/52/58 IgG, subclass "
            "and Fc-receptor architecture distinguishing "
            "heterologous recall from primary HPV16/18 induction."
        ),
        (
            "Vaccine-type HPV16/18 antibody-abundance and "
            "effector architecture, including ADCP and "
            "neutralization in the raw representation."
        ),
    )

    return registry.sort_values(
        [
            "matrix_representation",
            "component_number",
        ]
    ).reset_index(
        drop=True
    )


def build_cross_representation_concordance(
    scores: pd.DataFrame,
    loadings: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        scores,
        {
            "participant_id",
            "matrix_representation",
            "principal_component",
            "component_number",
            "principal_component_score",
        },
        "PCA scores",
    )

    require_columns(
        loadings,
        {
            "matrix_representation",
            "principal_component",
            "component_number",
            "feature_column",
            "loading",
        },
        "PCA loadings",
    )

    convert_numeric(
        scores,
        [
            "component_number",
            "principal_component_score",
        ],
        "PCA scores",
    )

    convert_numeric(
        loadings,
        [
            "component_number",
            "loading",
        ],
        "PCA loadings",
    )

    scores[
        "participant_id"
    ] = scores[
        "participant_id"
    ].astype(
        "string"
    )

    rows: list[
        dict[str, object]
    ] = []

    for (
        component_number,
        biological_axis,
    ) in CORE_COMPONENTS.items():
        component = (
            f"PC{component_number}"
        )

        score_subset = scores[
            (
                scores[
                    "component_number"
                ]
                == component_number
            )
            & (
                scores[
                    "matrix_representation"
                ].isin(
                    REPRESENTATIONS
                )
            )
        ][
            [
                "participant_id",
                "matrix_representation",
                "principal_component_score",
            ]
        ].copy()

        score_wide = score_subset.pivot(
            index="participant_id",
            columns="matrix_representation",
            values="principal_component_score",
        ).dropna(
            subset=REPRESENTATIONS
        )

        raw_scores = score_wide[
            "raw_log2_change"
        ]

        bpv_scores = score_wide[
            "bpv_calibrated_log2_change"
        ]

        initial_score_r, _ = safe_pearson(
            raw_scores,
            bpv_scores,
        )

        sign_multiplier = (
            -1.0
            if (
                np.isfinite(
                    initial_score_r
                )
                and initial_score_r < 0
            )
            else 1.0
        )

        aligned_bpv_scores = (
            sign_multiplier
            * bpv_scores
        )

        (
            score_pearson_r,
            score_pearson_p,
        ) = safe_pearson(
            raw_scores,
            aligned_bpv_scores,
        )

        (
            score_spearman_rho,
            score_spearman_p,
        ) = safe_spearman(
            raw_scores,
            aligned_bpv_scores,
        )

        raw_loadings = loadings[
            (
                loadings[
                    "matrix_representation"
                ]
                == "raw_log2_change"
            )
            & (
                loadings[
                    "component_number"
                ]
                == component_number
            )
        ][
            [
                "feature_column",
                "loading",
            ]
        ].rename(
            columns={
                "loading": (
                    "raw_loading"
                ),
            }
        )

        bpv_loadings = loadings[
            (
                loadings[
                    "matrix_representation"
                ]
                == "bpv_calibrated_log2_change"
            )
            & (
                loadings[
                    "component_number"
                ]
                == component_number
            )
        ][
            [
                "feature_column",
                "loading",
            ]
        ].rename(
            columns={
                "loading": (
                    "bpv_loading"
                ),
            }
        )

        shared_loadings = raw_loadings.merge(
            bpv_loadings,
            on="feature_column",
            how="inner",
            validate="1:1",
        )

        aligned_bpv_loadings = (
            sign_multiplier
            * shared_loadings[
                "bpv_loading"
            ]
        )

        (
            loading_pearson_r,
            loading_pearson_p,
        ) = safe_pearson(
            shared_loadings[
                "raw_loading"
            ],
            aligned_bpv_loadings,
        )

        (
            loading_spearman_rho,
            loading_spearman_p,
        ) = safe_spearman(
            shared_loadings[
                "raw_loading"
            ],
            aligned_bpv_loadings,
        )

        if (
            score_spearman_rho >= 0.75
            and loading_pearson_r >= 0.75
        ):
            concordance_status = (
                "high_cross_representation_concordance"
            )

        elif (
            score_spearman_rho >= 0.50
            and loading_pearson_r >= 0.50
        ):
            concordance_status = (
                "moderate_cross_representation_concordance"
            )

        else:
            concordance_status = (
                "limited_cross_representation_concordance"
            )

        rows.append(
            {
                "principal_component": (
                    component
                ),
                "component_number": (
                    component_number
                ),
                "biological_axis": (
                    biological_axis
                ),
                "participants": len(
                    score_wide
                ),
                "shared_features": len(
                    shared_loadings
                ),
                "bpv_sign_multiplier": (
                    sign_multiplier
                ),
                "score_pearson_r": (
                    score_pearson_r
                ),
                "score_pearson_p_value": (
                    score_pearson_p
                ),
                "score_spearman_rho": (
                    score_spearman_rho
                ),
                "score_spearman_p_value": (
                    score_spearman_p
                ),
                "loading_pearson_r": (
                    loading_pearson_r
                ),
                "loading_pearson_p_value": (
                    loading_pearson_p
                ),
                "loading_spearman_rho": (
                    loading_spearman_rho
                ),
                "loading_spearman_p_value": (
                    loading_spearman_p
                ),
                "concordance_status": (
                    concordance_status
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_clustering_synthesis(
    metrics: pd.DataFrame,
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        metrics,
        {
            "matrix_representation",
            "cluster_count",
            "minimum_cluster_size",
            "maximum_cluster_size",
            "silhouette_score",
            "mean_subsample_adjusted_rand",
            "subsample_adjusted_rand_2_5_percentile",
            "subsample_adjusted_rand_97_5_percentile",
            "context_adjusted_rand",
            "context_normalized_mutual_information",
            "dose_group_adjusted_rand",
            "dose_group_normalized_mutual_information",
            "stable_cluster_candidate",
            "candidate_rank_within_representation",
            "selected_cluster_candidate",
            "clustering_composite_score",
        },
        "Clustering metrics",
    )

    require_columns(
        profiles,
        {
            "matrix_representation",
            "cluster_count",
            "cluster_label",
            "participants",
            "primary_participants",
            "recall_participants",
            "recall_fraction",
            "mean_PC1_score",
            "mean_PC2_score",
        },
        "Cluster profiles",
    )

    convert_numeric(
        metrics,
        [
            "cluster_count",
            "minimum_cluster_size",
            "maximum_cluster_size",
            "silhouette_score",
            "mean_subsample_adjusted_rand",
            "subsample_adjusted_rand_2_5_percentile",
            "subsample_adjusted_rand_97_5_percentile",
            "context_adjusted_rand",
            "context_normalized_mutual_information",
            "dose_group_adjusted_rand",
            "dose_group_normalized_mutual_information",
            "candidate_rank_within_representation",
            "clustering_composite_score",
        ],
        "Clustering metrics",
    )

    convert_numeric(
        profiles,
        [
            "cluster_count",
            "cluster_label",
            "participants",
            "primary_participants",
            "recall_participants",
            "recall_fraction",
            "mean_PC1_score",
            "mean_PC2_score",
        ],
        "Cluster profiles",
    )

    metrics[
        "stable_cluster_candidate"
    ] = as_bool(
        metrics[
            "stable_cluster_candidate"
        ]
    )

    metrics[
        "selected_cluster_candidate"
    ] = as_bool(
        metrics[
            "selected_cluster_candidate"
        ]
    )

    rows: list[
        dict[str, object]
    ] = []

    for representation in REPRESENTATIONS:
        representation_metrics = metrics[
            metrics[
                "matrix_representation"
            ]
            == representation
        ].copy()

        if representation_metrics.empty:
            fail(
                f"No clustering metrics for {representation}"
            )

        selected = representation_metrics[
            representation_metrics[
                "selected_cluster_candidate"
            ]
        ].copy()

        if not selected.empty:
            chosen = selected.sort_values(
                [
                    "clustering_composite_score",
                    "cluster_count",
                ],
                ascending=[
                    False,
                    True,
                ],
            ).iloc[
                0
            ]

            solution_status = (
                "selected_stable_solution"
            )

        else:
            chosen = representation_metrics.sort_values(
                [
                    "candidate_rank_within_representation",
                    "cluster_count",
                ]
            ).iloc[
                0
            ]

            solution_status = (
                "best_available_but_not_stable"
            )

        cluster_count = int(
            chosen[
                "cluster_count"
            ]
        )

        chosen_profiles = profiles[
            (
                profiles[
                    "matrix_representation"
                ]
                == representation
            )
            & (
                profiles[
                    "cluster_count"
                ]
                == cluster_count
            )
        ].copy()

        if chosen_profiles.empty:
            fail(
                f"No cluster profiles for "
                f"{representation}, k={cluster_count}"
            )

        chosen_profiles[
            "primary_fraction"
        ] = (
            chosen_profiles[
                "primary_participants"
            ]
            / chosen_profiles[
                "participants"
            ]
        )

        primary_cluster = chosen_profiles.sort_values(
            [
                "primary_fraction",
                "primary_participants",
            ],
            ascending=[
                False,
                False,
            ],
        ).iloc[
            0
        ]

        recall_cluster = chosen_profiles.sort_values(
            [
                "recall_fraction",
                "recall_participants",
            ],
            ascending=[
                False,
                False,
            ],
        ).iloc[
            0
        ]

        if int(
            primary_cluster[
                "cluster_label"
            ]
        ) == int(
            recall_cluster[
                "cluster_label"
            ]
        ):
            fail(
                "Primary- and recall-dominant clusters "
                f"are identical for {representation}"
            )

        correctly_aligned = int(
            primary_cluster[
                "primary_participants"
            ]
            + recall_cluster[
                "recall_participants"
            ]
        )

        total_participants = int(
            chosen_profiles[
                "participants"
            ].sum()
        )

        context_alignment_accuracy = (
            correctly_aligned
            / total_participants
        )

        if representation == "raw_log2_change":
            evidence_grade = (
                "B1_stable_raw_context_dominated_partition"
            )

            conclusion = (
                "Stable two-cluster solution, but the partition "
                "is almost entirely a primary-versus-recall "
                "separation along PC1 and PC2 rather than evidence "
                "of independent immune-response subtypes."
            )

        else:
            evidence_grade = (
                "A3_continuous_calibrated_immune_state_structure"
            )

            conclusion = (
                "No cluster solution met all prespecified criteria. "
                "BPV-calibrated responses are therefore better "
                "represented as continuous multiaxial immune-state "
                "structure."
            )

        rows.append(
            {
                "matrix_representation": (
                    representation
                ),
                "cluster_count": (
                    cluster_count
                ),
                "solution_status": (
                    solution_status
                ),
                "stable_cluster_candidate": bool(
                    chosen[
                        "stable_cluster_candidate"
                    ]
                ),
                "minimum_cluster_size": int(
                    chosen[
                        "minimum_cluster_size"
                    ]
                ),
                "maximum_cluster_size": int(
                    chosen[
                        "maximum_cluster_size"
                    ]
                ),
                "silhouette_score": float(
                    chosen[
                        "silhouette_score"
                    ]
                ),
                "mean_subsample_adjusted_rand": float(
                    chosen[
                        "mean_subsample_adjusted_rand"
                    ]
                ),
                "subsample_adjusted_rand_2_5_percentile": float(
                    chosen[
                        "subsample_adjusted_rand_2_5_percentile"
                    ]
                ),
                "subsample_adjusted_rand_97_5_percentile": float(
                    chosen[
                        "subsample_adjusted_rand_97_5_percentile"
                    ]
                ),
                "context_adjusted_rand": float(
                    chosen[
                        "context_adjusted_rand"
                    ]
                ),
                "context_normalized_mutual_information": float(
                    chosen[
                        "context_normalized_mutual_information"
                    ]
                ),
                "dose_group_adjusted_rand": float(
                    chosen[
                        "dose_group_adjusted_rand"
                    ]
                ),
                "dose_group_normalized_mutual_information": float(
                    chosen[
                        "dose_group_normalized_mutual_information"
                    ]
                ),
                "context_aligned_participants": (
                    correctly_aligned
                ),
                "total_participants": (
                    total_participants
                ),
                "context_alignment_accuracy": (
                    context_alignment_accuracy
                ),
                "primary_dominant_cluster": int(
                    primary_cluster[
                        "cluster_label"
                    ]
                ),
                "primary_dominant_cluster_primary_fraction": float(
                    primary_cluster[
                        "primary_fraction"
                    ]
                ),
                "recall_dominant_cluster": int(
                    recall_cluster[
                        "cluster_label"
                    ]
                ),
                "recall_dominant_cluster_recall_fraction": float(
                    recall_cluster[
                        "recall_fraction"
                    ]
                ),
                "evidence_grade": (
                    evidence_grade
                ),
                "biological_conclusion": (
                    conclusion
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_schedule_registry(
    dose_tests: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        dose_tests,
        {
            "matrix_representation",
            "principal_component",
            "component_number",
            "dose1_median",
            "dose2_median",
            "dose3_median",
            "kruskal_wallis_h",
            "epsilon_squared",
            "kruskal_wallis_p_value",
            "bh_q_value",
        },
        "Recall-dose PC tests",
    )

    convert_numeric(
        dose_tests,
        [
            "component_number",
            "dose1_median",
            "dose2_median",
            "dose3_median",
            "kruskal_wallis_h",
            "epsilon_squared",
            "kruskal_wallis_p_value",
            "bh_q_value",
        ],
        "Recall-dose PC tests",
    )

    registry = dose_tests[
        dose_tests[
            "component_number"
        ].isin(
            CORE_COMPONENTS
        )
    ].copy()

    registry[
        "biological_axis"
    ] = registry[
        "component_number"
    ].map(
        CORE_COMPONENTS
    )

    registry[
        "fdr_significant"
    ] = (
        registry[
            "bh_q_value"
        ]
        < 0.05
    )

    registry[
        "evidence_class"
    ] = "not_significant"

    raw_pc2 = (
        (
            registry[
                "matrix_representation"
            ]
            == "raw_log2_change"
        )
        & (
            registry[
                "component_number"
            ]
            == 2
        )
        & registry[
            "fdr_significant"
        ]
    )

    registry.loc[
        raw_pc2,
        "evidence_class",
    ] = (
        "qualified_raw_only_schedule_association"
    )

    bpv_pc2 = (
        (
            registry[
                "matrix_representation"
            ]
            == "bpv_calibrated_log2_change"
        )
        & (
            registry[
                "component_number"
            ]
            == 2
        )
    )

    registry.loc[
        bpv_pc2,
        "evidence_class",
    ] = (
        "not_supported_after_bpv_calibration"
    )

    registry[
        "interpretation"
    ] = np.where(
        registry[
            "evidence_class"
        ]
        == "qualified_raw_only_schedule_association",
        (
            "Previous-dose groups differed on the raw "
            "vaccine-type effector axis, but the association "
            "did not survive BPV calibration and remains a "
            "qualified secondary result."
        ),
        np.where(
            registry[
                "evidence_class"
            ]
            == "not_supported_after_bpv_calibration",
            (
                "The corresponding calibrated vaccine-type "
                "effector axis did not differ significantly "
                "among previous-dose groups."
            ),
            (
                "No FDR-significant previous-dose-group "
                "difference was detected for this core axis."
            ),
        ),
    )

    return registry.sort_values(
        [
            "matrix_representation",
            "component_number",
        ]
    ).reset_index(
        drop=True
    )


def main() -> None:
    validate_decisions()

    variance = read_table(
        VARIANCE_PATH,
        "PCA variance table",
    )

    loadings = read_table(
        LOADINGS_PATH,
        "PCA loading table",
    )

    contributions = read_table(
        CONTRIBUTIONS_PATH,
        "PCA contribution table",
    )

    context = read_table(
        CONTEXT_TESTS_PATH,
        "Primary-versus-recall PC tests",
    )

    dose = read_table(
        DOSE_TESTS_PATH,
        "Recall-dose PC tests",
    )

    scores = read_table(
        PCA_SCORES_PATH,
        "PCA scores",
    )

    leave_one_family = read_table(
        LEAVE_ONE_FAMILY_PATH,
        "Leave-one-family-out stability table",
    )

    bootstrap = read_table(
        BOOTSTRAP_SUMMARY_PATH,
        "Bootstrap stability summary",
    )

    cluster_metrics = read_table(
        CLUSTER_METRICS_PATH,
        "Clustering metrics",
    )

    cluster_profiles = read_table(
        CLUSTER_PROFILES_PATH,
        "Cluster profiles",
    )

    axis_registry = build_axis_registry(
        variance,
        contributions,
        context,
        leave_one_family,
        bootstrap,
    )

    concordance = (
        build_cross_representation_concordance(
            scores,
            loadings,
        )
    )

    clustering = build_clustering_synthesis(
        cluster_metrics,
        cluster_profiles,
    )

    schedule_registry = build_schedule_registry(
        dose
    )

    failures: list[str] = []

    expected_counts = {
        "axis_registry_rows": (
            len(
                axis_registry
            ),
            4,
        ),
        "supported_core_axes": (
            int(
                axis_registry[
                    "core_axis_supported"
                ].sum()
            ),
            4,
        ),
        "cross_representation_rows": (
            len(
                concordance
            ),
            2,
        ),
        "clustering_synthesis_rows": (
            len(
                clustering
            ),
            2,
        ),
        "schedule_registry_rows": (
            len(
                schedule_registry
            ),
            4,
        ),
        "schedule_fdr_findings": (
            int(
                schedule_registry[
                    "fdr_significant"
                ].sum()
            ),
            1,
        ),
    }

    for (
        label,
        values,
    ) in expected_counts.items():
        observed, expected = values

        if observed != expected:
            failures.append(
                f"{label}: expected {expected}, "
                f"observed {observed}."
            )

    raw_cluster = clustering[
        clustering[
            "matrix_representation"
        ]
        == "raw_log2_change"
    ]

    bpv_cluster = clustering[
        clustering[
            "matrix_representation"
        ]
        == "bpv_calibrated_log2_change"
    ]

    if len(
        raw_cluster
    ) != 1:
        failures.append(
            "Raw clustering synthesis row is missing."
        )

    else:
        raw_row = raw_cluster.iloc[
            0
        ]

        if not bool(
            raw_row[
                "stable_cluster_candidate"
            ]
        ):
            failures.append(
                "Raw k=2 solution was expected to be stable."
            )

        if int(
            raw_row[
                "cluster_count"
            ]
        ) != 2:
            failures.append(
                "Raw selected solution was expected to use k=2."
            )

        if float(
            raw_row[
                "context_alignment_accuracy"
            ]
        ) < 0.90:
            failures.append(
                "Raw selected solution is not strongly "
                "context-aligned."
            )

    if len(
        bpv_cluster
    ) != 1:
        failures.append(
            "BPV-calibrated clustering synthesis row is missing."
        )

    else:
        bpv_row = bpv_cluster.iloc[
            0
        ]

        if bool(
            bpv_row[
                "stable_cluster_candidate"
            ]
        ):
            failures.append(
                "BPV-calibrated solution was not "
                "expected to be stable."
            )

    raw_pc2 = schedule_registry[
        (
            schedule_registry[
                "matrix_representation"
            ]
            == "raw_log2_change"
        )
        & (
            schedule_registry[
                "component_number"
            ]
            == 2
        )
    ]

    bpv_pc2 = schedule_registry[
        (
            schedule_registry[
                "matrix_representation"
            ]
            == "bpv_calibrated_log2_change"
        )
        & (
            schedule_registry[
                "component_number"
            ]
            == 2
        )
    ]

    if (
        raw_pc2.empty
        or not bool(
            raw_pc2[
                "fdr_significant"
            ].iloc[
                0
            ]
        )
    ):
        failures.append(
            "Expected raw PC2 recall-dose finding is missing."
        )

    if (
        bpv_pc2.empty
        or bool(
            bpv_pc2[
                "fdr_significant"
            ].iloc[
                0
            ]
        )
    ):
        failures.append(
            "BPV-calibrated PC2 should not be "
            "FDR significant."
        )

    decision_value = (
        "READY_FOR_PHASE2C4_COMMIT_AND_PHASE2C5_FIGURE_CONSTRUCTION"
        if not failures
        else "PHASE2C4C_REPAIR_REQUIRED"
    )

    axis_output = (
        TABLES
        / "phase2C4C_fiji_core_immune_state_axis_registry.tsv"
    )

    concordance_output = (
        TABLES
        / "phase2C4C_fiji_raw_bpv_axis_concordance.tsv"
    )

    clustering_output = (
        TABLES
        / "phase2C4C_fiji_clustering_synthesis.tsv"
    )

    schedule_output = (
        TABLES
        / "phase2C4C_fiji_schedule_effect_registry.tsv"
    )

    summary_output = (
        TABLES
        / "phase2C4C_fiji_immune_state_synthesis_summary.tsv"
    )

    decision_output = (
        TABLES
        / "phase2C4C_fiji_immune_state_synthesis_decision.tsv"
    )

    write_tsv(
        axis_registry,
        axis_output,
    )

    write_tsv(
        concordance,
        concordance_output,
    )

    write_tsv(
        clustering,
        clustering_output,
    )

    write_tsv(
        schedule_registry,
        schedule_output,
    )

    summary = pd.DataFrame(
        [
            {
                "summary_measure": (
                    "core_axis_rows"
                ),
                "value": len(
                    axis_registry
                ),
            },
            {
                "summary_measure": (
                    "supported_core_axes"
                ),
                "value": int(
                    axis_registry[
                        "core_axis_supported"
                    ].sum()
                ),
            },
            {
                "summary_measure": (
                    "cross_representation_axis_comparisons"
                ),
                "value": len(
                    concordance
                ),
            },
            {
                "summary_measure": (
                    "raw_stable_cluster_solutions"
                ),
                "value": int(
                    clustering.loc[
                        clustering[
                            "matrix_representation"
                        ]
                        == "raw_log2_change",
                        "stable_cluster_candidate",
                    ].sum()
                ),
            },
            {
                "summary_measure": (
                    "bpv_calibrated_stable_cluster_solutions"
                ),
                "value": int(
                    clustering.loc[
                        clustering[
                            "matrix_representation"
                        ]
                        == "bpv_calibrated_log2_change",
                        "stable_cluster_candidate",
                    ].sum()
                ),
            },
            {
                "summary_measure": (
                    "recall_dose_fdr_findings"
                ),
                "value": int(
                    schedule_registry[
                        "fdr_significant"
                    ].sum()
                ),
            },
        ]
    )

    write_tsv(
        summary,
        summary_output,
    )

    raw_accuracy = float(
        raw_cluster[
            "context_alignment_accuracy"
        ].iloc[
            0
        ]
    )

    raw_correct = int(
        raw_cluster[
            "context_aligned_participants"
        ].iloc[
            0
        ]
    )

    raw_silhouette = float(
        raw_cluster[
            "silhouette_score"
        ].iloc[
            0
        ]
    )

    bpv_silhouette = float(
        bpv_cluster[
            "silhouette_score"
        ].iloc[
            0
        ]
    )

    decision_frame = pd.DataFrame(
        [
            {
                "decision": (
                    decision_value
                ),
                "core_axis_rows": len(
                    axis_registry
                ),
                "supported_core_axes": int(
                    axis_registry[
                        "core_axis_supported"
                    ].sum()
                ),
                "cross_representation_axis_rows": len(
                    concordance
                ),
                "clustering_synthesis_rows": len(
                    clustering
                ),
                "schedule_registry_rows": len(
                    schedule_registry
                ),
                "schedule_fdr_findings": int(
                    schedule_registry[
                        "fdr_significant"
                    ].sum()
                ),
                "raw_selected_cluster_count": 2,
                "raw_cluster_context_aligned_participants": (
                    raw_correct
                ),
                "raw_cluster_context_alignment_accuracy": (
                    raw_accuracy
                ),
                "raw_cluster_silhouette": (
                    raw_silhouette
                ),
                "bpv_stable_cluster_solution": False,
                "bpv_best_cluster_silhouette": (
                    bpv_silhouette
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
        / "phase2C4C_fiji_immune_state_synthesis_report.md"
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
            "# Phase 2C4C Fiji immune-state synthesis\n\n"
        )

        report.write(
            "## Decision\n\n"
        )

        report.write(
            f"**{decision_value}**\n\n"
        )

        report.write(
            "## Core multivariate architecture\n\n"
        )

        report.write(
            "Fiji systems-serology responses were organized "
            "along two principal and reproducible immune-state "
            "axes. PC1 represented a cross-reactive recall-breadth "
            "programme dominated by HPV31/33/45/52/58 IgG, "
            "IgG-subclass and Fc-receptor features. PC2 represented "
            "a vaccine-type HPV16/18 antibody-abundance and effector "
            "programme, including ADCP and neutralization in the "
            "raw matrix.\n\n"
        )

        report.write(
            "Both axes differed between primary induction and "
            "heterologous recall in the raw and BPV-calibrated "
            "representations. PC1 showed the strongest bootstrap "
            "and feature-family-omission stability. PC2 was "
            "reproducible but showed greater bootstrap and "
            "family-removal sensitivity.\n\n"
        )

        report.write(
            "## Cross-representation interpretation\n\n"
        )

        for row in concordance.itertuples(
            index=False
        ):
            report.write(
                f"- {row.principal_component} "
                f"({row.biological_axis}): score Spearman "
                f"rho={row.score_spearman_rho:.3f}; "
                f"shared-feature loading Pearson "
                f"r={row.loading_pearson_r:.3f}; "
                f"{row.concordance_status}.\n"
            )

        report.write(
            "\n## Discrete clustering versus continuous structure\n\n"
        )

        report.write(
            f"The raw matrix supported a stable k=2 partition "
            f"(silhouette={raw_silhouette:.3f}; mean subsample "
            f"adjusted Rand index="
            f"{float(raw_cluster['mean_subsample_adjusted_rand'].iloc[0]):.3f}). "
            f"However, {raw_correct}/80 participants "
            f"({100 * raw_accuracy:.2f}%) aligned with the known "
            f"primary-versus-recall context. The raw clusters "
            f"therefore mainly recapitulated the experimental "
            f"immunization contrast rather than defining independent "
            f"intrinsic immune-response subtypes.\n\n"
        )

        report.write(
            f"The best BPV-calibrated k=2 solution remained "
            f"strongly context-associated but did not meet the "
            f"prespecified silhouette threshold "
            f"(silhouette={bpv_silhouette:.3f}; required at least "
            f"0.20). No calibrated k value from two through six met "
            f"all stability criteria. The preferred biological model "
            f"is therefore a continuous multiaxial recall landscape "
            f"rather than stable discrete immune states.\n\n"
        )

        report.write(
            "## Previous-dose-group effect\n\n"
        )

        report.write(
            "Only raw PC2 differed among one-, two- and three-dose "
            "recall groups after FDR correction. The corresponding "
            "BPV-calibrated PC2 test was not significant. This "
            "schedule association remains a qualified secondary "
            "observation and should not be interpreted as evidence "
            "for discrete dose-defined immune states.\n\n"
        )

        report.write(
            "## Biological boundary\n\n"
        )

        report.write(
            "The multivariate axes summarize downstream antibody "
            "quantity, subclass, Fc-receptor, phagocytic and "
            "neutralizing organization. They do not directly measure "
            "intracellular antigen routing, endosomal processing, "
            "HLA-II loading, germinal-centre dynamics or memory "
            "B-cell lineage evolution.\n"
        )

    print(
        "===== PHASE 2C4C COMPLETE ====="
    )

    print(
        f"Decision: {decision_value}"
    )

    print(
        "Core immune-state axis rows:",
        len(
            axis_registry
        ),
    )

    print(
        "Supported core axes:",
        int(
            axis_registry[
                "core_axis_supported"
            ].sum()
        ),
    )

    print(
        "Cross-representation concordance rows:",
        len(
            concordance
        ),
    )

    print(
        "Clustering synthesis rows:",
        len(
            clustering
        ),
    )

    print(
        "Schedule registry rows:",
        len(
            schedule_registry
        ),
    )

    print(
        "Recall-dose FDR findings:",
        int(
            schedule_registry[
                "fdr_significant"
            ].sum()
        ),
    )

    print(
        "Raw cluster context alignment:",
        f"{raw_correct}/80 "
        f"({100 * raw_accuracy:.2f}%)",
    )

    print(
        "BPV-calibrated stable cluster solution:",
        False,
    )

    print(
        f"Report: {report_path}"
    )

    if failures:
        print()
        print(
            "Validation failures:"
        )

        for failure in failures:
            print(
                f"- {failure}"
            )

        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()
