#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from pandas.api.types import is_bool_dtype


ROOT = Path(
    "/mnt/d/HPV_Vaccine_Trafficome_Project"
)

TABLES = ROOT / "08_results" / "tables"

DECISION_PATH = (
    TABLES
    / "phase2C4B_fiji_pca_stability_clustering_decision.tsv"
)

FAMILY_MANIFEST_PATH = (
    TABLES
    / "phase2C4B_fiji_feature_family_omission_manifest.tsv"
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

CLUSTER_ASSIGNMENTS_PATH = (
    TABLES
    / "phase2C4B_fiji_cluster_assignments.tsv"
)

CLUSTER_PROFILES_PATH = (
    TABLES
    / "phase2C4B_fiji_cluster_profiles.tsv"
)

CLUSTER_MARKERS_PATH = (
    TABLES
    / "phase2C4B_fiji_cluster_feature_markers.tsv"
)

EXPECTED_DECISION = (
    "READY_FOR_PHASE2C4C_IMMUNE_STATE_SYNTHESIS"
)


def require_file(
    path: Path,
) -> None:
    if not path.exists():
        sys.exit(
            f"ERROR: Required input file is missing: {path}"
        )


def read_table(
    path: Path,
    label: str,
) -> pd.DataFrame:
    require_file(
        path
    )

    frame = pd.read_csv(
        path,
        sep="\t",
    )

    if frame.empty:
        sys.exit(
            f"ERROR: {label} is empty: {path}"
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
        sys.exit(
            f"ERROR: {label} is missing columns: "
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
            sys.exit(
                f"ERROR: {label} does not contain "
                f"numeric column {column}."
            )

        frame[
            column
        ] = pd.to_numeric(
            frame[
                column
            ],
            errors="coerce",
        )


def boolean_series(
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


def print_section(
    title: str,
) -> None:
    print()
    print(
        f"===== {title} ====="
    )


def print_frame(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
    sort_columns: list[str] | None = None,
    ascending: list[bool] | bool = True,
) -> None:
    output = frame.copy()

    # Sort before selecting display columns. This permits hidden
    # component-number columns to be used as stable sort keys.
    if sort_columns is not None:
        missing_sort = set(
            sort_columns
        ) - set(
            output.columns
        )

        if missing_sort:
            sys.exit(
                "ERROR: Missing sort columns: "
                + ", ".join(
                    sorted(
                        missing_sort
                    )
                )
            )

        output = output.sort_values(
            sort_columns,
            ascending=ascending,
            na_position="last",
        )

    if columns is not None:
        missing_display = set(
            columns
        ) - set(
            output.columns
        )

        if missing_display:
            sys.exit(
                "ERROR: Missing display columns: "
                + ", ".join(
                    sorted(
                        missing_display
                    )
                )
            )

        output = output[
            columns
        ]

    if output.empty:
        print(
            "No rows."
        )
    else:
        print(
            output.to_string(
                index=False
            )
        )


def main() -> None:
    decision = read_table(
        DECISION_PATH,
        "Phase 2C4B decision table",
    )

    family_manifest = read_table(
        FAMILY_MANIFEST_PATH,
        "Feature-family omission manifest",
    )

    leave_one_family = read_table(
        LEAVE_ONE_FAMILY_PATH,
        "Leave-one-family-out PCA stability table",
    )

    bootstrap_summary = read_table(
        BOOTSTRAP_SUMMARY_PATH,
        "Bootstrap PCA stability summary",
    )

    clustering_metrics = read_table(
        CLUSTER_METRICS_PATH,
        "Clustering metrics table",
    )

    assignments = read_table(
        CLUSTER_ASSIGNMENTS_PATH,
        "Cluster assignments table",
    )

    profiles = read_table(
        CLUSTER_PROFILES_PATH,
        "Cluster profiles table",
    )

    markers = read_table(
        CLUSTER_MARKERS_PATH,
        "Cluster feature-marker table",
    )

    require_columns(
        decision,
        {
            "decision",
            "raw_selected_cluster_count",
            "bpv_selected_cluster_count",
            "raw_stable_cluster_candidates",
            "bpv_stable_cluster_candidates",
        },
        "Phase 2C4B decision table",
    )

    observed_decision = str(
        decision.loc[
            0,
            "decision",
        ]
    )

    if observed_decision != EXPECTED_DECISION:
        sys.exit(
            "ERROR: Phase 2C4B decision is "
            f"{observed_decision}; expected "
            f"{EXPECTED_DECISION}."
        )

    require_columns(
        family_manifest,
        {
            "matrix_representation",
            "omitted_feature_family",
            "original_features",
            "omitted_features",
            "retained_features",
            "reduced_components_for_80_percent_variance",
        },
        "Feature-family omission manifest",
    )

    require_columns(
        leave_one_family,
        {
            "matrix_representation",
            "omitted_feature_family",
            "reference_principal_component",
            "reference_component_number",
            "matched_reduced_principal_component",
            "matched_reduced_component_number",
            "retained_features",
            "absolute_loading_correlation",
            "sign_aligned_score_correlation",
            "reference_explained_variance_ratio",
            "reduced_explained_variance_ratio",
            "absolute_variance_ratio_difference",
        },
        "Leave-one-family-out PCA stability table",
    )

    require_columns(
        bootstrap_summary,
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
        "Bootstrap PCA stability summary",
    )

    require_columns(
        clustering_metrics,
        {
            "matrix_representation",
            "cluster_count",
            "pca_components_used",
            "minimum_cluster_size",
            "maximum_cluster_size",
            "silhouette_score",
            "calinski_harabasz_score",
            "davies_bouldin_score",
            "mean_subsample_adjusted_rand",
            "median_subsample_adjusted_rand",
            "subsample_adjusted_rand_2_5_percentile",
            "subsample_adjusted_rand_97_5_percentile",
            "context_adjusted_rand",
            "context_normalized_mutual_information",
            "dose_group_adjusted_rand",
            "dose_group_normalized_mutual_information",
            "stable_cluster_candidate",
            "clustering_composite_score",
            "candidate_rank_within_representation",
            "selected_cluster_candidate",
        },
        "Clustering metrics table",
    )

    require_columns(
        assignments,
        {
            "matrix_representation",
            "cluster_count",
            "participant_id",
            "previous_4vHPV_doses",
            "analysis_context",
            "cluster_label",
            "PC1_score",
            "PC2_score",
        },
        "Cluster assignments table",
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
            "dose0_participants",
            "dose1_participants",
            "dose2_participants",
            "dose3_participants",
            "mean_PC1_score",
            "mean_PC2_score",
        },
        "Cluster profiles table",
    )

    require_columns(
        markers,
        {
            "matrix_representation",
            "cluster_count",
            "cluster_label",
            "marker_direction",
            "marker_rank",
            "feature_column",
            "antigen_target",
            "assay_feature",
            "feature_family",
            "cluster_mean_standardized_value",
        },
        "Cluster marker table",
    )

    convert_numeric(
        family_manifest,
        [
            "original_features",
            "omitted_features",
            "retained_features",
            "reduced_components_for_80_percent_variance",
        ],
        "Feature-family omission manifest",
    )

    convert_numeric(
        leave_one_family,
        [
            "reference_component_number",
            "matched_reduced_component_number",
            "retained_features",
            "absolute_loading_correlation",
            "sign_aligned_score_correlation",
            "reference_explained_variance_ratio",
            "reduced_explained_variance_ratio",
            "absolute_variance_ratio_difference",
        ],
        "Leave-one-family-out PCA stability table",
    )

    convert_numeric(
        bootstrap_summary,
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
        "Bootstrap PCA stability summary",
    )

    convert_numeric(
        clustering_metrics,
        [
            "cluster_count",
            "pca_components_used",
            "minimum_cluster_size",
            "maximum_cluster_size",
            "silhouette_score",
            "calinski_harabasz_score",
            "davies_bouldin_score",
            "mean_subsample_adjusted_rand",
            "median_subsample_adjusted_rand",
            "subsample_adjusted_rand_2_5_percentile",
            "subsample_adjusted_rand_97_5_percentile",
            "context_adjusted_rand",
            "context_normalized_mutual_information",
            "dose_group_adjusted_rand",
            "dose_group_normalized_mutual_information",
            "clustering_composite_score",
            "candidate_rank_within_representation",
        ],
        "Clustering metrics table",
    )

    convert_numeric(
        assignments,
        [
            "cluster_count",
            "previous_4vHPV_doses",
            "cluster_label",
            "PC1_score",
            "PC2_score",
        ],
        "Cluster assignments table",
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
            "dose0_participants",
            "dose1_participants",
            "dose2_participants",
            "dose3_participants",
            "mean_PC1_score",
            "mean_PC2_score",
        ],
        "Cluster profiles table",
    )

    convert_numeric(
        markers,
        [
            "cluster_count",
            "cluster_label",
            "marker_rank",
            "cluster_mean_standardized_value",
        ],
        "Cluster feature-marker table",
    )

    clustering_metrics[
        "stable_cluster_candidate"
    ] = boolean_series(
        clustering_metrics[
            "stable_cluster_candidate"
        ]
    )

    clustering_metrics[
        "selected_cluster_candidate"
    ] = boolean_series(
        clustering_metrics[
            "selected_cluster_candidate"
        ]
    )

    print_section(
        "PHASE 2C4B DECISION"
    )

    print_frame(
        decision
    )

    print_section(
        "FEATURE-FAMILY OMISSION MANIFEST"
    )

    print_frame(
        family_manifest,
        columns=[
            "matrix_representation",
            "omitted_feature_family",
            "original_features",
            "omitted_features",
            "retained_features",
            "reduced_components_for_80_percent_variance",
        ],
        sort_columns=[
            "matrix_representation",
            "omitted_feature_family",
        ],
    )

    print_section(
        "LEAVE-ONE-FAMILY-OUT STABILITY: PC1-PC2"
    )

    leading_family_stability = leave_one_family[
        leave_one_family[
            "reference_component_number"
        ].isin(
            [
                1,
                2,
            ]
        )
    ].copy()

    print_frame(
        leading_family_stability,
        columns=[
            "matrix_representation",
            "omitted_feature_family",
            "reference_principal_component",
            "matched_reduced_principal_component",
            "absolute_loading_correlation",
            "sign_aligned_score_correlation",
            "absolute_variance_ratio_difference",
        ],
        sort_columns=[
            "matrix_representation",
            "reference_component_number",
            "absolute_loading_correlation",
        ],
        ascending=[
            True,
            True,
            True,
        ],
    )

    print_section(
        "LEAVE-ONE-FAMILY-OUT STABILITY SUMMARY"
    )

    family_summary = (
        leave_one_family.groupby(
            [
                "matrix_representation",
                "reference_principal_component",
                "reference_component_number",
            ],
            observed=True,
        )
        .agg(
            omission_tests=(
                "omitted_feature_family",
                "size",
            ),
            minimum_loading_correlation=(
                "absolute_loading_correlation",
                "min",
            ),
            median_loading_correlation=(
                "absolute_loading_correlation",
                "median",
            ),
            minimum_score_correlation=(
                "sign_aligned_score_correlation",
                "min",
            ),
            median_score_correlation=(
                "sign_aligned_score_correlation",
                "median",
            ),
            maximum_variance_ratio_difference=(
                "absolute_variance_ratio_difference",
                "max",
            ),
        )
        .reset_index()
    )

    print_frame(
        family_summary,
        sort_columns=[
            "matrix_representation",
            "reference_component_number",
        ],
    )

    print_section(
        "BOOTSTRAP PCA STABILITY: PC1-PC5"
    )

    print_frame(
        bootstrap_summary,
        columns=[
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
        ],
        sort_columns=[
            "matrix_representation",
            "reference_component_number",
        ],
    )

    print_section(
        "ALL CLUSTERING CANDIDATES"
    )

    metric_columns = [
        "matrix_representation",
        "cluster_count",
        "pca_components_used",
        "minimum_cluster_size",
        "maximum_cluster_size",
        "silhouette_score",
        "calinski_harabasz_score",
        "davies_bouldin_score",
        "mean_subsample_adjusted_rand",
        "median_subsample_adjusted_rand",
        "subsample_adjusted_rand_2_5_percentile",
        "subsample_adjusted_rand_97_5_percentile",
        "context_adjusted_rand",
        "context_normalized_mutual_information",
        "dose_group_adjusted_rand",
        "dose_group_normalized_mutual_information",
        "stable_cluster_candidate",
        "clustering_composite_score",
        "candidate_rank_within_representation",
        "selected_cluster_candidate",
    ]

    print_frame(
        clustering_metrics,
        columns=metric_columns,
        sort_columns=[
            "matrix_representation",
            "candidate_rank_within_representation",
            "cluster_count",
        ],
    )

    selected_rows = clustering_metrics[
        clustering_metrics[
            "selected_cluster_candidate"
        ]
    ].copy()

    best_rows = (
        clustering_metrics.sort_values(
            [
                "matrix_representation",
                "candidate_rank_within_representation",
                "cluster_count",
            ]
        )
        .groupby(
            "matrix_representation",
            observed=True,
        )
        .head(
            1
        )
        .copy()
    )

    selected_or_best_parts = []

    for representation in sorted(
        clustering_metrics[
            "matrix_representation"
        ].dropna().unique()
    ):
        selected = selected_rows[
            selected_rows[
                "matrix_representation"
            ]
            == representation
        ].copy()

        if not selected.empty:
            selected[
                "solution_status"
            ] = "selected_stable_solution"

            selected_or_best_parts.append(
                selected
            )
        else:
            best = best_rows[
                best_rows[
                    "matrix_representation"
                ]
                == representation
            ].copy()

            best[
                "solution_status"
            ] = "best_available_but_not_stable"

            selected_or_best_parts.append(
                best
            )

    selected_or_best = pd.concat(
        selected_or_best_parts,
        ignore_index=True,
    )

    print_section(
        "SELECTED OR BEST CLUSTERING SOLUTIONS"
    )

    print_frame(
        selected_or_best,
        columns=[
            "matrix_representation",
            "cluster_count",
            "solution_status",
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
            "clustering_composite_score",
        ],
        sort_columns=[
            "matrix_representation",
        ],
    )

    raw_selected_metrics = clustering_metrics[
        (
            clustering_metrics[
                "matrix_representation"
            ]
            == "raw_log2_change"
        )
        & (
            clustering_metrics[
                "selected_cluster_candidate"
            ]
        )
    ].copy()

    print_section(
        "SELECTED RAW CLUSTER PROFILES"
    )

    raw_selected_k: int | None

    if raw_selected_metrics.empty:
        raw_selected_k = None

        print(
            "No selected raw clustering solution."
        )
    else:
        raw_selected_k = int(
            raw_selected_metrics[
                "cluster_count"
            ].iloc[0]
        )

        raw_profiles = profiles[
            (
                profiles[
                    "matrix_representation"
                ]
                == "raw_log2_change"
            )
            & (
                profiles[
                    "cluster_count"
                ]
                == raw_selected_k
            )
        ].copy()

        print_frame(
            raw_profiles,
            columns=[
                "matrix_representation",
                "cluster_count",
                "cluster_label",
                "participants",
                "primary_participants",
                "recall_participants",
                "recall_fraction",
                "dose0_participants",
                "dose1_participants",
                "dose2_participants",
                "dose3_participants",
                "mean_PC1_score",
                "mean_PC2_score",
            ],
            sort_columns=[
                "cluster_label",
            ],
        )

    print_section(
        "RAW SELECTED CLUSTERS BY IMMUNIZATION CONTEXT"
    )

    if raw_selected_k is None:
        print(
            "No selected raw clustering solution."
        )
    else:
        raw_assignments = assignments[
            (
                assignments[
                    "matrix_representation"
                ]
                == "raw_log2_change"
            )
            & (
                assignments[
                    "cluster_count"
                ]
                == raw_selected_k
            )
        ].copy()

        context_counts = pd.crosstab(
            raw_assignments[
                "cluster_label"
            ],
            raw_assignments[
                "analysis_context"
            ],
            margins=True,
        )

        print(
            context_counts.to_string()
        )

        print_section(
            "RAW SELECTED CLUSTERS BY PREVIOUS-DOSE GROUP"
        )

        dose_counts = pd.crosstab(
            raw_assignments[
                "cluster_label"
            ],
            raw_assignments[
                "previous_4vHPV_doses"
            ],
            margins=True,
        )

        print(
            dose_counts.to_string()
        )

        print_section(
            "RAW SELECTED CLUSTER PC1-PC2 DISTRIBUTION"
        )

        raw_score_summary = (
            raw_assignments.groupby(
                "cluster_label",
                observed=True,
            )
            .agg(
                participants=(
                    "participant_id",
                    "size",
                ),
                mean_PC1=(
                    "PC1_score",
                    "mean",
                ),
                median_PC1=(
                    "PC1_score",
                    "median",
                ),
                sd_PC1=(
                    "PC1_score",
                    "std",
                ),
                mean_PC2=(
                    "PC2_score",
                    "mean",
                ),
                median_PC2=(
                    "PC2_score",
                    "median",
                ),
                sd_PC2=(
                    "PC2_score",
                    "std",
                ),
            )
            .reset_index()
        )

        print_frame(
            raw_score_summary,
            sort_columns=[
                "cluster_label",
            ],
        )

    print_section(
        "RAW SELECTED CLUSTER FEATURE MARKERS"
    )

    if raw_selected_k is None:
        print(
            "No selected raw clustering solution."
        )
    else:
        raw_markers = markers[
            (
                markers[
                    "matrix_representation"
                ]
                == "raw_log2_change"
            )
            & (
                markers[
                    "cluster_count"
                ]
                == raw_selected_k
            )
        ].copy()

        print_frame(
            raw_markers,
            columns=[
                "cluster_label",
                "marker_direction",
                "marker_rank",
                "feature_column",
                "antigen_target",
                "assay_feature",
                "feature_family",
                "cluster_mean_standardized_value",
            ],
            sort_columns=[
                "cluster_label",
                "marker_direction",
                "marker_rank",
            ],
        )

    print_section(
        "BPV-CALIBRATED BEST NON-STABLE SOLUTION"
    )

    bpv_best = best_rows[
        best_rows[
            "matrix_representation"
        ]
        == "bpv_calibrated_log2_change"
    ].copy()

    print_frame(
        bpv_best,
        columns=[
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
            "clustering_composite_score",
        ],
    )

    if not bpv_best.empty:
        bpv_best_k = int(
            bpv_best[
                "cluster_count"
            ].iloc[0]
        )

        print_section(
            "BPV-CALIBRATED BEST-SOLUTION PROFILES"
        )

        bpv_profiles = profiles[
            (
                profiles[
                    "matrix_representation"
                ]
                == "bpv_calibrated_log2_change"
            )
            & (
                profiles[
                    "cluster_count"
                ]
                == bpv_best_k
            )
        ].copy()

        print_frame(
            bpv_profiles,
            columns=[
                "cluster_count",
                "cluster_label",
                "participants",
                "primary_participants",
                "recall_participants",
                "recall_fraction",
                "dose0_participants",
                "dose1_participants",
                "dose2_participants",
                "dose3_participants",
                "mean_PC1_score",
                "mean_PC2_score",
            ],
            sort_columns=[
                "cluster_label",
            ],
        )

    raw_stable_count = int(
        (
            (
                clustering_metrics[
                    "matrix_representation"
                ]
                == "raw_log2_change"
            )
            & (
                clustering_metrics[
                    "stable_cluster_candidate"
                ]
            )
        ).sum()
    )

    raw_selected_count = int(
        (
            (
                clustering_metrics[
                    "matrix_representation"
                ]
                == "raw_log2_change"
            )
            & (
                clustering_metrics[
                    "selected_cluster_candidate"
                ]
            )
        ).sum()
    )

    bpv_stable_count = int(
        (
            (
                clustering_metrics[
                    "matrix_representation"
                ]
                == "bpv_calibrated_log2_change"
            )
            & (
                clustering_metrics[
                    "stable_cluster_candidate"
                ]
            )
        ).sum()
    )

    bpv_selected_count = int(
        (
            (
                clustering_metrics[
                    "matrix_representation"
                ]
                == "bpv_calibrated_log2_change"
            )
            & (
                clustering_metrics[
                    "selected_cluster_candidate"
                ]
            )
        ).sum()
    )

    print_section(
        "INSPECTION SUMMARY"
    )

    print(
        "Raw stable cluster candidates:",
        raw_stable_count,
    )

    print(
        "Raw selected solutions:",
        raw_selected_count,
    )

    print(
        "BPV-calibrated stable cluster candidates:",
        bpv_stable_count,
    )

    print(
        "BPV-calibrated selected solutions:",
        bpv_selected_count,
    )

    print(
        "Bootstrap PCA components inspected:",
        len(
            bootstrap_summary
        ),
    )

    print(
        "Leave-one-family-out stability rows:",
        len(
            leave_one_family
        ),
    )

    expected_counts = {
        "raw_stable_cluster_candidates": (
            raw_stable_count,
            1,
        ),
        "raw_selected_solutions": (
            raw_selected_count,
            1,
        ),
        "bpv_stable_cluster_candidates": (
            bpv_stable_count,
            0,
        ),
        "bpv_selected_solutions": (
            bpv_selected_count,
            0,
        ),
        "bootstrap_summary_rows": (
            len(
                bootstrap_summary
            ),
            10,
        ),
        "leave_one_family_rows": (
            len(
                leave_one_family
            ),
            40,
        ),
    }

    failures = []

    for label, (
        observed,
        expected,
    ) in expected_counts.items():
        if observed != expected:
            failures.append(
                f"{label}: expected {expected}, "
                f"observed {observed}."
            )

    if failures:
        print()
        print(
            "Inspection validation failures:"
        )

        for failure in failures:
            print(
                f"- {failure}"
            )

        sys.exit(
            1
        )

    print()
    print(
        "Phase 2C4B evidence inspection: PASS"
    )


if __name__ == "__main__":
    main()
