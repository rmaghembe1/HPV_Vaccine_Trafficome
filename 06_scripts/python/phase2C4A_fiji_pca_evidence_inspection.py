#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(
    "/mnt/d/HPV_Vaccine_Trafficome_Project"
)

TABLES = ROOT / "08_results" / "tables"

VARIANCE_PATH = (
    TABLES
    / "phase2C4A_fiji_pca_variance.tsv"
)

LOADINGS_PATH = (
    TABLES
    / "phase2C4A_fiji_pca_loadings_top10.tsv"
)

EXTREMES_PATH = (
    TABLES
    / "phase2C4A_fiji_pca_loading_extremes.tsv"
)

CONTRIBUTIONS_PATH = (
    TABLES
    / "phase2C4A_fiji_pca_axis_contributions.tsv"
)

CENTROIDS_PATH = (
    TABLES
    / "phase2C4A_fiji_pca_centroids.tsv"
)

CONTEXT_PATH = (
    TABLES
    / "phase2C4A_fiji_primary_vs_recall_pc_tests.tsv"
)

DOSE_PATH = (
    TABLES
    / "phase2C4A_fiji_recall_dose_pc_global_tests.tsv"
)

DECISION_PATH = (
    TABLES
    / "phase2C4A_fiji_pca_architecture_decision.tsv"
)

EXPECTED_DECISION = (
    "READY_FOR_PHASE2C4B_PCA_STABILITY_AND_CLUSTERING"
)


def require_file(
    path: Path,
) -> None:
    if not path.exists():
        sys.exit(
            f"ERROR: Required input file is missing: {path}"
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
            f"ERROR: {label} is missing required columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
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


def main() -> None:
    decision = read_table(
        DECISION_PATH,
        "Phase 2C4A decision table",
    )

    require_columns(
        decision,
        {
            "decision",
        },
        "Phase 2C4A decision table",
    )

    observed_decision = str(
        decision.loc[
            0,
            "decision",
        ]
    )

    if observed_decision != EXPECTED_DECISION:
        sys.exit(
            "ERROR: Phase 2C4A decision is "
            f"{observed_decision}; expected "
            f"{EXPECTED_DECISION}."
        )

    variance = read_table(
        VARIANCE_PATH,
        "PCA variance table",
    )

    loadings = read_table(
        LOADINGS_PATH,
        "PCA loading table",
    )

    extremes = read_table(
        EXTREMES_PATH,
        "PCA loading-extreme table",
    )

    contributions = read_table(
        CONTRIBUTIONS_PATH,
        "PCA contribution table",
    )

    centroids = read_table(
        CENTROIDS_PATH,
        "PCA centroid table",
    )

    context = read_table(
        CONTEXT_PATH,
        "Primary-versus-recall PC test table",
    )

    dose = read_table(
        DOSE_PATH,
        "Recall-dose PC test table",
    )

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
        loadings,
        {
            "matrix_representation",
            "principal_component",
            "component_number",
            "feature_column",
            "antigen_target",
            "assay_feature",
            "feature_family",
            "loading",
            "absolute_loading",
            "contribution_percent",
        },
        "PCA loading table",
    )

    require_columns(
        extremes,
        {
            "matrix_representation",
            "principal_component",
            "component_number",
            "loading_direction",
            "loading_rank",
            "feature_column",
            "loading",
        },
        "PCA loading-extreme table",
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
        centroids,
        {
            "matrix_representation",
            "principal_component",
            "component_number",
            "analysis_stratum",
            "participants",
            "centroid_score",
            "median_score",
            "score_standard_deviation",
        },
        "PCA centroid table",
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
            "primary_median",
            "recall_median",
            "median_difference_recall_minus_primary",
            "rank_biserial_recall_minus_primary",
            "hedges_g_recall_minus_primary",
            "mann_whitney_p_value",
            "bh_q_value",
        },
        "Primary-versus-recall PC test table",
    )

    require_columns(
        dose,
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
        "Recall-dose PC test table",
    )

    numeric_columns = [
        (
            variance,
            [
                "component_number",
                "explained_variance_percent",
                "cumulative_variance_percent",
            ],
        ),
        (
            loadings,
            [
                "component_number",
                "loading",
                "absolute_loading",
                "contribution_percent",
            ],
        ),
        (
            contributions,
            [
                "component_number",
                "contribution_percent",
            ],
        ),
        (
            centroids,
            [
                "component_number",
                "participants",
                "centroid_score",
                "median_score",
                "score_standard_deviation",
            ],
        ),
        (
            context,
            [
                "component_number",
                "mann_whitney_p_value",
                "bh_q_value",
            ],
        ),
        (
            dose,
            [
                "component_number",
                "kruskal_wallis_p_value",
                "bh_q_value",
            ],
        ),
    ]

    for frame, columns in numeric_columns:
        for column in columns:
            frame[
                column
            ] = pd.to_numeric(
                frame[
                    column
                ],
                errors="coerce",
            )

    if variance[
        "component_number"
    ].isna().any():
        sys.exit(
            "ERROR: Invalid component numbers in variance table."
        )

    if context[
        "bh_q_value"
    ].isna().any():
        sys.exit(
            "ERROR: Missing context-test q-values."
        )

    if dose[
        "bh_q_value"
    ].isna().any():
        sys.exit(
            "ERROR: Missing recall-dose-test q-values."
        )

    print(
        "===== PHASE 2C4A INSPECTION INPUTS ====="
    )

    print(
        "Decision:",
        observed_decision,
    )

    print(
        "Variance rows:",
        len(
            variance
        ),
    )

    print(
        "Loading rows:",
        len(
            loadings
        ),
    )

    print(
        "Loading-extreme rows:",
        len(
            extremes
        ),
    )

    print(
        "Contribution rows:",
        len(
            contributions
        ),
    )

    print(
        "Centroid rows:",
        len(
            centroids
        ),
    )

    print(
        "Context-test rows:",
        len(
            context
        ),
    )

    print(
        "Recall-dose-test rows:",
        len(
            dose
        ),
    )

    print(
        "\n===== VARIANCE ARCHITECTURE: PC1-PC10 ====="
    )

    variance_top = variance.loc[
        variance[
            "component_number"
        ]
        <= 10,
        [
            "matrix_representation",
            "principal_component",
            "component_number",
            "explained_variance_percent",
            "cumulative_variance_percent",
        ],
    ].copy()

    print(
        variance_top.sort_values(
            [
                "matrix_representation",
                "component_number",
            ]
        ).to_string(
            index=False
        )
    )

    print(
        "\n===== PRIMARY-VERSUS-RECALL PC DIFFERENCES ====="
    )

    context_supported = context.loc[
        context[
            "bh_q_value"
        ]
        < 0.05
    ].copy()

    context_columns = [
        "matrix_representation",
        "principal_component",
        "component_number",
        "primary_mean",
        "recall_mean",
        "mean_difference_recall_minus_primary",
        "primary_median",
        "recall_median",
        "median_difference_recall_minus_primary",
        "rank_biserial_recall_minus_primary",
        "hedges_g_recall_minus_primary",
        "mann_whitney_p_value",
        "bh_q_value",
    ]

    if context_supported.empty:
        print(
            "No primary-versus-recall PCs survived FDR correction."
        )
    else:
        print(
            context_supported[
                context_columns
            ]
            .sort_values(
                [
                    "matrix_representation",
                    "bh_q_value",
                    "component_number",
                ]
            )
            .to_string(
                index=False
            )
        )

    print(
        "\n===== RECALL-DOSE PC DIFFERENCES ====="
    )

    dose_supported = dose.loc[
        dose[
            "bh_q_value"
        ]
        < 0.05
    ].copy()

    dose_columns = [
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
    ]

    if dose_supported.empty:
        print(
            "No recall-dose PCs survived FDR correction."
        )
    else:
        print(
            dose_supported[
                dose_columns
            ]
            .sort_values(
                [
                    "matrix_representation",
                    "bh_q_value",
                    "component_number",
                ]
            )
            .to_string(
                index=False
            )
        )

    significant_keys = pd.concat(
        [
            context_supported[
                [
                    "matrix_representation",
                    "principal_component",
                    "component_number",
                ]
            ],
            dose_supported[
                [
                    "matrix_representation",
                    "principal_component",
                    "component_number",
                ]
            ],
        ],
        ignore_index=True,
    ).drop_duplicates()

    print(
        "\n===== SUPPORTED COMPONENT INVENTORY ====="
    )

    if significant_keys.empty:
        print(
            "No supported components were identified."
        )
    else:
        print(
            significant_keys.sort_values(
                [
                    "matrix_representation",
                    "component_number",
                ]
            ).to_string(
                index=False
            )
        )

    print(
        "\n===== TOP ABSOLUTE LOADINGS FOR SUPPORTED PCs ====="
    )

    supported_loadings = loadings.merge(
        significant_keys,
        on=[
            "matrix_representation",
            "principal_component",
            "component_number",
        ],
        how="inner",
        validate="many_to_one",
    )

    top_loading_rows: list[
        pd.DataFrame
    ] = []

    for _, group in supported_loadings.groupby(
        [
            "matrix_representation",
            "principal_component",
            "component_number",
        ],
        observed=True,
        dropna=False,
    ):
        selected = group.sort_values(
            [
                "absolute_loading",
                "feature_column",
            ],
            ascending=[
                False,
                True,
            ],
        ).head(
            15
        ).copy()

        selected[
            "absolute_loading_rank"
        ] = range(
            1,
            len(
                selected
            )
            + 1,
        )

        top_loading_rows.append(
            selected
        )

    if top_loading_rows:
        top_loadings = pd.concat(
            top_loading_rows,
            ignore_index=True,
        )

        print(
            top_loadings[
                [
                    "matrix_representation",
                    "principal_component",
                    "component_number",
                    "absolute_loading_rank",
                    "feature_column",
                    "antigen_target",
                    "assay_feature",
                    "feature_family",
                    "loading",
                    "absolute_loading",
                    "contribution_percent",
                ]
            ]
            .sort_values(
                [
                    "matrix_representation",
                    "component_number",
                    "absolute_loading_rank",
                ]
            )
            .to_string(
                index=False
            )
        )
    else:
        print(
            "No supported-PC loading rows were found."
        )

    print(
        "\n===== POSITIVE AND NEGATIVE LOADING EXTREMES ====="
    )

    supported_extremes = extremes.merge(
        significant_keys,
        on=[
            "matrix_representation",
            "principal_component",
            "component_number",
        ],
        how="inner",
        validate="many_to_one",
    )

    if supported_extremes.empty:
        print(
            "No loading extremes were found for supported PCs."
        )
    else:
        print(
            supported_extremes.sort_values(
                [
                    "matrix_representation",
                    "component_number",
                    "loading_direction",
                    "loading_rank",
                ]
            ).to_string(
                index=False
            )
        )

    print(
        "\n===== FEATURE-FAMILY CONTRIBUTIONS ====="
    )

    family_contributions = contributions.loc[
        contributions[
            "annotation_level"
        ]
        == "feature_family"
    ].merge(
        significant_keys,
        on=[
            "matrix_representation",
            "principal_component",
            "component_number",
        ],
        how="inner",
        validate="many_to_one",
    )

    if family_contributions.empty:
        print(
            "No feature-family contributions were found."
        )
    else:
        print(
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
            ).to_string(
                index=False
            )
        )

    print(
        "\n===== ANTIGEN CONTRIBUTIONS ====="
    )

    antigen_contributions = contributions.loc[
        contributions[
            "annotation_level"
        ]
        == "antigen_target"
    ].merge(
        significant_keys,
        on=[
            "matrix_representation",
            "principal_component",
            "component_number",
        ],
        how="inner",
        validate="many_to_one",
    )

    if antigen_contributions.empty:
        print(
            "No antigen contributions were found."
        )
    else:
        print(
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
            ).to_string(
                index=False
            )
        )

    print(
        "\n===== GROUP CENTROIDS FOR SUPPORTED PCs ====="
    )

    supported_centroids = centroids.merge(
        significant_keys,
        on=[
            "matrix_representation",
            "principal_component",
            "component_number",
        ],
        how="inner",
        validate="many_to_one",
    )

    if supported_centroids.empty:
        print(
            "No centroids were found for supported PCs."
        )
    else:
        print(
            supported_centroids[
                [
                    "matrix_representation",
                    "principal_component",
                    "component_number",
                    "analysis_stratum",
                    "participants",
                    "centroid_score",
                    "median_score",
                    "score_standard_deviation",
                ]
            ]
            .sort_values(
                [
                    "matrix_representation",
                    "component_number",
                    "analysis_stratum",
                ]
            )
            .to_string(
                index=False
            )
        )

    print(
        "\n===== ALL PC1-PC10 CONTEXT TESTS ====="
    )

    print(
        context[
            context_columns
        ]
        .sort_values(
            [
                "matrix_representation",
                "component_number",
            ]
        )
        .to_string(
            index=False
        )
    )

    print(
        "\n===== ALL PC1-PC10 RECALL-DOSE TESTS ====="
    )

    print(
        dose[
            dose_columns
        ]
        .sort_values(
            [
                "matrix_representation",
                "component_number",
            ]
        )
        .to_string(
            index=False
        )
    )

    print(
        "\n===== INSPECTION SUMMARY ====="
    )

    print(
        "Primary-versus-recall FDR PCs:",
        len(
            context_supported
        ),
    )

    print(
        "Recall-dose FDR PCs:",
        len(
            dose_supported
        ),
    )

    print(
        "Unique supported representation-PC combinations:",
        len(
            significant_keys
        ),
    )

    print(
        "Supported loading rows:",
        len(
            supported_loadings
        ),
    )

    print(
        "Supported centroid rows:",
        len(
            supported_centroids
        ),
    )

    print(
        "\nPhase 2C4A evidence inspection: PASS"
    )


if __name__ == "__main__":
    main()
