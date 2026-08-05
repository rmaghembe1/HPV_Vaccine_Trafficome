#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(
    "/mnt/d/HPV_Vaccine_Trafficome_Project"
)

PROCESSED = (
    ROOT
    / "07_data_processed"
    / "fiji_nct02276521"
)

TABLES = ROOT / "08_results" / "tables"

SOURCE_DATA = (
    ROOT
    / "08_results"
    / "figure_source_data"
    / "hpv_specific"
    / "fiji_nct02276521"
    / "phase2C5A"
)

FIGURE_DIR = (
    ROOT
    / "09_figures"
    / "hpv_specific"
    / "fiji_nct02276521"
    / "phase2C5"
)

REPORT_DIR = (
    ROOT
    / "02_dataset_audit"
    / "hpv_specific"
    / "fiji_nct02276521"
)

C4C_DECISION_PATH = (
    TABLES
    / "phase2C4C_fiji_immune_state_synthesis_decision.tsv"
)

SCORES_PATH = (
    PROCESSED
    / "phase2C4A_fiji_pca_scores_long.tsv"
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

BOOTSTRAP_PATH = (
    TABLES
    / "phase2C4B_fiji_bootstrap_pca_stability_summary.tsv"
)

LEAVE_FAMILY_PATH = (
    TABLES
    / "phase2C4B_fiji_leave_one_family_out_pca_stability.tsv"
)

CLUSTER_METRICS_PATH = (
    TABLES
    / "phase2C4B_fiji_clustering_metrics.tsv"
)

CLUSTER_PROFILES_PATH = (
    TABLES
    / "phase2C4B_fiji_cluster_profiles.tsv"
)

CLUSTER_SYNTHESIS_PATH = (
    TABLES
    / "phase2C4C_fiji_clustering_synthesis.tsv"
)

EXPECTED_DECISION = (
    "READY_FOR_PHASE2C4_COMMIT_AND_PHASE2C5_FIGURE_CONSTRUCTION"
)

REPRESENTATION_LABELS = {
    "raw_log2_change": "Raw",
    "bpv_calibrated_log2_change": "BPV-calibrated",
}

REPRESENTATION_COLORS = {
    "raw_log2_change": "#1F77B4",
    "bpv_calibrated_log2_change": "#D62728",
}

CONTEXT_COLORS = {
    "primary_2vHPV_induction": "#5E3C99",
    "heterologous_2vHPV_recall": "#E66101",
}

CONTEXT_LABELS = {
    "primary_2vHPV_induction": "Primary induction",
    "heterologous_2vHPV_recall": "Heterologous recall",
}

DOSE_MARKERS = {
    0: "o",
    1: "s",
    2: "^",
    3: "D",
}

FAMILY_ORDER = [
    "binding_antibody_abundance",
    "igg_subclass_architecture",
    "fc_receptor_communication",
    "phagocytic_function",
    "neutralization",
]

FAMILY_LABELS = {
    "binding_antibody_abundance": "Binding\nabundance",
    "igg_subclass_architecture": "IgG\nsubclasses",
    "fc_receptor_communication": "Fc receptor\ncommunication",
    "phagocytic_function": "ADCP",
    "neutralization": "Neutralization",
}

POSITIVE_COLOR = "#C65D21"
NEGATIVE_COLOR = "#4B3F8C"

RASTER_DPI = 600

MAIN_BASENAME = (
    FIGURE_DIR
    / "Figure_Fiji_multivariate_immune_state_architecture_v1"
)

SUPPLEMENT_BASENAME = (
    FIGURE_DIR
    / "FigureS_Fiji_multivariate_stability_and_loadings_v1"
)


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


def panel_label(
    axis: plt.Axes,
    label: str,
) -> None:
    axis.text(
        -0.12,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
        ha="left",
    )


def style_axis(
    axis: plt.Axes,
) -> None:
    axis.spines[
        "top"
    ].set_visible(
        False
    )

    axis.spines[
        "right"
    ].set_visible(
        False
    )

    axis.tick_params(
        axis="both",
        labelsize=8,
        width=0.8,
    )


def save_figure(
    figure: plt.Figure,
    basename: Path,
) -> list[Path]:
    basename.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_paths = [
        basename.with_suffix(
            ".png"
        ),
        basename.with_suffix(
            ".tiff"
        ),
        basename.with_suffix(
            ".svg"
        ),
    ]

    figure.savefig(
        output_paths[0],
        dpi=RASTER_DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    figure.savefig(
        output_paths[1],
        dpi=RASTER_DPI,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={
            "compression": "tiff_lzw",
        },
    )

    figure.savefig(
        output_paths[2],
        bbox_inches="tight",
        facecolor="white",
    )

    return output_paths


def prepare_scores(
    scores: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        scores,
        {
            "participant_id",
            "previous_4vHPV_doses",
            "analysis_context",
            "matrix_representation",
            "principal_component",
            "component_number",
            "principal_component_score",
        },
        "PCA score table",
    )

    convert_numeric(
        scores,
        [
            "previous_4vHPV_doses",
            "component_number",
            "principal_component_score",
        ],
        "PCA score table",
    )

    core = scores[
        scores[
            "component_number"
        ].isin(
            [
                1,
                2,
            ]
        )
    ].copy()

    wide = (
        core.pivot_table(
            index=[
                "participant_id",
                "previous_4vHPV_doses",
                "analysis_context",
                "matrix_representation",
            ],
            columns="principal_component",
            values="principal_component_score",
            aggfunc="first",
        )
        .reset_index()
    )

    wide.columns.name = None

    require_columns(
        wide,
        {
            "PC1",
            "PC2",
        },
        "Wide PCA score table",
    )

    if len(
        wide
    ) != 160:
        fail(
            f"Expected 160 representation-participant rows, "
            f"observed {len(wide)}"
        )

    return wide


def prepare_context_effects(
    context: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        context,
        {
            "matrix_representation",
            "principal_component",
            "component_number",
            "hedges_g_recall_minus_primary",
            "bh_q_value",
        },
        "Primary-versus-recall PC test table",
    )

    convert_numeric(
        context,
        [
            "component_number",
            "hedges_g_recall_minus_primary",
            "bh_q_value",
        ],
        "Primary-versus-recall PC test table",
    )

    output = context[
        context[
            "component_number"
        ]
        <= 10
    ].copy()

    output[
        "fdr_significant"
    ] = (
        output[
            "bh_q_value"
        ]
        < 0.05
    )

    return output


def prepare_axis_contributions(
    contributions: pd.DataFrame,
) -> pd.DataFrame:
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

    convert_numeric(
        contributions,
        [
            "component_number",
            "contribution_percent",
        ],
        "PCA contribution table",
    )

    core = contributions[
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
                [
                    1,
                    2,
                ]
            )
        )
    ].copy()

    row_order = [
        (
            "raw_log2_change",
            1,
            "Raw PC1",
        ),
        (
            "bpv_calibrated_log2_change",
            1,
            "BPV PC1",
        ),
        (
            "raw_log2_change",
            2,
            "Raw PC2",
        ),
        (
            "bpv_calibrated_log2_change",
            2,
            "BPV PC2",
        ),
    ]

    rows = []

    for (
        representation,
        component_number,
        row_label,
    ) in row_order:
        subset = core[
            (
                core[
                    "matrix_representation"
                ]
                == representation
            )
            & (
                core[
                    "component_number"
                ]
                == component_number
            )
        ]

        value_map = dict(
            zip(
                subset[
                    "annotation_value"
                ],
                subset[
                    "contribution_percent"
                ],
            )
        )

        row = {
            "axis_label": row_label,
            "matrix_representation": representation,
            "component_number": component_number,
        }

        for family in FAMILY_ORDER:
            row[
                family
            ] = float(
                value_map.get(
                    family,
                    0.0,
                )
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def prepare_top_loadings(
    loadings: pd.DataFrame,
    top_n: int = 12,
) -> pd.DataFrame:
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
        },
        "PCA loading table",
    )

    convert_numeric(
        loadings,
        [
            "component_number",
            "loading",
            "absolute_loading",
        ],
        "PCA loading table",
    )

    core = loadings[
        loadings[
            "component_number"
        ].isin(
            [
                1,
                2,
            ]
        )
    ].copy()

    parts = []

    for (
        representation,
        component_number,
    ), group in core.groupby(
        [
            "matrix_representation",
            "component_number",
        ],
        observed=True,
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
            top_n
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

        parts.append(
            selected
        )

    output = pd.concat(
        parts,
        ignore_index=True,
    )

    if len(
        output
    ) != 4 * top_n:
        fail(
            f"Expected {4 * top_n} top-loading rows, "
            f"observed {len(output)}"
        )

    return output


def plot_pca_scatter(
    axis: plt.Axes,
    scores: pd.DataFrame,
    representation: str,
    title: str,
) -> None:
    subset = scores[
        scores[
            "matrix_representation"
        ]
        == representation
    ].copy()

    for context in [
        "primary_2vHPV_induction",
        "heterologous_2vHPV_recall",
    ]:
        context_subset = subset[
            subset[
                "analysis_context"
            ]
            == context
        ]

        for dose in sorted(
            context_subset[
                "previous_4vHPV_doses"
            ].dropna().unique()
        ):
            dose_integer = int(
                dose
            )

            selected = context_subset[
                context_subset[
                    "previous_4vHPV_doses"
                ]
                == dose
            ]

            axis.scatter(
                selected[
                    "PC1"
                ],
                selected[
                    "PC2"
                ],
                s=42,
                marker=DOSE_MARKERS[
                    dose_integer
                ],
                facecolor=CONTEXT_COLORS[
                    context
                ],
                edgecolor="white",
                linewidth=0.7,
                alpha=0.88,
                zorder=3,
            )

    centroids = (
        subset.groupby(
            "analysis_context",
            observed=True,
        )[
            [
                "PC1",
                "PC2",
            ]
        ]
        .mean()
        .reset_index()
    )

    for row in centroids.itertuples(
        index=False
    ):
        axis.scatter(
            row.PC1,
            row.PC2,
            s=150,
            marker="X",
            facecolor=CONTEXT_COLORS[
                row.analysis_context
            ],
            edgecolor="black",
            linewidth=0.9,
            zorder=5,
        )

    axis.axhline(
        0,
        color="#BDBDBD",
        linewidth=0.7,
        zorder=1,
    )

    axis.axvline(
        0,
        color="#BDBDBD",
        linewidth=0.7,
        zorder=1,
    )

    axis.set_xlabel(
        "PC1 score",
        fontsize=9,
    )

    axis.set_ylabel(
        "PC2 score",
        fontsize=9,
    )

    axis.set_title(
        title,
        fontsize=10,
        fontweight="bold",
        pad=8,
    )

    style_axis(
        axis
    )


def plot_context_effects(
    axis: plt.Axes,
    effects: pd.DataFrame,
) -> None:
    offsets = {
        "raw_log2_change": -0.10,
        "bpv_calibrated_log2_change": 0.10,
    }

    for representation in REPRESENTATION_LABELS:
        subset = effects[
            effects[
                "matrix_representation"
            ]
            == representation
        ].sort_values(
            "component_number"
        )

        x_values = (
            subset[
                "component_number"
            ].to_numpy(
                dtype=float
            )
            + offsets[
                representation
            ]
        )

        y_values = subset[
            "hedges_g_recall_minus_primary"
        ].to_numpy(
            dtype=float
        )

        significant = subset[
            "fdr_significant"
        ].to_numpy(
            dtype=bool
        )

        axis.plot(
            x_values,
            y_values,
            color=REPRESENTATION_COLORS[
                representation
            ],
            linewidth=1.3,
            alpha=0.8,
            zorder=2,
        )

        axis.scatter(
            x_values[
                ~significant
            ],
            y_values[
                ~significant
            ],
            s=38,
            facecolor="white",
            edgecolor=REPRESENTATION_COLORS[
                representation
            ],
            linewidth=1.1,
            zorder=3,
        )

        axis.scatter(
            x_values[
                significant
            ],
            y_values[
                significant
            ],
            s=52,
            facecolor=REPRESENTATION_COLORS[
                representation
            ],
            edgecolor="black",
            linewidth=0.6,
            zorder=4,
        )

    axis.axhline(
        0,
        color="black",
        linewidth=0.8,
    )

    axis.set_xticks(
        range(
            1,
            11,
        )
    )

    axis.set_xlabel(
        "Principal component",
        fontsize=9,
    )

    axis.set_ylabel(
        "Hedges g\n(recall minus primary)",
        fontsize=9,
    )

    axis.set_title(
        "Primary versus recall separation",
        fontsize=10,
        fontweight="bold",
        pad=8,
    )

    axis.legend(
        handles=[
            Line2D(
                [
                    0
                ],
                [
                    0
                ],
                color=REPRESENTATION_COLORS[
                    "raw_log2_change"
                ],
                marker="o",
                label="Raw",
            ),
            Line2D(
                [
                    0
                ],
                [
                    0
                ],
                color=REPRESENTATION_COLORS[
                    "bpv_calibrated_log2_change"
                ],
                marker="o",
                label="BPV-calibrated",
            ),
            Line2D(
                [
                    0
                ],
                [
                    0
                ],
                color="black",
                marker="o",
                markerfacecolor="black",
                linestyle="",
                label="FDR < 0.05",
            ),
        ],
        frameon=False,
        fontsize=7,
        loc="best",
    )

    style_axis(
        axis
    )


def plot_contribution_heatmap(
    axis: plt.Axes,
    contribution_matrix: pd.DataFrame,
) -> None:
    values = contribution_matrix[
        FAMILY_ORDER
    ].to_numpy(
        dtype=float
    )

    image = axis.imshow(
        values,
        aspect="auto",
        cmap="YlOrBr",
        vmin=0,
        vmax=max(
            50,
            float(
                np.nanmax(
                    values
                )
            ),
        ),
    )

    axis.set_yticks(
        range(
            len(
                contribution_matrix
            )
        )
    )

    axis.set_yticklabels(
        contribution_matrix[
            "axis_label"
        ],
        fontsize=8,
    )

    axis.set_xticks(
        range(
            len(
                FAMILY_ORDER
            )
        )
    )

    axis.set_xticklabels(
        [
            FAMILY_LABELS[
                family
            ]
            for family in FAMILY_ORDER
        ],
        fontsize=7,
    )

    for row_index in range(
        values.shape[
            0
        ]
    ):
        for column_index in range(
            values.shape[
                1
            ]
        ):
            value = values[
                row_index,
                column_index,
            ]

            text_color = (
                "white"
                if value >= 30
                else "black"
            )

            axis.text(
                column_index,
                row_index,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
            )

    axis.set_title(
        "Feature-family composition of PC1 and PC2",
        fontsize=10,
        fontweight="bold",
        pad=8,
    )

    colorbar = axis.figure.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
    )

    colorbar.set_label(
        "Contribution (%)",
        fontsize=8,
    )

    colorbar.ax.tick_params(
        labelsize=7
    )

    axis.tick_params(
        length=0
    )


def plot_bootstrap_stability(
    axis: plt.Axes,
    bootstrap: pd.DataFrame,
) -> None:
    require_columns(
        bootstrap,
        {
            "matrix_representation",
            "reference_component_number",
            "median_absolute_loading_correlation",
            "loading_correlation_2_5_percentile",
            "loading_correlation_97_5_percentile",
        },
        "Bootstrap stability table",
    )

    convert_numeric(
        bootstrap,
        [
            "reference_component_number",
            "median_absolute_loading_correlation",
            "loading_correlation_2_5_percentile",
            "loading_correlation_97_5_percentile",
        ],
        "Bootstrap stability table",
    )

    offsets = {
        "raw_log2_change": -0.10,
        "bpv_calibrated_log2_change": 0.10,
    }

    for representation in REPRESENTATION_LABELS:
        subset = bootstrap[
            bootstrap[
                "matrix_representation"
            ]
            == representation
        ].sort_values(
            "reference_component_number"
        )

        x_values = (
            subset[
                "reference_component_number"
            ].to_numpy(
                dtype=float
            )
            + offsets[
                representation
            ]
        )

        medians = subset[
            "median_absolute_loading_correlation"
        ].to_numpy(
            dtype=float
        )

        lower = medians - subset[
            "loading_correlation_2_5_percentile"
        ].to_numpy(
            dtype=float
        )

        upper = subset[
            "loading_correlation_97_5_percentile"
        ].to_numpy(
            dtype=float
        ) - medians

        axis.errorbar(
            x_values,
            medians,
            yerr=np.vstack(
                [
                    lower,
                    upper,
                ]
            ),
            fmt="o",
            markersize=5,
            capsize=3,
            linewidth=1.2,
            color=REPRESENTATION_COLORS[
                representation
            ],
            label=REPRESENTATION_LABELS[
                representation
            ],
            zorder=3,
        )

    axis.axhline(
        0.80,
        color="#666666",
        linestyle="--",
        linewidth=0.9,
    )

    axis.axhline(
        0.90,
        color="#999999",
        linestyle=":",
        linewidth=0.9,
    )

    axis.set_ylim(
        0,
        1.03,
    )

    axis.set_xticks(
        range(
            1,
            6,
        )
    )

    axis.set_xlabel(
        "Reference principal component",
        fontsize=9,
    )

    axis.set_ylabel(
        "Absolute loading correlation",
        fontsize=9,
    )

    axis.set_title(
        "Participant-bootstrap PCA stability",
        fontsize=10,
        fontweight="bold",
        pad=8,
    )

    axis.legend(
        frameon=False,
        fontsize=7,
        loc="lower left",
    )

    style_axis(
        axis
    )


def plot_clustering_metrics(
    axis: plt.Axes,
    metrics: pd.DataFrame,
) -> None:
    require_columns(
        metrics,
        {
            "matrix_representation",
            "cluster_count",
            "silhouette_score",
            "mean_subsample_adjusted_rand",
            "selected_cluster_candidate",
        },
        "Clustering metrics",
    )

    convert_numeric(
        metrics,
        [
            "cluster_count",
            "silhouette_score",
            "mean_subsample_adjusted_rand",
        ],
        "Clustering metrics",
    )

    selected_text = (
        metrics[
            "selected_cluster_candidate"
        ]
        .astype(
            "string"
        )
        .str.lower()
    )

    metrics[
        "selected_cluster_candidate"
    ] = selected_text.isin(
        [
            "true",
            "1",
            "yes",
        ]
    )

    for representation in REPRESENTATION_LABELS:
        subset = metrics[
            metrics[
                "matrix_representation"
            ]
            == representation
        ].sort_values(
            "cluster_count"
        )

        sizes = (
            35
            + 95
            * subset[
                "mean_subsample_adjusted_rand"
            ].clip(
                lower=0,
                upper=1,
            )
        )

        axis.plot(
            subset[
                "cluster_count"
            ],
            subset[
                "silhouette_score"
            ],
            color=REPRESENTATION_COLORS[
                representation
            ],
            linewidth=1.5,
            zorder=2,
        )

        axis.scatter(
            subset[
                "cluster_count"
            ],
            subset[
                "silhouette_score"
            ],
            s=sizes,
            color=REPRESENTATION_COLORS[
                representation
            ],
            edgecolor="white",
            linewidth=0.7,
            alpha=0.85,
            zorder=3,
        )

        selected = subset[
            subset[
                "selected_cluster_candidate"
            ]
        ]

        if not selected.empty:
            axis.scatter(
                selected[
                    "cluster_count"
                ],
                selected[
                    "silhouette_score"
                ],
                s=190,
                marker="*",
                color="#FFD700",
                edgecolor="black",
                linewidth=0.8,
                zorder=5,
            )

    axis.axhline(
        0.20,
        color="black",
        linestyle="--",
        linewidth=0.9,
        label="Silhouette threshold",
    )

    axis.set_xticks(
        [
            2,
            3,
            4,
            5,
            6,
        ]
    )

    axis.set_xlabel(
        "Candidate cluster count",
        fontsize=9,
    )

    axis.set_ylabel(
        "Silhouette score",
        fontsize=9,
    )

    axis.set_title(
        "Discrete clustering evaluation",
        fontsize=10,
        fontweight="bold",
        pad=8,
    )

    axis.text(
        0.02,
        0.04,
        "Marker size = subsample ARI\nStar = selected stable solution",
        transform=axis.transAxes,
        fontsize=7,
        va="bottom",
        ha="left",
    )

    axis.legend(
        handles=[
            Line2D(
                [
                    0
                ],
                [
                    0
                ],
                color=REPRESENTATION_COLORS[
                    "raw_log2_change"
                ],
                marker="o",
                label="Raw",
            ),
            Line2D(
                [
                    0
                ],
                [
                    0
                ],
                color=REPRESENTATION_COLORS[
                    "bpv_calibrated_log2_change"
                ],
                marker="o",
                label="BPV-calibrated",
            ),
            Line2D(
                [
                    0
                ],
                [
                    0
                ],
                color="black",
                linestyle="--",
                label="Threshold = 0.20",
            ),
        ],
        frameon=False,
        fontsize=7,
        loc="upper right",
    )

    style_axis(
        axis
    )


def plot_loading_panel(
    axis: plt.Axes,
    top_loadings: pd.DataFrame,
    representation: str,
    component_number: int,
    title: str,
) -> None:
    subset = top_loadings[
        (
            top_loadings[
                "matrix_representation"
            ]
            == representation
        )
        & (
            top_loadings[
                "component_number"
            ]
            == component_number
        )
    ].copy()

    subset = subset.sort_values(
        "loading"
    )

    labels = (
        subset[
            "feature_column"
        ]
        .str.replace(
            "__",
            " ",
            regex=False,
        )
        .tolist()
    )

    colors = [
        POSITIVE_COLOR
        if value >= 0
        else NEGATIVE_COLOR
        for value in subset[
            "loading"
        ]
    ]

    y_positions = np.arange(
        len(
            subset
        )
    )

    axis.barh(
        y_positions,
        subset[
            "loading"
        ],
        color=colors,
        edgecolor="white",
        linewidth=0.4,
    )

    axis.axvline(
        0,
        color="black",
        linewidth=0.8,
    )

    axis.set_yticks(
        y_positions
    )

    axis.set_yticklabels(
        labels,
        fontsize=7,
    )

    axis.set_xlabel(
        "PCA loading",
        fontsize=8,
    )

    axis.set_title(
        title,
        fontsize=10,
        fontweight="bold",
        pad=8,
    )

    style_axis(
        axis
    )


def plot_leave_family(
    axis: plt.Axes,
    leave_family: pd.DataFrame,
) -> None:
    require_columns(
        leave_family,
        {
            "matrix_representation",
            "omitted_feature_family",
            "reference_component_number",
            "sign_aligned_score_correlation",
        },
        "Leave-one-family-out stability table",
    )

    convert_numeric(
        leave_family,
        [
            "reference_component_number",
            "sign_aligned_score_correlation",
        ],
        "Leave-one-family-out stability table",
    )

    core = leave_family[
        leave_family[
            "reference_component_number"
        ].isin(
            [
                1,
                2,
            ]
        )
    ].copy()

    family_labels = sorted(
        core[
            "omitted_feature_family"
        ].unique()
    )

    x_map = {
        family: index
        for index, family in enumerate(
            family_labels
        )
    }

    offsets = {
        (
            "raw_log2_change",
            1,
        ): -0.18,
        (
            "raw_log2_change",
            2,
        ): -0.06,
        (
            "bpv_calibrated_log2_change",
            1,
        ): 0.06,
        (
            "bpv_calibrated_log2_change",
            2,
        ): 0.18,
    }

    marker_map = {
        1: "o",
        2: "s",
    }

    for row in core.itertuples(
        index=False
    ):
        x_value = (
            x_map[
                row.omitted_feature_family
            ]
            + offsets[
                (
                    row.matrix_representation,
                    int(
                        row.reference_component_number
                    ),
                )
            ]
        )

        axis.scatter(
            x_value,
            row.sign_aligned_score_correlation,
            s=52,
            marker=marker_map[
                int(
                    row.reference_component_number
                )
            ],
            color=REPRESENTATION_COLORS[
                row.matrix_representation
            ],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )

    axis.axhline(
        0.75,
        color="#666666",
        linestyle="--",
        linewidth=0.9,
    )

    axis.set_xticks(
        range(
            len(
                family_labels
            )
        )
    )

    axis.set_xticklabels(
        [
            FAMILY_LABELS.get(
                family,
                family.replace(
                    "_",
                    " "
                ),
            )
            for family in family_labels
        ],
        rotation=35,
        ha="right",
        fontsize=7,
    )

    axis.set_ylim(
        -0.10,
        1.04,
    )

    axis.set_ylabel(
        "Sign-aligned score correlation",
        fontsize=8,
    )

    axis.set_title(
        "Leave-one-feature-family-out stability",
        fontsize=10,
        fontweight="bold",
        pad=8,
    )

    axis.legend(
        handles=[
            Line2D(
                [
                    0
                ],
                [
                    0
                ],
                marker="o",
                linestyle="",
                color="#555555",
                label="PC1",
            ),
            Line2D(
                [
                    0
                ],
                [
                    0
                ],
                marker="s",
                linestyle="",
                color="#555555",
                label="PC2",
            ),
            Patch(
                facecolor=REPRESENTATION_COLORS[
                    "raw_log2_change"
                ],
                label="Raw",
            ),
            Patch(
                facecolor=REPRESENTATION_COLORS[
                    "bpv_calibrated_log2_change"
                ],
                label="BPV-calibrated",
            ),
        ],
        frameon=False,
        fontsize=7,
        ncol=2,
        loc="lower left",
    )

    style_axis(
        axis
    )


def plot_cluster_composition(
    axis: plt.Axes,
    profiles: pd.DataFrame,
    synthesis: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        synthesis,
        {
            "matrix_representation",
            "cluster_count",
        },
        "Clustering synthesis table",
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
        },
        "Cluster profile table",
    )

    convert_numeric(
        synthesis,
        [
            "cluster_count",
        ],
        "Clustering synthesis table",
    )

    convert_numeric(
        profiles,
        [
            "cluster_count",
            "cluster_label",
            "participants",
            "primary_participants",
            "recall_participants",
        ],
        "Cluster profile table",
    )

    rows = []

    for representation in REPRESENTATION_LABELS:
        selected = synthesis[
            synthesis[
                "matrix_representation"
            ]
            == representation
        ]

        if len(
            selected
        ) != 1:
            fail(
                f"Expected one clustering synthesis row for "
                f"{representation}"
            )

        cluster_count = int(
            selected[
                "cluster_count"
            ].iloc[
                0
            ]
        )

        subset = profiles[
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

        for row in subset.itertuples(
            index=False
        ):
            total = float(
                row.participants
            )

            rows.append(
                {
                    "matrix_representation": representation,
                    "cluster_count": cluster_count,
                    "cluster_label": int(
                        row.cluster_label
                    ),
                    "participants": int(
                        row.participants
                    ),
                    "primary_participants": int(
                        row.primary_participants
                    ),
                    "recall_participants": int(
                        row.recall_participants
                    ),
                    "primary_fraction": (
                        row.primary_participants
                        / total
                    ),
                    "recall_fraction": (
                        row.recall_participants
                        / total
                    ),
                    "display_label": (
                        f"{REPRESENTATION_LABELS[representation]}\n"
                        f"C{int(row.cluster_label)}"
                    ),
                }
            )

    composition = pd.DataFrame(
        rows
    )

    x_values = np.arange(
        len(
            composition
        )
    )

    axis.bar(
        x_values,
        composition[
            "primary_fraction"
        ],
        color=CONTEXT_COLORS[
            "primary_2vHPV_induction"
        ],
        label="Primary induction",
    )

    axis.bar(
        x_values,
        composition[
            "recall_fraction"
        ],
        bottom=composition[
            "primary_fraction"
        ],
        color=CONTEXT_COLORS[
            "heterologous_2vHPV_recall"
        ],
        label="Heterologous recall",
    )

    for index, row in composition.iterrows():
        axis.text(
            index,
            1.02,
            f"n={int(row['participants'])}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    axis.set_xticks(
        x_values
    )

    axis.set_xticklabels(
        composition[
            "display_label"
        ],
        fontsize=7,
    )

    axis.set_ylim(
        0,
        1.12,
    )

    axis.set_ylabel(
        "Participant fraction",
        fontsize=8,
    )

    axis.set_title(
        "Cluster composition by immunization context",
        fontsize=10,
        fontweight="bold",
        pad=8,
    )

    axis.legend(
        frameon=False,
        fontsize=7,
        loc="center left",
        bbox_to_anchor=(
            1.00,
            0.50,
        ),
    )

    style_axis(
        axis
    )

    return composition


def create_main_figure(
    scores: pd.DataFrame,
    effects: pd.DataFrame,
    contributions: pd.DataFrame,
    bootstrap: pd.DataFrame,
    cluster_metrics: pd.DataFrame,
) -> plt.Figure:
    figure = plt.figure(
        figsize=(
            15.2,
            9.6,
        ),
        constrained_layout=True,
    )

    grid = figure.add_gridspec(
        2,
        3,
        width_ratios=[
            1.0,
            1.0,
            1.05,
        ],
        height_ratios=[
            1.0,
            0.95,
        ],
    )

    axes = [
        figure.add_subplot(
            grid[
                0,
                0,
            ]
        ),
        figure.add_subplot(
            grid[
                0,
                1,
            ]
        ),
        figure.add_subplot(
            grid[
                0,
                2,
            ]
        ),
        figure.add_subplot(
            grid[
                1,
                0,
            ]
        ),
        figure.add_subplot(
            grid[
                1,
                1,
            ]
        ),
        figure.add_subplot(
            grid[
                1,
                2,
            ]
        ),
    ]

    plot_pca_scatter(
        axes[
            0
        ],
        scores,
        "raw_log2_change",
        "Raw systems-serology PCA",
    )

    plot_pca_scatter(
        axes[
            1
        ],
        scores,
        "bpv_calibrated_log2_change",
        "BPV-calibrated PCA",
    )

    context_legend = [
        Patch(
            facecolor=CONTEXT_COLORS[
                context
            ],
            label=CONTEXT_LABELS[
                context
            ],
        )
        for context in CONTEXT_LABELS
    ]

    dose_legend = [
        Line2D(
            [
                0
            ],
            [
                0
            ],
            marker=DOSE_MARKERS[
                dose
            ],
            linestyle="",
            markerfacecolor="#777777",
            markeredgecolor="white",
            markersize=7,
            label=f"Previous dose {dose}",
        )
        for dose in [
            0,
            1,
            2,
            3,
        ]
    ]

    axes[
        1
    ].legend(
        handles=context_legend
        + dose_legend,
        frameon=False,
        fontsize=7,
        ncol=2,
        loc="upper right",
    )

    plot_context_effects(
        axes[
            2
        ],
        effects,
    )

    plot_contribution_heatmap(
        axes[
            3
        ],
        contributions,
    )

    plot_bootstrap_stability(
        axes[
            4
        ],
        bootstrap,
    )

    plot_clustering_metrics(
        axes[
            5
        ],
        cluster_metrics,
    )

    for axis, label in zip(
        axes,
        [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
        ],
    ):
        panel_label(
            axis,
            label,
        )

    figure.suptitle(
        (
            "Fiji HPV systems-serology reveals a stable two-axis "
            "continuous immune-state architecture"
        ),
        fontsize=15,
        fontweight="bold",
    )

    return figure


def create_supplementary_figure(
    top_loadings: pd.DataFrame,
    leave_family: pd.DataFrame,
    cluster_profiles: pd.DataFrame,
    cluster_synthesis: pd.DataFrame,
) -> tuple[
    plt.Figure,
    pd.DataFrame,
]:
    figure = plt.figure(
        figsize=(
            15.2,
            10.8,
        ),
        constrained_layout=True,
    )

    grid = figure.add_gridspec(
        2,
        3,
        width_ratios=[
            1.0,
            1.0,
            1.05,
        ],
    )

    axes = [
        figure.add_subplot(
            grid[
                0,
                0,
            ]
        ),
        figure.add_subplot(
            grid[
                0,
                1,
            ]
        ),
        figure.add_subplot(
            grid[
                0,
                2,
            ]
        ),
        figure.add_subplot(
            grid[
                1,
                0,
            ]
        ),
        figure.add_subplot(
            grid[
                1,
                1,
            ]
        ),
        figure.add_subplot(
            grid[
                1,
                2,
            ]
        ),
    ]

    plot_loading_panel(
        axes[
            0
        ],
        top_loadings,
        "raw_log2_change",
        1,
        "Raw PC1 leading features",
    )

    plot_loading_panel(
        axes[
            1
        ],
        top_loadings,
        "bpv_calibrated_log2_change",
        1,
        "BPV-calibrated PC1 leading features",
    )

    plot_loading_panel(
        axes[
            2
        ],
        top_loadings,
        "raw_log2_change",
        2,
        "Raw PC2 leading features",
    )

    plot_loading_panel(
        axes[
            3
        ],
        top_loadings,
        "bpv_calibrated_log2_change",
        2,
        "BPV-calibrated PC2 leading features",
    )

    plot_leave_family(
        axes[
            4
        ],
        leave_family,
    )

    cluster_composition = plot_cluster_composition(
        axes[
            5
        ],
        cluster_profiles,
        cluster_synthesis,
    )

    for axis, label in zip(
        axes,
        [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
        ],
    ):
        panel_label(
            axis,
            label,
        )

    figure.suptitle(
        (
            "Loading architecture, feature-family stability and "
            "context-dominated clustering"
        ),
        fontsize=15,
        fontweight="bold",
    )

    return (
        figure,
        cluster_composition,
    )


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    decision = read_table(
        C4C_DECISION_PATH,
        "Phase 2C4C decision table",
    )

    require_columns(
        decision,
        {
            "decision",
        },
        "Phase 2C4C decision table",
    )

    observed_decision = str(
        decision.loc[
            0,
            "decision",
        ]
    )

    if observed_decision != EXPECTED_DECISION:
        fail(
            f"Unexpected Phase 2C4C decision: "
            f"{observed_decision}"
        )

    raw_scores = read_table(
        SCORES_PATH,
        "PCA score table",
    )

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

    context_tests = read_table(
        CONTEXT_TESTS_PATH,
        "Primary-versus-recall PC tests",
    )

    bootstrap = read_table(
        BOOTSTRAP_PATH,
        "Bootstrap PCA stability table",
    )

    leave_family = read_table(
        LEAVE_FAMILY_PATH,
        "Leave-one-feature-family-out table",
    )

    cluster_metrics = read_table(
        CLUSTER_METRICS_PATH,
        "Clustering metrics table",
    )

    cluster_profiles = read_table(
        CLUSTER_PROFILES_PATH,
        "Cluster profile table",
    )

    cluster_synthesis = read_table(
        CLUSTER_SYNTHESIS_PATH,
        "Clustering synthesis table",
    )

    scores = prepare_scores(
        raw_scores
    )

    effects = prepare_context_effects(
        context_tests
    )

    contribution_matrix = (
        prepare_axis_contributions(
            contributions
        )
    )

    top_loadings = prepare_top_loadings(
        loadings,
        top_n=12,
    )

    require_columns(
        variance,
        {
            "matrix_representation",
            "component_number",
            "explained_variance_percent",
            "cumulative_variance_percent",
        },
        "PCA variance table",
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

    main_figure = create_main_figure(
        scores,
        effects,
        contribution_matrix,
        bootstrap,
        cluster_metrics,
    )

    (
        supplementary_figure,
        cluster_composition,
    ) = create_supplementary_figure(
        top_loadings,
        leave_family,
        cluster_profiles,
        cluster_synthesis,
    )

    main_paths = save_figure(
        main_figure,
        MAIN_BASENAME,
    )

    supplementary_paths = save_figure(
        supplementary_figure,
        SUPPLEMENT_BASENAME,
    )

    plt.close(
        main_figure
    )

    plt.close(
        supplementary_figure
    )

    SOURCE_DATA.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_tables = {
        "phase2C5A_main_pca_scores.tsv": scores,
        "phase2C5A_main_context_effects.tsv": effects,
        "phase2C5A_main_axis_contributions.tsv": (
            contribution_matrix
        ),
        "phase2C5A_main_bootstrap_stability.tsv": (
            bootstrap
        ),
        "phase2C5A_main_clustering_metrics.tsv": (
            cluster_metrics
        ),
        "phase2C5A_supplementary_top_loadings.tsv": (
            top_loadings
        ),
        "phase2C5A_supplementary_leave_family_stability.tsv": (
            leave_family
        ),
        "phase2C5A_supplementary_cluster_composition.tsv": (
            cluster_composition
        ),
    }

    for filename, frame in source_tables.items():
        write_tsv(
            frame,
            SOURCE_DATA
            / filename,
        )

    all_figure_paths = (
        main_paths
        + supplementary_paths
    )

    missing_outputs = [
        str(
            path
        )
        for path in all_figure_paths
        if not path.exists()
    ]

    empty_outputs = [
        str(
            path
        )
        for path in all_figure_paths
        if path.exists()
        and path.stat().st_size == 0
    ]

    failures = []

    if missing_outputs:
        failures.append(
            "Missing figure outputs: "
            + ", ".join(
                missing_outputs
            )
        )

    if empty_outputs:
        failures.append(
            "Empty figure outputs: "
            + ", ".join(
                empty_outputs
            )
        )

    if len(
        scores
    ) != 160:
        failures.append(
            f"Expected 160 PCA scatter rows; "
            f"observed {len(scores)}."
        )

    if len(
        effects
    ) != 20:
        failures.append(
            f"Expected 20 context-effect rows; "
            f"observed {len(effects)}."
        )

    if len(
        contribution_matrix
    ) != 4:
        failures.append(
            "Expected four axis-contribution rows."
        )

    if len(
        top_loadings
    ) != 48:
        failures.append(
            f"Expected 48 top-loading rows; "
            f"observed {len(top_loadings)}."
        )

    if len(
        bootstrap
    ) != 10:
        failures.append(
            f"Expected 10 bootstrap summary rows; "
            f"observed {len(bootstrap)}."
        )

    if len(
        cluster_metrics
    ) != 10:
        failures.append(
            f"Expected 10 clustering metric rows; "
            f"observed {len(cluster_metrics)}."
        )

    decision_value = (
        "READY_FOR_PHASE2C5B_FIGURE_QA_AND_MANUSCRIPT_INTEGRATION"
        if not failures
        else "PHASE2C5A_REPAIR_REQUIRED"
    )

    manifest_rows = []

    for path in all_figure_paths:
        manifest_rows.append(
            {
                "figure_file": str(
                    path.relative_to(
                        ROOT
                    )
                ),
                "format": (
                    path.suffix.lstrip(
                        "."
                    ).lower()
                ),
                "file_size_bytes": (
                    path.stat().st_size
                    if path.exists()
                    else np.nan
                ),
                "raster_dpi": (
                    RASTER_DPI
                    if path.suffix.lower()
                    in {
                        ".png",
                        ".tiff",
                    }
                    else np.nan
                ),
                "editable_vector": (
                    path.suffix.lower()
                    == ".svg"
                ),
            }
        )

    manifest = pd.DataFrame(
        manifest_rows
    )

    manifest_path = (
        TABLES
        / "phase2C5A_fiji_figure_output_manifest.tsv"
    )

    write_tsv(
        manifest,
        manifest_path,
    )

    decision_frame = pd.DataFrame(
        [
            {
                "decision": decision_value,
                "main_figure_files": len(
                    main_paths
                ),
                "supplementary_figure_files": len(
                    supplementary_paths
                ),
                "total_figure_files": len(
                    all_figure_paths
                ),
                "figure_source_data_tables": len(
                    source_tables
                ),
                "pca_score_rows": len(
                    scores
                ),
                "context_effect_rows": len(
                    effects
                ),
                "axis_contribution_rows": len(
                    contribution_matrix
                ),
                "top_loading_rows": len(
                    top_loadings
                ),
                "bootstrap_summary_rows": len(
                    bootstrap
                ),
                "clustering_metric_rows": len(
                    cluster_metrics
                ),
                "raster_dpi": RASTER_DPI,
                "validation_failures": "; ".join(
                    failures
                ),
            }
        ]
    )

    decision_path = (
        TABLES
        / "phase2C5A_fiji_figure_construction_decision.tsv"
    )

    write_tsv(
        decision_frame,
        decision_path,
    )

    report_path = (
        REPORT_DIR
        / "phase2C5A_fiji_multivariate_figure_construction_report.md"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as report:
        report.write(
            "# Phase 2C5A Fiji multivariate figure construction\n\n"
        )

        report.write(
            "## Decision\n\n"
        )

        report.write(
            f"**{decision_value}**\n\n"
        )

        report.write(
            "## Main figure\n\n"
        )

        report.write(
            "The main six-panel figure presents raw and "
            "BPV-calibrated PC1-PC2 score architecture, "
            "primary-versus-recall PC effect sizes, feature-family "
            "composition of PC1 and PC2, participant-bootstrap "
            "component stability and clustering evaluation across "
            "k=2 through k=6.\n\n"
        )

        report.write(
            "## Supplementary figure\n\n"
        )

        report.write(
            "The supplementary six-panel figure presents the leading "
            "signed loadings for raw and BPV-calibrated PC1 and PC2, "
            "leave-one-feature-family-out score stability and cluster "
            "composition by known immunization context.\n\n"
        )

        report.write(
            "## Biological message\n\n"
        )

        report.write(
            "The figures emphasize a reproducible two-axis continuous "
            "immune-state architecture. Raw k=2 clustering is shown as "
            "a context-dominated primary-versus-recall partition rather "
            "than as evidence of intrinsic immune-response subtypes. "
            "No BPV-calibrated cluster solution met all prespecified "
            "criteria.\n\n"
        )

        report.write(
            "## Output formats\n\n"
        )

        report.write(
            f"- PNG: {RASTER_DPI} dpi\n"
        )

        report.write(
            f"- TIFF: {RASTER_DPI} dpi with LZW compression\n"
        )

        report.write(
            "- SVG: editable vector output with text retained as text\n"
        )

        report.write(
            f"- Figure source-data tables: {len(source_tables)}\n"
        )

    print(
        "===== PHASE 2C5A COMPLETE ====="
    )

    print(
        f"Decision: {decision_value}"
    )

    print(
        "Main figure files:",
        len(
            main_paths
        ),
    )

    print(
        "Supplementary figure files:",
        len(
            supplementary_paths
        ),
    )

    print(
        "Total figure files:",
        len(
            all_figure_paths
        ),
    )

    print(
        "Figure source-data tables:",
        len(
            source_tables
        ),
    )

    print(
        "PCA score rows:",
        len(
            scores
        ),
    )

    print(
        "Context-effect rows:",
        len(
            effects
        ),
    )

    print(
        "Axis-contribution rows:",
        len(
            contribution_matrix
        ),
    )

    print(
        "Top-loading rows:",
        len(
            top_loadings
        ),
    )

    print(
        "Bootstrap summary rows:",
        len(
            bootstrap
        ),
    )

    print(
        "Clustering metric rows:",
        len(
            cluster_metrics
        ),
    )

    print(
        "Figure directory:",
        FIGURE_DIR,
    )

    print(
        "Report:",
        report_path,
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
