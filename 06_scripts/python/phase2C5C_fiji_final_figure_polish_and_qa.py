#!/usr/bin/env python3

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(
    "/mnt/d/HPV_Vaccine_Trafficome_Project"
)

PROCESSED = (
    ROOT
    / "07_data_processed"
    / "fiji_nct02276521"
)

TABLES = (
    ROOT
    / "08_results"
    / "tables"
)

SOURCE_DATA = (
    ROOT
    / "08_results"
    / "figure_source_data"
    / "hpv_specific"
    / "fiji_nct02276521"
    / "phase2C5C"
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

MANUSCRIPT_DIR = (
    ROOT
    / "10_manuscript"
    / "hpv_specific"
    / "fiji_nct02276521"
)

C5A_DECISION_PATH = (
    TABLES
    / "phase2C5A_fiji_figure_construction_decision.tsv"
)

C4C_AXIS_REGISTRY_PATH = (
    TABLES
    / "phase2C4C_fiji_core_immune_state_axis_registry.tsv"
)

C4C_CLUSTERING_PATH = (
    TABLES
    / "phase2C4C_fiji_clustering_synthesis.tsv"
)

SCORES_PATH = (
    PROCESSED
    / "phase2C4A_fiji_pca_scores_long.tsv"
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

EXPECTED_C5A_DECISION = (
    "READY_FOR_PHASE2C5B_FIGURE_QA_AND_MANUSCRIPT_INTEGRATION"
)

MAIN_BASENAME = (
    "Figure_Fiji_multivariate_immune_state_architecture_v2"
)

SUPPLEMENT_BASENAME = (
    "FigureS_Fiji_multivariate_stability_and_loadings_v2"
)

MAIN_OUTPUT = (
    FIGURE_DIR
    / MAIN_BASENAME
)

SUPPLEMENT_OUTPUT = (
    FIGURE_DIR
    / SUPPLEMENT_BASENAME
)

CONTACT_SHEET_PATH = (
    FIGURE_DIR
    / "phase2C5C_fiji_final_figure_QA_contact_sheet.png"
)

QA_MANIFEST_PATH = (
    TABLES
    / "phase2C5C_fiji_final_figure_QA_manifest.tsv"
)

LEGEND_REGISTRY_PATH = (
    TABLES
    / "phase2C5C_fiji_final_figure_legend_registry.tsv"
)

DECISION_PATH = (
    TABLES
    / "phase2C5C_fiji_final_figure_decision.tsv"
)

REPORT_PATH = (
    REPORT_DIR
    / "phase2C5C_fiji_final_figure_polish_and_QA_report.md"
)

LEGENDS_PATH = (
    MANUSCRIPT_DIR
    / "phase2C5C_fiji_final_multivariate_figure_legends.md"
)

REPRESENTATIONS = [
    "raw_log2_change",
    "bpv_calibrated_log2_change",
]

REPRESENTATION_LABELS = {
    "raw_log2_change": "Raw",
    "bpv_calibrated_log2_change": "BPV-calibrated",
}

REPRESENTATION_COLORS = {
    "raw_log2_change": "#0072B2",
    "bpv_calibrated_log2_change": "#D55E00",
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

SHORT_FAMILY_LABELS = {
    "binding_antibody_abundance": "Binding",
    "igg_subclass_architecture": "IgG subclass",
    "fc_receptor_communication": "Fc receptor",
    "phagocytic_function": "ADCP",
    "neutralization": "Neutralization",
}

POSITIVE_COLOR = "#C65D21"
NEGATIVE_COLOR = "#4B3F8C"

RASTER_DPI = 600

MIN_RASTER_WIDTH = 3000
MIN_RASTER_HEIGHT = 1800
MIN_RASTER_DPI = 590.0
MIN_GRAYSCALE_SD = 5.0
MIN_CONTENT_FRACTION = 0.20
MAX_CONTENT_FRACTION = 0.99


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


def panel_label(
    axis: plt.Axes,
    label: str,
) -> None:
    axis.text(
        -0.13,
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
            f"Expected 160 PCA score rows, "
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
        axis_label,
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

        mapping = dict(
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
            "axis_label": axis_label,
            "matrix_representation": representation,
            "component_number": component_number,
        }

        for family in FAMILY_ORDER:
            row[
                family
            ] = float(
                mapping.get(
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
    ) != 48:
        fail(
            f"Expected 48 top-loading rows, "
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

    for context in CONTEXT_LABELS:
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
            dose_int = int(
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
                s=43,
                marker=DOSE_MARKERS[
                    dose_int
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
            s=155,
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

    for representation in REPRESENTATIONS:
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
            alpha=0.85,
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
            s=54,
            facecolor=REPRESENTATION_COLORS[
                representation
            ],
            edgecolor="black",
            linewidth=0.7,
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
                marker="o",
                linestyle="",
                markerfacecolor="#777777",
                markeredgecolor="black",
                label="Filled symbol: FDR < 0.05",
            ),
        ],
        frameon=False,
        fontsize=7.3,
        loc="upper right",
    )

    style_axis(
        axis
    )


def plot_contribution_heatmap(
    axis: plt.Axes,
    contribution_matrix: pd.DataFrame,
) -> None:
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    values = contribution_matrix[
        FAMILY_ORDER
    ].to_numpy(
        dtype=float
    )

    row_count = int(
        values.shape[
            0
        ]
    )

    column_count = int(
        values.shape[
            1
        ]
    )

    maximum = max(
        50.0,
        float(
            np.nanmax(
                values
            )
        ),
    )

    color_map = plt.get_cmap(
        "YlOrBr"
    )

    normalization = Normalize(
        vmin=0.0,
        vmax=maximum,
    )

    for row_index in range(
        row_count
    ):
        for column_index in range(
            column_count
        ):
            value = float(
                values[
                    row_index,
                    column_index,
                ]
            )

            cell = Rectangle(
                (
                    column_index
                    - 0.5,
                    row_index
                    - 0.5,
                ),
                1.0,
                1.0,
                facecolor=color_map(
                    normalization(
                        value
                    )
                ),
                edgecolor="white",
                linewidth=0.8,
            )

            axis.add_patch(
                cell
            )

            axis.text(
                column_index,
                row_index,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=7.4,
                fontweight=(
                    "bold"
                    if value >= 30
                    else "normal"
                ),
                color=(
                    "white"
                    if value >= 30
                    else "black"
                ),
            )

    axis.set_xlim(
        -0.5,
        column_count
        - 0.5,
    )

    axis.set_ylim(
        row_count
        - 0.5,
        -0.5,
    )

    axis.set_aspect(
        "auto"
    )

    axis.set_yticks(
        range(
            row_count
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
            column_count
        )
    )

    axis.set_xticklabels(
        [
            FAMILY_LABELS[
                family
            ]
            for family in FAMILY_ORDER
        ],
        fontsize=7.5,
    )

    axis.set_title(
        "Feature-family composition of PC1 and PC2 (%)",
        fontsize=10,
        fontweight="bold",
        pad=8,
    )

    axis.text(
        1.0,
        -0.19,
        "Cell values are contribution percentages",
        transform=axis.transAxes,
        fontsize=7.2,
        ha="right",
        va="top",
    )

    axis.tick_params(
        length=0
    )

    for spine in axis.spines.values():
        spine.set_visible(
            False
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

    for representation in REPRESENTATIONS:
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

    metrics = metrics.copy()

    metrics[
        "selected_cluster_candidate"
    ] = as_bool(
        metrics[
            "selected_cluster_candidate"
        ]
    )

    for representation in REPRESENTATIONS:
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
            alpha=0.88,
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
        0.03,
        0.05,
        (
            "Marker size: subsample ARI\n"
            "Star: selected stable solution"
        ),
        transform=axis.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.85,
            "pad": 2.5,
        },
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
                label="Silhouette threshold",
            ),
        ],
        frameon=False,
        fontsize=7.2,
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
        fontsize=8,
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
) -> pd.DataFrame:
    require_columns(
        leave_family,
        {
            "matrix_representation",
            "omitted_feature_family",
            "reference_component_number",
            "sign_aligned_score_correlation",
        },
        "Leave-one-feature-family-out stability table",
    )

    convert_numeric(
        leave_family,
        [
            "reference_component_number",
            "sign_aligned_score_correlation",
        ],
        "Leave-one-feature-family-out stability table",
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

    family_order = [
        "binding_antibody_abundance",
        "fc_receptor_communication",
        "igg_subclass_architecture",
        "neutralization",
        "phagocytic_function",
    ]

    x_map = {
        family: index
        for index, family in enumerate(
            family_order
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
        family = str(
            row.omitted_feature_family
        )

        x_value = (
            x_map[
                family
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
            s=58,
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
                family_order
            )
        )
    )

    axis.set_xticklabels(
        [
            SHORT_FAMILY_LABELS[
                family
            ]
            for family in family_order
        ],
        rotation=28,
        ha="right",
        fontsize=8,
    )

    axis.set_ylim(
        0.72,
        1.02,
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

    axis.text(
        0.98,
        0.04,
        (
            "BPV-calibrated matrix:\n"
            "ADCP and neutralization not present"
        ),
        transform=axis.transAxes,
        fontsize=7.5,
        ha="right",
        va="bottom",
        bbox={
            "facecolor": "white",
            "edgecolor": "#BDBDBD",
            "linewidth": 0.6,
            "alpha": 0.90,
            "pad": 3,
        },
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
        loc="upper left",
    )

    style_axis(
        axis
    )

    return core


def build_cluster_composition(
    profiles: pd.DataFrame,
    synthesis: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        synthesis,
        {
            "matrix_representation",
            "cluster_count",
            "primary_dominant_cluster",
            "recall_dominant_cluster",
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
            "primary_dominant_cluster",
            "recall_dominant_cluster",
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

    for representation in REPRESENTATIONS:
        synthesis_row = synthesis[
            synthesis[
                "matrix_representation"
            ]
            == representation
        ]

        if len(
            synthesis_row
        ) != 1:
            fail(
                f"Expected one synthesis row for {representation}"
            )

        synthesis_record = synthesis_row.iloc[
            0
        ]

        cluster_count = int(
            synthesis_record[
                "cluster_count"
            ]
        )

        primary_cluster = int(
            synthesis_record[
                "primary_dominant_cluster"
            ]
        )

        recall_cluster = int(
            synthesis_record[
                "recall_dominant_cluster"
            ]
        )

        representation_profiles = profiles[
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

        ordered_clusters = [
            (
                primary_cluster,
                "primary-dominant",
            ),
            (
                recall_cluster,
                "recall-dominant",
            ),
        ]

        for cluster_label, state_label in ordered_clusters:
            selected = representation_profiles[
                representation_profiles[
                    "cluster_label"
                ]
                == cluster_label
            ]

            if len(
                selected
            ) != 1:
                fail(
                    f"Expected one profile for {representation}, "
                    f"cluster {cluster_label}"
                )

            row = selected.iloc[
                0
            ]

            participants = int(
                row[
                    "participants"
                ]
            )

            primary_participants = int(
                row[
                    "primary_participants"
                ]
            )

            recall_participants = int(
                row[
                    "recall_participants"
                ]
            )

            rows.append(
                {
                    "matrix_representation": representation,
                    "cluster_count": cluster_count,
                    "cluster_label": cluster_label,
                    "state_label": state_label,
                    "participants": participants,
                    "primary_participants": primary_participants,
                    "recall_participants": recall_participants,
                    "primary_fraction": (
                        primary_participants
                        / participants
                    ),
                    "recall_fraction": (
                        recall_participants
                        / participants
                    ),
                    "display_label": (
                        f"{REPRESENTATION_LABELS[representation]}\n"
                        f"{state_label}"
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def plot_cluster_composition(
    axis: plt.Axes,
    composition: pd.DataFrame,
) -> None:
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
            1.015,
            f"n={int(row['participants'])}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    axis.set_xticks(
        x_values
    )

    axis.set_xticklabels(
        composition[
            "display_label"
        ],
        fontsize=7.5,
    )

    axis.set_ylim(
        0,
        1.11,
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
        fontsize=7.2,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(
            0.50,
            1.02,
        ),
    )

    style_axis(
        axis
    )


def create_main_figure(
    scores: pd.DataFrame,
    effects: pd.DataFrame,
    contributions: pd.DataFrame,
    bootstrap: pd.DataFrame,
    cluster_metrics: pd.DataFrame,
) -> plt.Figure:
    figure = plt.figure(
        figsize=(
            15.5,
            10.0,
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

    shared_legend = [
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

    shared_legend.extend(
        [
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
                label=(
                    "No previous 4vHPV"
                    if dose == 0
                    else f"{dose} previous 4vHPV dose"
                    + (
                        ""
                        if dose == 1
                        else "s"
                    )
                ),
            )
            for dose in [
                0,
                1,
                2,
                3,
            ]
        ]
    )

    figure.legend(
        handles=shared_legend,
        frameon=False,
        fontsize=7.4,
        ncol=6,
        loc="upper center",
        bbox_to_anchor=(
            0.50,
            0.945,
        ),
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
    cluster_composition: pd.DataFrame,
) -> plt.Figure:
    figure = plt.figure(
        figsize=(
            15.5,
            11.0,
        ),
        constrained_layout=True,
    )

    grid = figure.add_gridspec(
        2,
        3,
        width_ratios=[
            1.0,
            1.0,
            1.10,
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

    plot_cluster_composition(
        axes[
            5
        ],
        cluster_composition,
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

    return figure


def save_figure(
    figure: plt.Figure,
    basename: Path,
) -> list[Path]:
    basename.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = basename.with_suffix(
        ".png"
    )

    tiff_path = basename.with_suffix(
        ".tiff"
    )

    svg_path = basename.with_suffix(
        ".svg"
    )

    figure.savefig(
        png_path,
        dpi=RASTER_DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    figure.savefig(
        tiff_path,
        dpi=RASTER_DPI,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={
            "compression": "tiff_lzw",
        },
    )

    figure.savefig(
        svg_path,
        bbox_inches="tight",
        facecolor="white",
    )

    return [
        png_path,
        tiff_path,
        svg_path,
    ]


def raster_content_metrics(
    image: Image.Image,
) -> dict[str, float]:
    preview = image.convert(
        "RGB"
    )

    preview.thumbnail(
        (
            1200,
            1200,
        ),
        Image.Resampling.LANCZOS,
    )

    array = np.asarray(
        preview,
        dtype=np.uint8,
    )

    grayscale = (
        0.299
        * array[
            :,
            :,
            0
        ]
        + 0.587
        * array[
            :,
            :,
            1
        ]
        + 0.114
        * array[
            :,
            :,
            2
        ]
    )

    nonwhite = np.any(
        array
        < 248,
        axis=2,
    )

    if np.any(
        nonwhite
    ):
        coordinates = np.argwhere(
            nonwhite
        )

        minimum_y, minimum_x = coordinates.min(
            axis=0
        )

        maximum_y, maximum_x = coordinates.max(
            axis=0
        )

        bounding_fraction = float(
            (
                maximum_x
                - minimum_x
                + 1
            )
            * (
                maximum_y
                - minimum_y
                + 1
            )
            / (
                preview.width
                * preview.height
            )
        )

    else:
        bounding_fraction = 0.0

    return {
        "grayscale_standard_deviation": float(
            np.std(
                grayscale
            )
        ),
        "nonwhite_pixel_fraction": float(
            np.mean(
                nonwhite
            )
        ),
        "content_bounding_fraction": (
            bounding_fraction
        ),
    }


def inspect_raster(
    path: Path,
) -> dict[str, object]:
    with Image.open(
        path
    ) as image:
        image.load()

        width, height = image.size

        dpi_value = image.info.get(
            "dpi",
            (
                np.nan,
                np.nan,
            ),
        )

        if isinstance(
            dpi_value,
            tuple,
        ):
            x_dpi = float(
                dpi_value[
                    0
                ]
            )

            y_dpi = float(
                dpi_value[
                    1
                ]
            )

        else:
            x_dpi = float(
                dpi_value
            )

            y_dpi = float(
                dpi_value
            )

        metrics = raster_content_metrics(
            image
        )

        image_format = str(
            image.format
        )

        image_mode = str(
            image.mode
        )

    qa_pass = bool(
        width >= MIN_RASTER_WIDTH
        and height >= MIN_RASTER_HEIGHT
        and np.isfinite(
            x_dpi
        )
        and np.isfinite(
            y_dpi
        )
        and x_dpi >= MIN_RASTER_DPI
        and y_dpi >= MIN_RASTER_DPI
        and metrics[
            "grayscale_standard_deviation"
        ]
        >= MIN_GRAYSCALE_SD
        and metrics[
            "content_bounding_fraction"
        ]
        >= MIN_CONTENT_FRACTION
        and metrics[
            "content_bounding_fraction"
        ]
        <= MAX_CONTENT_FRACTION
    )

    return {
        "figure_file": str(
            path.relative_to(
                ROOT
            )
        ),
        "file_type": path.suffix.lower().lstrip(
            "."
        ),
        "file_size_bytes": int(
            path.stat().st_size
        ),
        "image_format": image_format,
        "image_mode": image_mode,
        "width_pixels": int(
            width
        ),
        "height_pixels": int(
            height
        ),
        "x_dpi": x_dpi,
        "y_dpi": y_dpi,
        "grayscale_standard_deviation": (
            metrics[
                "grayscale_standard_deviation"
            ]
        ),
        "nonwhite_pixel_fraction": (
            metrics[
                "nonwhite_pixel_fraction"
            ]
        ),
        "content_bounding_fraction": (
            metrics[
                "content_bounding_fraction"
            ]
        ),
        "editable_vector": False,
        "svg_text_elements": np.nan,
        "svg_embedded_images": np.nan,
        "qa_pass": qa_pass,
    }


def count_svg_elements(
    root: ET.Element,
    local_name: str,
) -> int:
    count = 0

    for element in root.iter():
        tag = str(
            element.tag
        )

        if (
            tag == local_name
            or tag.endswith(
                f"}}{local_name}"
            )
        ):
            count += 1

    return count


def inspect_svg(
    path: Path,
) -> dict[str, object]:
    tree = ET.parse(
        path
    )

    root = tree.getroot()

    text_count = count_svg_elements(
        root,
        "text",
    )

    image_count = count_svg_elements(
        root,
        "image",
    )

    viewbox_present = bool(
        root.attrib.get(
            "viewBox"
        )
    )

    qa_pass = bool(
        path.stat().st_size
        > 10000
        and text_count
        > 20
        and image_count
        == 0
        and viewbox_present
    )

    return {
        "figure_file": str(
            path.relative_to(
                ROOT
            )
        ),
        "file_type": "svg",
        "file_size_bytes": int(
            path.stat().st_size
        ),
        "image_format": "SVG",
        "image_mode": "",
        "width_pixels": np.nan,
        "height_pixels": np.nan,
        "x_dpi": np.nan,
        "y_dpi": np.nan,
        "grayscale_standard_deviation": np.nan,
        "nonwhite_pixel_fraction": np.nan,
        "content_bounding_fraction": np.nan,
        "editable_vector": True,
        "svg_text_elements": int(
            text_count
        ),
        "svg_embedded_images": int(
            image_count
        ),
        "svg_viewbox_present": viewbox_present,
        "qa_pass": qa_pass,
    }


def make_contact_sheet(
    main_png: Path,
    supplementary_png: Path,
    output_path: Path,
) -> None:
    images = []

    for path in [
        main_png,
        supplementary_png,
    ]:
        with Image.open(
            path
        ) as image:
            preview = image.convert(
                "RGB"
            )

            preview.thumbnail(
                (
                    2300,
                    1500,
                ),
                Image.Resampling.LANCZOS,
            )

            images.append(
                (
                    path.name,
                    preview.copy(),
                )
            )

    margin = 60
    heading_height = 55
    gap = 70

    width = max(
        image.width
        for _, image in images
    ) + 2 * margin

    height = (
        sum(
            image.height
            for _, image in images
        )
        + 2 * heading_height
        + gap
        + 2 * margin
    )

    sheet = Image.new(
        "RGB",
        (
            width,
            height,
        ),
        "white",
    )

    draw = ImageDraw.Draw(
        sheet
    )

    font = ImageFont.load_default()

    y_position = margin

    for filename, image in images:
        draw.text(
            (
                margin,
                y_position,
            ),
            filename,
            fill="black",
            font=font,
        )

        y_position += heading_height

        x_position = (
            width
            - image.width
        ) // 2

        sheet.paste(
            image,
            (
                x_position,
                y_position,
            ),
        )

        y_position += (
            image.height
            + gap
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sheet.save(
        output_path,
        format="PNG",
        optimize=True,
    )


def build_legends() -> pd.DataFrame:
    main_legend = (
        "Figure X. Fiji HPV systems-serology reveals a stable "
        "two-axis continuous immune-state architecture. "
        "(A-B) Principal-component score maps for the raw and "
        "BPV-calibrated systems-serology matrices. Each point "
        "represents one participant. Color distinguishes primary "
        "2vHPV induction from heterologous 2vHPV recall, marker shape "
        "identifies the number of previous 4vHPV doses, and X symbols "
        "indicate immunization-context centroids. "
        "(C) Standardized primary-versus-recall effect sizes across "
        "PC1-PC10. Filled symbols denote components significant after "
        "within-representation false-discovery rate correction. Raw and "
        "BPV-calibrated PC1 and PC2 distinguished primary induction from "
        "recall. (D) Feature-family contributions to PC1 and PC2. PC1 "
        "was organized principally by IgG-subclass and Fc-receptor "
        "communication features, whereas PC2 emphasized vaccine-type "
        "binding-antibody abundance and, in the raw matrix, phagocytic "
        "and neutralizing function. (E) Participant-bootstrap stability "
        "of PC1-PC5, expressed as absolute correlations between reference "
        "and bootstrap loading vectors following component matching. "
        "Points show medians and error bars show the 2.5th to 97.5th "
        "percentiles across 300 bootstrap replicates. PC1 was the most "
        "stable component in both representations, whereas PC2 showed "
        "moderate-to-strong reproducibility. (F) Evaluation of candidate "
        "k-means partitions from k=2 to k=6. Marker size represents mean "
        "subsampling adjusted Rand index. The raw k=2 solution met the "
        "prespecified silhouette, stability and cluster-size criteria, "
        "but aligned 77 of 80 participants with the known "
        "primary-versus-recall context. No BPV-calibrated solution met "
        "all prespecified criteria. The preferred interpretation is "
        "therefore a continuous two-axis immune-state landscape rather "
        "than stable intrinsic antibody-response subtypes."
    )

    supplementary_legend = (
        "Supplementary Figure X. Loading architecture, feature-family "
        "stability and context composition of Fiji multivariate "
        "systems-serology responses. (A-B) Twelve features with the "
        "largest absolute loadings on raw and BPV-calibrated PC1. "
        "Positive PC1 loadings were dominated by cross-reactive "
        "HPV31/33/45/52/58 IgG, IgG-subclass and Fc-receptor features, "
        "whereas HPV16 vaccine-type features generally occupied the "
        "opposing loading direction. (C-D) Twelve features with the "
        "largest absolute loadings on raw and BPV-calibrated PC2. Raw "
        "PC2 combined HPV16/18 binding-antibody abundance, IgG-subclass "
        "features, ADCP and neutralization; calibrated PC2 retained a "
        "predominantly HPV16/18 binding and Fc-receptor architecture. "
        "(E) Sign-aligned score correlations after separately omitting "
        "each feature family. The dashed horizontal line marks a score "
        "correlation of 0.75. ADCP and neutralization were not present in "
        "the BPV-calibrated matrix. PC1 and PC2 remained reproducible "
        "after removal of individual feature families, although PC2 was "
        "more sensitive than PC1. (F) Immunization-context composition "
        "of primary-dominant and recall-dominant clusters for the selected "
        "raw k=2 solution and the best available BPV-calibrated k=2 "
        "solution. Both partitions were strongly context-associated. "
        "The calibrated solution did not meet the prespecified silhouette "
        "threshold and was not accepted as a stable discrete immune-state "
        "partition."
    )

    return pd.DataFrame(
        [
            {
                "figure_identifier": (
                    "Fiji_multivariate_main_figure"
                ),
                "figure_file_stem": (
                    MAIN_BASENAME
                ),
                "figure_role": (
                    "main_figure_candidate"
                ),
                "panel_count": 6,
                "legend_text": (
                    main_legend
                ),
            },
            {
                "figure_identifier": (
                    "Fiji_multivariate_supplementary_figure"
                ),
                "figure_file_stem": (
                    SUPPLEMENT_BASENAME
                ),
                "figure_role": (
                    "supplementary_figure_candidate"
                ),
                "panel_count": 6,
                "legend_text": (
                    supplementary_legend
                ),
            },
        ]
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
        C5A_DECISION_PATH,
        "Phase 2C5A decision table",
    )

    require_columns(
        decision,
        {
            "decision",
        },
        "Phase 2C5A decision table",
    )

    observed_decision = str(
        decision.loc[
            0,
            "decision",
        ]
    )

    if observed_decision != EXPECTED_C5A_DECISION:
        fail(
            f"Unexpected Phase 2C5A decision: "
            f"{observed_decision}"
        )

    axis_registry = read_table(
        C4C_AXIS_REGISTRY_PATH,
        "Core immune-state axis registry",
    )

    clustering_synthesis = read_table(
        C4C_CLUSTERING_PATH,
        "Clustering synthesis table",
    )

    scores = prepare_scores(
        read_table(
            SCORES_PATH,
            "PCA score table",
        )
    )

    effects = prepare_context_effects(
        read_table(
            CONTEXT_TESTS_PATH,
            "Primary-versus-recall PC tests",
        )
    )

    contribution_matrix = (
        prepare_axis_contributions(
            read_table(
                CONTRIBUTIONS_PATH,
                "PCA contribution table",
            )
        )
    )

    top_loadings = prepare_top_loadings(
        read_table(
            LOADINGS_PATH,
            "PCA loading table",
        )
    )

    bootstrap = read_table(
        BOOTSTRAP_PATH,
        "Bootstrap stability table",
    )

    leave_family = read_table(
        LEAVE_FAMILY_PATH,
        "Leave-one-family-out stability table",
    )

    cluster_metrics = read_table(
        CLUSTER_METRICS_PATH,
        "Clustering metrics table",
    )

    cluster_profiles = read_table(
        CLUSTER_PROFILES_PATH,
        "Cluster profile table",
    )

    cluster_composition = build_cluster_composition(
        cluster_profiles,
        clustering_synthesis,
    )

    main_figure = create_main_figure(
        scores,
        effects,
        contribution_matrix,
        bootstrap,
        cluster_metrics,
    )

    supplementary_figure = create_supplementary_figure(
        top_loadings,
        leave_family,
        cluster_composition,
    )

    main_paths = save_figure(
        main_figure,
        MAIN_OUTPUT,
    )

    supplementary_paths = save_figure(
        supplementary_figure,
        SUPPLEMENT_OUTPUT,
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
        "phase2C5C_main_pca_scores.tsv": (
            scores
        ),
        "phase2C5C_main_context_effects.tsv": (
            effects
        ),
        "phase2C5C_main_axis_contributions.tsv": (
            contribution_matrix
        ),
        "phase2C5C_main_bootstrap_stability.tsv": (
            bootstrap
        ),
        "phase2C5C_main_clustering_metrics.tsv": (
            cluster_metrics
        ),
        "phase2C5C_supplementary_top_loadings.tsv": (
            top_loadings
        ),
        "phase2C5C_supplementary_leave_family_stability.tsv": (
            leave_family
        ),
        "phase2C5C_supplementary_cluster_composition.tsv": (
            cluster_composition
        ),
    }

    for filename, frame in source_tables.items():
        write_tsv(
            frame,
            SOURCE_DATA
            / filename,
        )

    expected_outputs = (
        main_paths
        + supplementary_paths
    )

    qa_rows = []

    for path in expected_outputs:
        if not path.exists():
            fail(
                f"Expected figure output is missing: {path}"
            )

        if path.suffix.lower() in {
            ".png",
            ".tiff",
            ".tif",
        }:
            qa_rows.append(
                inspect_raster(
                    path
                )
            )

        elif path.suffix.lower() == ".svg":
            qa_rows.append(
                inspect_svg(
                    path
                )
            )

        else:
            fail(
                f"Unsupported figure format: {path}"
            )

    qa_manifest = pd.DataFrame(
        qa_rows
    )

    make_contact_sheet(
        MAIN_OUTPUT.with_suffix(
            ".png"
        ),
        SUPPLEMENT_OUTPUT.with_suffix(
            ".png"
        ),
        CONTACT_SHEET_PATH,
    )

    legends = build_legends()

    write_tsv(
        qa_manifest,
        QA_MANIFEST_PATH,
    )

    write_tsv(
        legends,
        LEGEND_REGISTRY_PATH,
    )

    MANUSCRIPT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with LEGENDS_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "# Final Fiji multivariate figure legends\n\n"
        )

        for row in legends.itertuples(
            index=False
        ):
            handle.write(
                f"## {row.figure_identifier}\n\n"
            )

            handle.write(
                row.legend_text
            )

            handle.write(
                "\n\n"
            )

    failures = []

    if len(
        qa_manifest
    ) != 6:
        failures.append(
            f"Expected six QA rows, observed "
            f"{len(qa_manifest)}."
        )

    qa_failures = qa_manifest[
        ~qa_manifest[
            "qa_pass"
        ].astype(
            bool
        )
    ]

    if not qa_failures.empty:
        failures.append(
            "Automated figure QA failures: "
            + ", ".join(
                qa_failures[
                    "figure_file"
                ].astype(
                    str
                )
            )
        )

    if not CONTACT_SHEET_PATH.exists():
        failures.append(
            "Final QA contact sheet was not generated."
        )

    elif CONTACT_SHEET_PATH.stat().st_size == 0:
        failures.append(
            "Final QA contact sheet is empty."
        )

    if len(
        legends
    ) != 2:
        failures.append(
            f"Expected two legends, observed "
            f"{len(legends)}."
        )

    require_columns(
        axis_registry,
        {
            "core_axis_supported",
        },
        "Core immune-state axis registry",
    )

    supported_axis_count = int(
        as_bool(
            axis_registry[
                "core_axis_supported"
            ]
        ).sum()
    )

    if supported_axis_count != 4:
        failures.append(
            f"Expected four supported core-axis rows, "
            f"observed {supported_axis_count}."
        )

    if len(
        cluster_composition
    ) != 4:
        failures.append(
            f"Expected four ordered cluster-composition rows, "
            f"observed {len(cluster_composition)}."
        )

    decision_value = (
        "READY_FOR_PHASE2C5_COMMIT_AND_MANUSCRIPT_INTEGRATION"
        if not failures
        else "PHASE2C5C_REPAIR_REQUIRED"
    )

    decision_frame = pd.DataFrame(
        [
            {
                "decision": (
                    decision_value
                ),
                "main_figure_files": len(
                    main_paths
                ),
                "supplementary_figure_files": len(
                    supplementary_paths
                ),
                "total_figure_files": len(
                    expected_outputs
                ),
                "figure_source_data_tables": len(
                    source_tables
                ),
                "figure_QA_rows": len(
                    qa_manifest
                ),
                "figure_QA_passes": int(
                    qa_manifest[
                        "qa_pass"
                    ].astype(
                        bool
                    ).sum()
                ),
                "figure_QA_failures": int(
                    (
                        ~qa_manifest[
                            "qa_pass"
                        ].astype(
                            bool
                        )
                    ).sum()
                ),
                "raster_files_checked": int(
                    qa_manifest[
                        "file_type"
                    ].isin(
                        [
                            "png",
                            "tiff",
                            "tif",
                        ]
                    ).sum()
                ),
                "svg_files_checked": int(
                    (
                        qa_manifest[
                            "file_type"
                        ]
                        == "svg"
                    ).sum()
                ),
                "editable_svg_files": int(
                    qa_manifest[
                        "editable_vector"
                    ].fillna(
                        False
                    ).astype(
                        bool
                    ).sum()
                ),
                "contact_sheet_generated": bool(
                    CONTACT_SHEET_PATH.exists()
                ),
                "figure_legends_generated": len(
                    legends
                ),
                "supported_core_axis_rows": (
                    supported_axis_count
                ),
                "ordered_cluster_composition_rows": len(
                    cluster_composition
                ),
                "raster_dpi": RASTER_DPI,
                "validation_failures": "; ".join(
                    failures
                ),
            }
        ]
    )

    write_tsv(
        decision_frame,
        DECISION_PATH,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as report:
        report.write(
            "# Phase 2C5C Fiji final figure polish and QA\n\n"
        )

        report.write(
            "## Decision\n\n"
        )

        report.write(
            f"**{decision_value}**\n\n"
        )

        report.write(
            "## Visual refinements\n\n"
        )

        report.write(
            "- Replaced the crowded panel-specific PCA legend with "
            "one shared legend beneath the main title.\n"
        )

        report.write(
            "- Clarified that filled symbols in the effect-size panel "
            "represent FDR-significant components.\n"
        )

        report.write(
            "- Enlarged and boxed the clustering interpretation note.\n"
        )

        report.write(
            "- Increased loading-feature label size in supplementary "
            "panels A-D.\n"
        )

        report.write(
            "- Restricted the leave-one-family-out y-axis to the "
            "biologically relevant stability range and explicitly noted "
            "that ADCP and neutralization were absent from the "
            "BPV-calibrated matrix.\n"
        )

        report.write(
            "- Reordered cluster-composition bars as primary-dominant "
            "followed by recall-dominant within each representation and "
            "replaced internal cluster identifiers with reader-facing "
            "labels.\n\n"
        )

        report.write(
            "## Automated QA\n\n"
        )

        report.write(
            f"- Figure files inspected: {len(qa_manifest)}\n"
        )

        report.write(
            f"- Figure files passing: "
            f"{int(qa_manifest['qa_pass'].astype(bool).sum())}\n"
        )

        report.write(
            f"- Raster outputs: 600-dpi PNG and TIFF\n"
        )

        report.write(
            "- Vector outputs: editable SVG with text retained as text\n"
        )

        report.write(
            f"- Final contact sheet: "
            f"`{CONTACT_SHEET_PATH.relative_to(ROOT)}`\n\n"
        )

        report.write(
            "## Final biological presentation\n\n"
        )

        report.write(
            "The final figures present PC1 as a cross-reactive recall "
            "breadth axis and PC2 as a vaccine-type HPV16/18 effector "
            "axis. The raw binary clustering solution is shown as an "
            "experimental-context-dominated partition, whereas the "
            "BPV-calibrated response landscape is presented as continuous "
            "rather than discretely subtyped.\n"
        )

    print(
        "===== PHASE 2C5C COMPLETE ====="
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
            expected_outputs
        ),
    )

    print(
        "Figure source-data tables:",
        len(
            source_tables
        ),
    )

    print(
        "Figure QA rows:",
        len(
            qa_manifest
        ),
    )

    print(
        "Figure QA passes:",
        int(
            qa_manifest[
                "qa_pass"
            ].astype(
                bool
            ).sum()
        ),
    )

    print(
        "Figure QA failures:",
        int(
            (
                ~qa_manifest[
                    "qa_pass"
                ].astype(
                    bool
                )
            ).sum()
        ),
    )

    print(
        "Raster files checked:",
        int(
            qa_manifest[
                "file_type"
            ].isin(
                [
                    "png",
                    "tiff",
                    "tif",
                ]
            ).sum()
        ),
    )

    print(
        "SVG files checked:",
        int(
            (
                qa_manifest[
                    "file_type"
                ]
                == "svg"
            ).sum()
        ),
    )

    print(
        "Figure legends generated:",
        len(
            legends
        ),
    )

    print(
        "Ordered cluster-composition rows:",
        len(
            cluster_composition
        ),
    )

    print(
        "Contact sheet:",
        CONTACT_SHEET_PATH,
    )

    print(
        "Legends:",
        LEGENDS_PATH,
    )

    print(
        "Report:",
        REPORT_PATH,
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
