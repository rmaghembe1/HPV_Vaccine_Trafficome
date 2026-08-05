#!/usr/bin/env python3

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(
    "/mnt/d/HPV_Vaccine_Trafficome_Project"
)

TABLES = (
    ROOT
    / "08_results"
    / "tables"
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

C4C_SCHEDULE_PATH = (
    TABLES
    / "phase2C4C_fiji_schedule_effect_registry.tsv"
)

EXPECTED_C5A_DECISION = (
    "READY_FOR_PHASE2C5B_FIGURE_QA_AND_MANUSCRIPT_INTEGRATION"
)

MAIN_BASENAME = (
    "Figure_Fiji_multivariate_immune_state_architecture_v1"
)

SUPPLEMENT_BASENAME = (
    "FigureS_Fiji_multivariate_stability_and_loadings_v1"
)

EXPECTED_FIGURES = [
    FIGURE_DIR
    / f"{MAIN_BASENAME}.png",
    FIGURE_DIR
    / f"{MAIN_BASENAME}.tiff",
    FIGURE_DIR
    / f"{MAIN_BASENAME}.svg",
    FIGURE_DIR
    / f"{SUPPLEMENT_BASENAME}.png",
    FIGURE_DIR
    / f"{SUPPLEMENT_BASENAME}.tiff",
    FIGURE_DIR
    / f"{SUPPLEMENT_BASENAME}.svg",
]

CONTACT_SHEET_PATH = (
    FIGURE_DIR
    / "phase2C5B_fiji_figure_QA_contact_sheet.png"
)

QA_MANIFEST_PATH = (
    TABLES
    / "phase2C5B_fiji_figure_QA_manifest.tsv"
)

CAPTION_TABLE_PATH = (
    TABLES
    / "phase2C5B_fiji_figure_legend_registry.tsv"
)

DECISION_PATH = (
    TABLES
    / "phase2C5B_fiji_figure_QA_decision.tsv"
)

REPORT_PATH = (
    REPORT_DIR
    / "phase2C5B_fiji_figure_QA_report.md"
)

LEGENDS_PATH = (
    MANUSCRIPT_DIR
    / "phase2C5B_fiji_multivariate_figure_legends.md"
)

MIN_RASTER_WIDTH = 3000
MIN_RASTER_HEIGHT = 1800
MIN_RASTER_DPI = 590.0
MIN_GRAYSCALE_SD = 5.0
MIN_CONTENT_FRACTION = 0.20
MAX_CONTENT_FRACTION = 0.98


def fail(
    message: str,
) -> None:
    raise SystemExit(
        f"ERROR: {message}"
    )


def require_file(
    path: Path,
) -> None:
    if not path.exists():
        fail(
            f"Required file is missing: {path}"
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


def normalize_dpi(
    image: Image.Image,
) -> tuple[float, float]:
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

    return (
        x_dpi,
        y_dpi,
    )


def content_metrics(
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

    grayscale_sd = float(
        np.std(
            grayscale
        )
    )

    nonwhite = np.any(
        array
        < 248,
        axis=2,
    )

    nonwhite_fraction = float(
        np.mean(
            nonwhite
        )
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

        bounding_width = (
            maximum_x
            - minimum_x
            + 1
        )

        bounding_height = (
            maximum_y
            - minimum_y
            + 1
        )

        bounding_fraction = float(
            (
                bounding_width
                * bounding_height
            )
            / (
                preview.width
                * preview.height
            )
        )

    else:
        bounding_fraction = 0.0

    return {
        "grayscale_standard_deviation": (
            grayscale_sd
        ),
        "nonwhite_pixel_fraction": (
            nonwhite_fraction
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

        x_dpi, y_dpi = normalize_dpi(
            image
        )

        metrics = content_metrics(
            image
        )

        format_name = str(
            image.format
        )

        image_mode = str(
            image.mode
        )

        frame_count = int(
            getattr(
                image,
                "n_frames",
                1,
            )
        )

    width_pass = (
        width
        >= MIN_RASTER_WIDTH
    )

    height_pass = (
        height
        >= MIN_RASTER_HEIGHT
    )

    dpi_pass = (
        np.isfinite(
            x_dpi
        )
        and np.isfinite(
            y_dpi
        )
        and x_dpi
        >= MIN_RASTER_DPI
        and y_dpi
        >= MIN_RASTER_DPI
    )

    variation_pass = (
        metrics[
            "grayscale_standard_deviation"
        ]
        >= MIN_GRAYSCALE_SD
    )

    content_pass = (
        metrics[
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
        "file_type": (
            path.suffix.lower().lstrip(
                "."
            )
        ),
        "file_size_bytes": int(
            path.stat().st_size
        ),
        "image_format": format_name,
        "image_mode": image_mode,
        "width_pixels": int(
            width
        ),
        "height_pixels": int(
            height
        ),
        "x_dpi": x_dpi,
        "y_dpi": y_dpi,
        "frame_count": frame_count,
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
        "dimension_pass": bool(
            width_pass
            and height_pass
        ),
        "dpi_pass": bool(
            dpi_pass
        ),
        "image_variation_pass": bool(
            variation_pass
        ),
        "content_occupancy_pass": bool(
            content_pass
        ),
        "editable_vector": False,
        "svg_text_elements": np.nan,
        "svg_embedded_images": np.nan,
        "svg_viewbox_present": np.nan,
        "qa_pass": bool(
            width_pass
            and height_pass
            and dpi_pass
            and variation_pass
            and content_pass
        ),
    }


def svg_tag_count(
    root: ET.Element,
    local_name: str,
) -> int:
    count = 0

    for element in root.iter():
        tag = str(
            element.tag
        )

        if tag.endswith(
            f"}}{local_name}"
        ) or tag == local_name:
            count += 1

    return count


def inspect_svg(
    path: Path,
) -> dict[str, object]:
    tree = ET.parse(
        path
    )

    root = tree.getroot()

    text_count = svg_tag_count(
        root,
        "text",
    )

    embedded_image_count = svg_tag_count(
        root,
        "image",
    )

    path_count = svg_tag_count(
        root,
        "path",
    )

    group_count = svg_tag_count(
        root,
        "g",
    )

    viewbox = root.attrib.get(
        "viewBox"
    )

    width_attribute = root.attrib.get(
        "width"
    )

    height_attribute = root.attrib.get(
        "height"
    )

    text_pass = (
        text_count
        > 20
    )

    embedded_image_pass = (
        embedded_image_count
        == 0
    )

    viewbox_pass = bool(
        viewbox
    )

    size_pass = (
        path.stat().st_size
        > 10000
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
        "frame_count": np.nan,
        "grayscale_standard_deviation": np.nan,
        "nonwhite_pixel_fraction": np.nan,
        "content_bounding_fraction": np.nan,
        "dimension_pass": bool(
            size_pass
        ),
        "dpi_pass": np.nan,
        "image_variation_pass": np.nan,
        "content_occupancy_pass": np.nan,
        "editable_vector": True,
        "svg_text_elements": int(
            text_count
        ),
        "svg_embedded_images": int(
            embedded_image_count
        ),
        "svg_path_elements": int(
            path_count
        ),
        "svg_group_elements": int(
            group_count
        ),
        "svg_viewbox_present": bool(
            viewbox_pass
        ),
        "svg_width_attribute": (
            width_attribute
        ),
        "svg_height_attribute": (
            height_attribute
        ),
        "qa_pass": bool(
            text_pass
            and embedded_image_pass
            and viewbox_pass
            and size_pass
        ),
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
                    2200,
                    1400,
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
    title_height = 70
    gap = 80

    sheet_width = max(
        image.width
        for _, image in images
    ) + 2 * margin

    sheet_height = (
        sum(
            image.height
            for _, image in images
        )
        + 2 * title_height
        + gap
        + 2 * margin
    )

    sheet = Image.new(
        "RGB",
        (
            sheet_width,
            sheet_height,
        ),
        "white",
    )

    draw = ImageDraw.Draw(
        sheet
    )

    font = ImageFont.load_default()

    y_position = margin

    for index, (
        filename,
        image,
    ) in enumerate(
        images,
        start=1,
    ):
        heading = (
            f"{index}. {filename}"
        )

        draw.text(
            (
                margin,
                y_position,
            ),
            heading,
            fill="black",
            font=font,
        )

        y_position += title_height

        x_position = (
            sheet_width
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
        "BPV-calibrated systems-serology matrices. Points represent "
        "individual participants, color identifies primary 2vHPV "
        "induction or heterologous 2vHPV recall, marker shape identifies "
        "the number of previous 4vHPV doses, and X symbols indicate "
        "context centroids. (C) Standardized primary-versus-recall effect "
        "sizes across PC1-PC10. Filled symbols identify components that "
        "remained significant after within-representation false-discovery "
        "rate correction. Both raw and BPV-calibrated PC1 and PC2 "
        "distinguished primary induction from recall. "
        "(D) Feature-family contributions to PC1 and PC2. PC1 was "
        "principally organized by IgG-subclass and Fc-receptor "
        "communication features, whereas PC2 emphasized vaccine-type "
        "binding-antibody abundance and, in the raw matrix, phagocytic "
        "and neutralizing function. (E) Participant-bootstrap stability "
        "of PC1-PC5, expressed as absolute correlations between reference "
        "and bootstrap loading vectors after component matching. PC1 was "
        "the most reproducible component in both representations; PC2 "
        "showed moderate-to-strong stability. Points show medians and "
        "error bars show the 2.5th to 97.5th percentiles across 300 "
        "bootstrap replicates. (F) Evaluation of candidate k-means "
        "partitions from k=2 through k=6. Marker size represents mean "
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
        "features, ADCP and neutralization; the calibrated PC2 retained "
        "a predominantly HPV16/18 binding and Fc-receptor structure. "
        "(E) Sign-aligned score correlations after separately omitting "
        "each feature family. PC1 and PC2 remained reproducible after "
        "removal of individual feature families, although PC2 was more "
        "sensitive than PC1. The dashed line indicates a score "
        "correlation of 0.75. (F) Immunization-context composition of "
        "the selected raw k=2 solution and the best available "
        "BPV-calibrated k=2 solution. Both partitions were strongly "
        "context-associated. The calibrated solution did not meet the "
        "prespecified silhouette threshold and was not accepted as a "
        "stable discrete immune-state partition."
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
            "Unexpected Phase 2C5A decision: "
            f"{observed_decision}"
        )

    axis_registry = read_table(
        C4C_AXIS_REGISTRY_PATH,
        "Core immune-state axis registry",
    )

    clustering = read_table(
        C4C_CLUSTERING_PATH,
        "Clustering synthesis",
    )

    schedule = read_table(
        C4C_SCHEDULE_PATH,
        "Schedule-effect registry",
    )

    require_columns(
        axis_registry,
        {
            "matrix_representation",
            "component_number",
            "core_axis_supported",
        },
        "Core immune-state axis registry",
    )

    require_columns(
        clustering,
        {
            "matrix_representation",
            "cluster_count",
            "stable_cluster_candidate",
            "context_alignment_accuracy",
        },
        "Clustering synthesis",
    )

    require_columns(
        schedule,
        {
            "matrix_representation",
            "component_number",
            "fdr_significant",
            "evidence_class",
        },
        "Schedule-effect registry",
    )

    for path in EXPECTED_FIGURES:
        require_file(
            path
        )

    qa_rows = []

    for path in EXPECTED_FIGURES:
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
                f"Unexpected figure format: {path}"
            )

    qa_manifest = pd.DataFrame(
        qa_rows
    )

    make_contact_sheet(
        FIGURE_DIR
        / f"{MAIN_BASENAME}.png",
        FIGURE_DIR
        / f"{SUPPLEMENT_BASENAME}.png",
        CONTACT_SHEET_PATH,
    )

    legends = build_legends()

    write_tsv(
        qa_manifest,
        QA_MANIFEST_PATH,
    )

    write_tsv(
        legends,
        CAPTION_TABLE_PATH,
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
            "# Fiji multivariate systems-serology figure legends\n\n"
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
            f"Expected six figure QA rows; "
            f"observed {len(qa_manifest)}."
        )

    failed_figure_rows = qa_manifest[
        ~qa_manifest[
            "qa_pass"
        ].astype(
            bool
        )
    ]

    if not failed_figure_rows.empty:
        failures.append(
            "One or more figure files failed automated QA: "
            + ", ".join(
                failed_figure_rows[
                    "figure_file"
                ].astype(
                    str
                )
            )
        )

    if not CONTACT_SHEET_PATH.exists():
        failures.append(
            "QA contact sheet was not generated."
        )

    elif CONTACT_SHEET_PATH.stat().st_size == 0:
        failures.append(
            "QA contact sheet is empty."
        )

    if len(
        legends
    ) != 2:
        failures.append(
            f"Expected two figure legends; "
            f"observed {len(legends)}."
        )

    supported_axes = (
        axis_registry[
            "core_axis_supported"
        ]
        .astype(
            "string"
        )
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes",
            ]
        )
        .sum()
    )

    if int(
        supported_axes
    ) != 4:
        failures.append(
            f"Expected four supported core-axis rows; "
            f"observed {supported_axes}."
        )

    decision_value = (
        "READY_FOR_PHASE2C5_COMMIT_AND_MANUSCRIPT_INTEGRATION"
        if not failures
        else "PHASE2C5B_REPAIR_REQUIRED"
    )

    decision_frame = pd.DataFrame(
        [
            {
                "decision": (
                    decision_value
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
                "supported_core_axis_rows": int(
                    supported_axes
                ),
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
            "# Phase 2C5B Fiji figure QA and manuscript integration\n\n"
        )

        report.write(
            "## Decision\n\n"
        )

        report.write(
            f"**{decision_value}**\n\n"
        )

        report.write(
            "## Automated file QA\n\n"
        )

        report.write(
            f"- Figure files inspected: {len(qa_manifest)}\n"
        )

        report.write(
            f"- Figure files passing automated QA: "
            f"{int(qa_manifest['qa_pass'].astype(bool).sum())}\n"
        )

        report.write(
            "- Raster checks: successful opening, minimum pixel "
            "dimensions, approximately 600-dpi metadata, image "
            "variation and non-empty content occupancy.\n"
        )

        report.write(
            "- SVG checks: valid XML, retained text elements, "
            "viewBox present and no embedded raster images.\n\n"
        )

        report.write(
            "## Visual review file\n\n"
        )

        report.write(
            f"- Contact sheet: "
            f"`{CONTACT_SHEET_PATH.relative_to(ROOT)}`\n\n"
        )

        report.write(
            "The contact sheet contains the main and supplementary "
            "figures at reduced scale for manual inspection of panel "
            "balance, text size, legends, axis labeling and clipping.\n\n"
        )

        report.write(
            "## Manuscript integration\n\n"
        )

        report.write(
            f"- Figure legends: "
            f"`{LEGENDS_PATH.relative_to(ROOT)}`\n"
        )

        report.write(
            f"- Legend registry: "
            f"`{CAPTION_TABLE_PATH.relative_to(ROOT)}`\n\n"
        )

        report.write(
            "The main legend presents the continuous two-axis "
            "immune-state architecture, while the supplementary legend "
            "documents loading structure, feature-family robustness and "
            "the context-dominated nature of the clustering solutions.\n\n"
        )

        report.write(
            "## Scientific interpretation\n\n"
        )

        report.write(
            "The publication-facing interpretation remains that PC1 "
            "represents cross-reactive recall breadth and PC2 represents "
            "vaccine-type HPV16/18 effector organization. Raw k=2 "
            "clustering is treated as an experimental-context partition, "
            "not as an intrinsic immune subtype. The raw-only recall-dose "
            "association remains a qualified secondary result because it "
            "was not retained after BPV calibration.\n"
        )

    print(
        "===== PHASE 2C5B COMPLETE ====="
    )

    print(
        f"Decision: {decision_value}"
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
