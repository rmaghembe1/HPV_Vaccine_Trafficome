#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    sys.exit(
        "ERROR: pandas and numpy are required.\n"
        "Install them with:\n"
        "  python -m pip install --user pandas numpy\n"
        f"Original error: {exc}"
    )


ROOT = Path("/mnt/d/HPV_Vaccine_Trafficome_Project")

PROCESSED_DIR = (
    ROOT
    / "07_data_processed"
    / "fiji_nct02276521"
)

LONG_FILE = (
    PROCESSED_DIR
    / "phase1C_fiji_participant_antigen_visit_feature_long.tsv"
)

PARTICIPANT_FILE = (
    PROCESSED_DIR
    / "phase1C_fiji_participant_metadata.tsv"
)

TABLE_DIR = ROOT / "08_results" / "tables"

REPORT_DIR = (
    ROOT
    / "02_dataset_audit"
    / "hpv_specific"
    / "fiji_nct02276521"
)

PAIRED_OUTPUT = (
    PROCESSED_DIR
    / "phase1D_fiji_paired_feature_changes.tsv"
)

REQUIRED_COLUMNS = {
    "participant_id",
    "previous_4vHPV_doses",
    "visit",
    "antigen_target",
    "antigen_class",
    "feature",
    "assay_family",
    "value",
}


def write_tsv(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(
        path,
        sep="\t",
        index=False,
        na_rep="",
    )


def recommend_transform(
    feature: str,
    minimum: float,
    positive_dynamic_range: float,
    skewness: float,
    unique_values: int,
) -> str:
    feature_lower = str(feature).lower()

    if feature_lower == "nab":
        return "log2_titer"

    if unique_values <= 1:
        return "uninformative_constant"

    if (
        minimum >= 0
        and (
            positive_dynamic_range >= 20
            or abs(skewness) >= 1
        )
    ):
        return "log2_with_target_feature_pseudocount"

    return "raw_for_univariate_scaled_for_multivariate"


def main() -> None:
    for required_file in [
        LONG_FILE,
        PARTICIPANT_FILE,
    ]:
        if not required_file.exists():
            sys.exit(
                f"ERROR: Required input file not found: "
                f"{required_file}"
            )

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    long_df = pd.read_csv(
        LONG_FILE,
        sep="\t",
        dtype={
            "participant_id": "string",
            "previous_4vHPV_doses": "string",
            "visit": "string",
            "antigen_target": "string",
            "feature": "string",
        },
    )

    participant_df = pd.read_csv(
        PARTICIPANT_FILE,
        sep="\t",
        dtype={
            "participant_id": "string",
            "previous_4vHPV_doses": "string",
        },
    )

    missing_columns = REQUIRED_COLUMNS - set(long_df.columns)

    if missing_columns:
        sys.exit(
            "ERROR: Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    long_df["participant_id"] = (
        long_df["participant_id"]
        .astype("string")
        .str.strip()
    )

    long_df["previous_4vHPV_doses"] = (
        long_df["previous_4vHPV_doses"]
        .astype("string")
        .str.strip()
    )

    long_df["visit"] = (
        long_df["visit"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    observed_visits = set(
        long_df["visit"]
        .dropna()
        .unique()
    )

    unexpected_visits = (
        observed_visits
        - {"v1", "v2"}
    )

    if unexpected_visits:
        sys.exit(
            "ERROR: Unexpected visits detected: "
            + ", ".join(sorted(unexpected_visits))
        )

    long_df["value_original"] = long_df["value"]

    long_df["value_numeric"] = pd.to_numeric(
        long_df["value"],
        errors="coerce",
    )

    nonnumeric_df = long_df[
        long_df["value"].notna()
        & (
            long_df["value"]
            .astype(str)
            .str.strip()
            != ""
        )
        & long_df["value_numeric"].isna()
    ].copy()

    nonnumeric_columns = [
        "participant_id",
        "previous_4vHPV_doses",
        "antigen_target",
        "visit",
        "feature",
        "assay_family",
        "value_original",
    ]

    nonnumeric_df = nonnumeric_df.reindex(
        columns=nonnumeric_columns
    )

    analytical_key = [
        "participant_id",
        "antigen_target",
        "visit",
        "feature",
    ]

    duplicate_df = (
        long_df.groupby(
            analytical_key,
            dropna=False,
        )
        .size()
        .reset_index(name="row_count")
    )

    duplicate_df = duplicate_df[
        duplicate_df["row_count"] > 1
    ].copy()

    numeric_df = long_df[
        long_df["value_numeric"].notna()
    ].copy()

    dose_distribution = (
        participant_df[
            [
                "participant_id",
                "previous_4vHPV_doses",
            ]
        ]
        .drop_duplicates("participant_id")
        .groupby(
            "previous_4vHPV_doses",
            dropna=False,
        )
        .agg(
            participants=(
                "participant_id",
                "nunique",
            )
        )
        .reset_index()
        .rename(
            columns={
                "previous_4vHPV_doses": "dose_group"
            }
        )
    )

    dose_distribution["dose_group"] = (
        dose_distribution["dose_group"]
        .fillna("missing")
    )

    transform_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []

    target_feature_groups = numeric_df.groupby(
        [
            "antigen_target",
            "antigen_class",
            "feature",
            "assay_family",
        ],
        dropna=False,
    )

    for (
        antigen_target,
        antigen_class,
        feature,
        assay_family,
    ), group in target_feature_groups:

        values = group["value_numeric"].dropna()

        if values.empty:
            continue

        positive_values = values[
            values > 0
        ]

        minimum = float(values.min())
        maximum = float(values.max())
        median = float(values.median())
        mean = float(values.mean())

        skewness = (
            float(values.skew())
            if len(values) >= 3
            else 0.0
        )

        unique_values = int(
            values.nunique()
        )

        minimum_positive = (
            float(positive_values.min())
            if not positive_values.empty
            else np.nan
        )

        positive_dynamic_range = (
            maximum / minimum_positive
            if (
                not pd.isna(minimum_positive)
                and minimum_positive > 0
            )
            else np.nan
        )

        pseudocount = (
            minimum_positive / 2
            if (
                not pd.isna(minimum_positive)
                and minimum_positive > 0
            )
            else 0.5
        )

        recommendation = recommend_transform(
            feature=str(feature),
            minimum=minimum,
            positive_dynamic_range=(
                positive_dynamic_range
                if not pd.isna(
                    positive_dynamic_range
                )
                else 1.0
            ),
            skewness=skewness,
            unique_values=unique_values,
        )

        transform_rows.append(
            {
                "antigen_target": antigen_target,
                "antigen_class": antigen_class,
                "feature": feature,
                "assay_family": assay_family,
                "observations": len(values),
                "minimum": minimum,
                "minimum_positive": minimum_positive,
                "median": median,
                "mean": mean,
                "maximum": maximum,
                "skewness": skewness,
                "unique_values": unique_values,
                "positive_dynamic_range": (
                    positive_dynamic_range
                ),
                "recommended_pseudocount": pseudocount,
                "transform_recommendation": (
                    recommendation
                ),
            }
        )

        for visit, visit_group in group.groupby(
            "visit",
            dropna=False,
        ):
            visit_values = (
                visit_group["value_numeric"]
                .dropna()
            )

            if visit_values.empty:
                continue

            visit_minimum = float(
                visit_values.min()
            )

            minimum_count = int(
                (
                    visit_values
                    == visit_minimum
                ).sum()
            )

            minimum_fraction = (
                minimum_count
                / len(visit_values)
            )

            zero_count = int(
                (
                    visit_values
                    == 0
                ).sum()
            )

            possible_floor = (
                "yes"
                if (
                    len(visit_values) >= 10
                    and minimum_fraction >= 0.05
                )
                else "no"
            )

            discrete_titer = (
                "yes"
                if (
                    str(feature).lower() == "nab"
                    or (
                        visit_values.nunique() <= 12
                        and len(visit_values) >= 10
                    )
                )
                else "no"
            )

            distribution_rows.append(
                {
                    "antigen_target": antigen_target,
                    "antigen_class": antigen_class,
                    "visit": visit,
                    "feature": feature,
                    "assay_family": assay_family,
                    "observations": len(
                        visit_values
                    ),
                    "unique_participants": (
                        visit_group[
                            "participant_id"
                        ].nunique()
                    ),
                    "minimum": visit_minimum,
                    "minimum_count": minimum_count,
                    "minimum_fraction": (
                        minimum_fraction
                    ),
                    "zero_count": zero_count,
                    "zero_fraction": (
                        zero_count
                        / len(visit_values)
                    ),
                    "q1": float(
                        visit_values.quantile(
                            0.25
                        )
                    ),
                    "median": float(
                        visit_values.median()
                    ),
                    "mean": float(
                        visit_values.mean()
                    ),
                    "q3": float(
                        visit_values.quantile(
                            0.75
                        )
                    ),
                    "maximum": float(
                        visit_values.max()
                    ),
                    "standard_deviation": (
                        float(
                            visit_values.std(
                                ddof=1
                            )
                        )
                        if len(visit_values) > 1
                        else np.nan
                    ),
                    "skewness": (
                        float(
                            visit_values.skew()
                        )
                        if len(visit_values) >= 3
                        else np.nan
                    ),
                    "unique_values": int(
                        visit_values.nunique()
                    ),
                    "possible_lower_bound_accumulation": (
                        possible_floor
                    ),
                    "discrete_titer_or_ordinal_assay": (
                        discrete_titer
                    ),
                }
            )

    transform_df = pd.DataFrame(
        transform_rows
    )

    distribution_df = pd.DataFrame(
        distribution_rows
    )

    transform_lookup = transform_df[
        [
            "antigen_target",
            "feature",
            "recommended_pseudocount",
            "transform_recommendation",
        ]
    ].copy()

    paired_source = numeric_df[
        [
            "participant_id",
            "previous_4vHPV_doses",
            "antigen_target",
            "antigen_class",
            "feature",
            "assay_family",
            "visit",
            "value_numeric",
        ]
    ].copy()

    paired_df = paired_source.pivot_table(
        index=[
            "participant_id",
            "previous_4vHPV_doses",
            "antigen_target",
            "antigen_class",
            "feature",
            "assay_family",
        ],
        columns="visit",
        values="value_numeric",
        aggfunc="first",
    ).reset_index()

    paired_df.columns.name = None

    if "v1" not in paired_df.columns:
        paired_df["v1"] = np.nan

    if "v2" not in paired_df.columns:
        paired_df["v2"] = np.nan

    paired_df = paired_df.merge(
        transform_lookup,
        on=[
            "antigen_target",
            "feature",
        ],
        how="left",
        validate="many_to_one",
    )

    paired_df["paired_complete"] = np.where(
        paired_df["v1"].notna()
        & paired_df["v2"].notna(),
        "yes",
        "no",
    )

    paired_df["raw_change_v2_minus_v1"] = (
        paired_df["v2"]
        - paired_df["v1"]
    )

    paired_df["fold_change_v2_over_v1"] = np.where(
        paired_df["v1"] != 0,
        paired_df["v2"]
        / paired_df["v1"],
        np.nan,
    )

    adjusted_v1 = (
        paired_df["v1"]
        + paired_df[
            "recommended_pseudocount"
        ]
    )

    adjusted_v2 = (
        paired_df["v2"]
        + paired_df[
            "recommended_pseudocount"
        ]
    )

    paired_df["log2_v1"] = np.where(
        adjusted_v1 > 0,
        np.log2(adjusted_v1),
        np.nan,
    )

    paired_df["log2_v2"] = np.where(
        adjusted_v2 > 0,
        np.log2(adjusted_v2),
        np.nan,
    )

    paired_df[
        "log2_change_v2_minus_v1"
    ] = (
        paired_df["log2_v2"]
        - paired_df["log2_v1"]
    )

    pairing_summary = (
        paired_df.groupby(
            [
                "antigen_target",
                "antigen_class",
                "feature",
                "assay_family",
                "previous_4vHPV_doses",
            ],
            dropna=False,
        )
        .agg(
            participant_feature_records=(
                "participant_id",
                "size",
            ),
            paired_participants=(
                "paired_complete",
                lambda values: int(
                    (
                        values
                        == "yes"
                    ).sum()
                ),
            ),
            visit1_observed=(
                "v1",
                lambda values: int(
                    values.notna().sum()
                ),
            ),
            visit2_observed=(
                "v2",
                lambda values: int(
                    values.notna().sum()
                ),
            ),
            median_log2_change=(
                "log2_change_v2_minus_v1",
                "median",
            ),
            mean_log2_change=(
                "log2_change_v2_minus_v1",
                "mean",
            ),
            sd_log2_change=(
                "log2_change_v2_minus_v1",
                "std",
            ),
            median_raw_change=(
                "raw_change_v2_minus_v1",
                "median",
            ),
            positive_log2_change_fraction=(
                "log2_change_v2_minus_v1",
                lambda values: (
                    float(
                        (
                            values.dropna()
                            > 0
                        ).mean()
                    )
                    if values.notna().any()
                    else np.nan
                ),
            ),
        )
        .reset_index()
    )

    pairing_summary["paired_fraction"] = (
        pairing_summary[
            "paired_participants"
        ]
        / pairing_summary[
            "participant_feature_records"
        ]
    )

    participant_visit_coverage = (
        numeric_df.groupby(
            [
                "participant_id",
                "previous_4vHPV_doses",
                "visit",
            ],
            dropna=False,
        )
        .agg(
            antigen_targets=(
                "antigen_target",
                "nunique",
            ),
            assay_features=(
                "feature",
                "nunique",
            ),
            nonmissing_measurements=(
                "value_numeric",
                "size",
            ),
        )
        .reset_index()
    )

    complete_paired_df = paired_df[
        paired_df["paired_complete"]
        == "yes"
    ].copy()

    floor_count = int(
        (
            distribution_df[
                "possible_lower_bound_accumulation"
            ]
            == "yes"
        ).sum()
    )

    discrete_count = int(
        (
            distribution_df[
                "discrete_titer_or_ordinal_assay"
            ]
            == "yes"
        ).sum()
    )

    if len(duplicate_df) > 0:
        decision = (
            "REPAIR_DUPLICATE_KEYS_BEFORE_MODELING"
        )
    elif len(nonnumeric_df) > 0:
        decision = (
            "REPAIR_NONNUMERIC_VALUES_BEFORE_MODELING"
        )
    elif len(complete_paired_df) == 0:
        decision = (
            "NO_COMPLETE_PAIRED_OBSERVATIONS"
        )
    else:
        decision = (
            "READY_FOR_PHASE2_DESCRIPTIVE_AND_MIXED_EFFECTS_ANALYSIS"
        )

    readiness_df = pd.DataFrame(
        [
            {
                "decision": decision,
                "unique_participants": (
                    participant_df[
                        "participant_id"
                    ].nunique()
                ),
                "numeric_feature_observations": (
                    len(numeric_df)
                ),
                "complete_paired_feature_observations": (
                    len(complete_paired_df)
                ),
                "duplicate_analytical_keys": (
                    len(duplicate_df)
                ),
                "nonnumeric_measurements": (
                    len(nonnumeric_df)
                ),
                "target_feature_transform_rules": (
                    len(transform_df)
                ),
                "possible_assay_floor_distributions": (
                    floor_count
                ),
                "discrete_or_titer_distributions": (
                    discrete_count
                ),
            }
        ]
    )

    write_tsv(
        dose_distribution,
        TABLE_DIR
        / "phase1D_fiji_dose_distribution.tsv",
    )

    write_tsv(
        transform_df,
        TABLE_DIR
        / "phase1D_fiji_transform_recommendations.tsv",
    )

    write_tsv(
        distribution_df,
        TABLE_DIR
        / "phase1D_fiji_feature_distribution_audit.tsv",
    )

    write_tsv(
        pairing_summary,
        TABLE_DIR
        / "phase1D_fiji_pairing_by_dose_target_feature.tsv",
    )

    write_tsv(
        participant_visit_coverage,
        TABLE_DIR
        / "phase1D_fiji_participant_visit_coverage.tsv",
    )

    write_tsv(
        duplicate_df,
        TABLE_DIR
        / "phase1D_fiji_duplicate_analytical_keys.tsv",
    )

    write_tsv(
        nonnumeric_df,
        TABLE_DIR
        / "phase1D_fiji_nonnumeric_measurements.tsv",
    )

    write_tsv(
        readiness_df,
        TABLE_DIR
        / "phase1D_fiji_statistical_readiness_decision.tsv",
    )

    write_tsv(
        paired_df,
        PAIRED_OUTPUT,
    )

    report_path = (
        REPORT_DIR
        / "phase1D_fiji_statistical_readiness_report.md"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as report:
        report.write(
            "# Phase 1D Fiji statistical-readiness audit\n\n"
        )

        report.write("## Decision\n\n")
        report.write(
            f"**{decision}**\n\n"
        )

        report.write("## Dataset structure\n\n")
        report.write(
            f"- Unique participants: "
            f"{participant_df['participant_id'].nunique()}\n"
        )
        report.write(
            f"- Numeric feature observations: "
            f"{len(numeric_df)}\n"
        )
        report.write(
            f"- Complete paired feature observations: "
            f"{len(complete_paired_df)}\n"
        )
        report.write(
            f"- Duplicate analytical keys: "
            f"{len(duplicate_df)}\n"
        )
        report.write(
            f"- Nonnumeric measurements: "
            f"{len(nonnumeric_df)}\n\n"
        )

        report.write(
            "## Previous 4vHPV dose distribution\n\n"
        )
        report.write(
            "| Previous doses | Participants |\n"
        )
        report.write("|---:|---:|\n")

        for _, row in (
            dose_distribution
            .sort_values("dose_group")
            .iterrows()
        ):
            report.write(
                f"| {row['dose_group']} "
                f"| {int(row['participants'])} |\n"
            )

        report.write(
            "\n## Assay-behaviour findings\n\n"
        )
        report.write(
            f"- Target-feature transformation rules: "
            f"{len(transform_df)}\n"
        )
        report.write(
            f"- Visit-feature distributions with possible "
            f"lower-bound accumulation: {floor_count}\n"
        )
        report.write(
            f"- Discrete or titer-like distributions: "
            f"{discrete_count}\n\n"
        )

        report.write(
            "## Transformation framework\n\n"
        )
        report.write(
            "- Neutralizing-antibody measurements are treated "
            "as discrete titers and analyzed on the log2 scale.\n"
        )
        report.write(
            "- Strongly skewed positive antibody and Fc-receptor "
            "measurements use target-feature-specific log2 "
            "transformations.\n"
        )
        report.write(
            "- The pseudocount is half the smallest positive "
            "value for the relevant antigen-feature combination.\n"
        )
        report.write(
            "- Repeated minimum values are flagged as possible "
            "assay-floor accumulation.\n\n"
        )

        report.write(
            "## Primary paired response\n\n"
        )
        report.write(
            "`log2(Visit 2 + pseudocount) - "
            "log2(Visit 1 + pseudocount)`\n\n"
        )
        report.write(
            "This represents the participant-specific transition "
            "from long-term persistence six years after previous "
            "4vHPV vaccination to day-28 recall following the "
            "2vHPV booster.\n\n"
        )

        report.write(
            "## Modeling implication\n\n"
        )
        report.write(
            "Initial Fiji models should include participant "
            "pairing, previous 4vHPV dose number, visit, antigen "
            "target or antigen class, and antibody-functional "
            "layer. Age, BMI, sex and ethnicity remain unavailable "
            "in the open workbook.\n"
        )

    print("===== PHASE 1D COMPLETE =====")
    print(f"Decision: {decision}")
    print(
        "Unique participants: "
        f"{participant_df['participant_id'].nunique()}"
    )
    print(
        "Complete paired feature observations: "
        f"{len(complete_paired_df)}"
    )
    print(
        "Duplicate analytical keys: "
        f"{len(duplicate_df)}"
    )
    print(
        "Nonnumeric measurements: "
        f"{len(nonnumeric_df)}"
    )
    print(
        "Possible assay-floor distributions: "
        f"{floor_count}"
    )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
