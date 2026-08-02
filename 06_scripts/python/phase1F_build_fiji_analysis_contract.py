#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError as exc:
    sys.exit(
        "ERROR: pandas is required.\n"
        "Install it with:\n"
        "  python -m pip install --user pandas\n"
        f"Original error: {exc}"
    )


ROOT = Path("/mnt/d/HPV_Vaccine_Trafficome_Project")

TABLE_DIR = ROOT / "08_results" / "tables"
PROTOCOL_DIR = ROOT / "01_concept_and_protocol"

READINESS_FILE = (
    TABLE_DIR
    / "phase1D_fiji_statistical_readiness_decision.tsv"
)

CONTEXT_FILE = (
    TABLE_DIR
    / "phase1E_fiji_immunization_context_decision.tsv"
)

DISTRIBUTION_FILE = (
    TABLE_DIR
    / "phase1D_fiji_feature_distribution_audit.tsv"
)

TRANSFORM_FILE = (
    TABLE_DIR
    / "phase1D_fiji_transform_recommendations.tsv"
)


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        sep="\t",
        index=False,
        na_rep="",
    )


def floor_severity(fraction: float) -> str:
    if fraction < 0.05:
        return "none"

    if fraction < 0.15:
        return "low"

    if fraction < 0.30:
        return "moderate"

    return "high"


def main() -> None:
    for required_file in [
        READINESS_FILE,
        CONTEXT_FILE,
        DISTRIBUTION_FILE,
        TRANSFORM_FILE,
    ]:
        if not required_file.exists():
            sys.exit(
                f"ERROR: Required input missing: "
                f"{required_file}"
            )

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)

    readiness = pd.read_csv(
        READINESS_FILE,
        sep="\t",
    )

    context = pd.read_csv(
        CONTEXT_FILE,
        sep="\t",
    )

    distribution = pd.read_csv(
        DISTRIBUTION_FILE,
        sep="\t",
    )

    transformations = pd.read_csv(
        TRANSFORM_FILE,
        sep="\t",
    )

    readiness_decision = str(
        readiness.loc[0, "decision"]
    )

    context_decision = str(
        context.loc[0, "decision"]
    )

    if readiness_decision != (
        "READY_FOR_PHASE2_DESCRIPTIVE_AND_MIXED_EFFECTS_ANALYSIS"
    ):
        sys.exit(
            "ERROR: Phase 1D is not analysis-ready: "
            f"{readiness_decision}"
        )

    if context_decision != "PASS":
        sys.exit(
            "ERROR: Phase 1E context decision is not PASS: "
            f"{context_decision}"
        )

    distribution[
        "floor_severity"
    ] = distribution[
        "minimum_fraction"
    ].apply(floor_severity)

    floor_summary = (
        distribution.groupby(
            [
                "floor_severity",
                "antigen_class",
                "assay_family",
            ],
            dropna=False,
        )
        .agg(
            visit_feature_distributions=(
                "feature",
                "size",
            ),
            antigen_targets=(
                "antigen_target",
                "nunique",
            ),
            features=(
                "feature",
                "nunique",
            ),
            median_floor_fraction=(
                "minimum_fraction",
                "median",
            ),
            maximum_floor_fraction=(
                "minimum_fraction",
                "max",
            ),
        )
        .reset_index()
    )

    floor_distribution = distribution[
        [
            "antigen_target",
            "antigen_class",
            "visit",
            "feature",
            "assay_family",
            "minimum",
            "minimum_count",
            "minimum_fraction",
            "floor_severity",
            "discrete_titer_or_ordinal_assay",
        ]
    ].copy()

    model_rows = [
        {
            "model_id": "HPVVT-FJ-M01",
            "model_family": "Primary 2vHPV induction",
            "population": "Participants with zero previous 4vHPV doses",
            "sample_size": 20,
            "biological_transition": (
                "Unvaccinated baseline to day-28 primary 2vHPV response"
            ),
            "primary_predictor": "visit",
            "interaction": "visit_by_antigen_target where estimable",
            "random_effect": "participant_id",
            "primary_effect": (
                "within-participant Visit 2 minus Visit 1 log2 change"
            ),
            "interpretation": "Primary immune induction",
        },
        {
            "model_id": "HPVVT-FJ-M02",
            "model_family": "Six-year 4vHPV persistence",
            "population": "Participants with one to three previous 4vHPV doses",
            "sample_size": 60,
            "biological_transition": "Comparison of Visit 1 persistent immunity",
            "primary_predictor": "previous_4vHPV_doses as categorical factor",
            "interaction": "dose_by_antigen_target where estimable",
            "random_effect": "participant_id for multi-antigen models",
            "primary_effect": (
                "dose-group difference in Visit 1 transformed response"
            ),
            "interpretation": "Long-term persistence after prior vaccination",
        },
        {
            "model_id": "HPVVT-FJ-M03",
            "model_family": "Heterologous 2vHPV recall",
            "population": "Participants with one to three previous 4vHPV doses",
            "sample_size": 60,
            "biological_transition": (
                "Six-year 4vHPV persistence to day-28 2vHPV recall"
            ),
            "primary_predictor": "visit",
            "interaction": "visit_by_previous_4vHPV_doses",
            "random_effect": "participant_id",
            "primary_effect": (
                "dose-specific within-participant log2 recall change"
            ),
            "interpretation": "Memory recall after heterologous boosting",
        },
        {
            "model_id": "HPVVT-FJ-M04",
            "model_family": "Primary-versus-recall contrast",
            "population": "All participants",
            "sample_size": 80,
            "biological_transition": (
                "Primary induction in dose 0 versus recall in doses 1 to 3"
            ),
            "primary_predictor": "prior_4vHPV_exposure_status",
            "interaction": "visit_by_prior_4vHPV_exposure_status",
            "random_effect": "participant_id",
            "primary_effect": (
                "difference in paired log2 change between naive and "
                "previously vaccinated participants"
            ),
            "interpretation": (
                "Qualitative distinction between primary induction "
                "and memory recall"
            ),
        },
        {
            "model_id": "HPVVT-FJ-M05",
            "model_family": "Cross-reactive breadth",
            "population": "All participants, stratified by immunization context",
            "sample_size": 80,
            "biological_transition": (
                "Responses to HPV31, HPV33, HPV45, HPV52 and HPV58"
            ),
            "primary_predictor": "visit and previous_4vHPV_doses",
            "interaction": "visit_by_dose_by_cross_reactive_antigen",
            "random_effect": "participant_id",
            "primary_effect": (
                "cross-reactive antigen-specific paired log2 change"
            ),
            "interpretation": "Breadth beyond HPV16 and HPV18",
        },
        {
            "model_id": "HPVVT-FJ-M06",
            "model_family": "Heterologous BPV control",
            "population": "All participants",
            "sample_size": 80,
            "biological_transition": "BPV assay response across visits",
            "primary_predictor": "visit and previous_4vHPV_doses",
            "interaction": "visit_by_dose",
            "random_effect": "participant_id",
            "primary_effect": (
                "BPV paired change relative to HPV-specific changes"
            ),
            "interpretation": (
                "Control for nonspecific or assay-wide shifts"
            ),
        },
        {
            "model_id": "HPVVT-FJ-M07",
            "model_family": "Antibody functional coupling",
            "population": "Participants with paired HPV16 or HPV18 measurements",
            "sample_size": 80,
            "biological_transition": (
                "Antibody subclass and Fc-receptor remodeling linked "
                "to ADCP and neutralization"
            ),
            "primary_predictor": (
                "IgG subclasses and Fc-gamma-receptor-binding features"
            ),
            "interaction": "immunization_context where supported",
            "random_effect": "participant_id",
            "primary_effect": (
                "partial association with ADCP or log2 neutralization titer"
            ),
            "interpretation": (
                "Functional organization of antibody quality"
            ),
        },
    ]

    model_contract = pd.DataFrame(model_rows)

    outcome_rows = [
        {
            "outcome_family": "Antibody abundance",
            "features": "IgG;IgM;IgA1;IgA2",
            "antigen_scope": (
                "HPV16;HPV18;HPV31;HPV33;HPV45;HPV52;HPV58;BPV"
            ),
            "primary_scale": "log2_raw_positive_value",
            "effect_measure": "log2_change_and_geometric_mean_ratio",
            "multiplicity_family": "abundance",
        },
        {
            "outcome_family": "IgG subclass architecture",
            "features": "IgG1;IgG2;IgG3;IgG4",
            "antigen_scope": (
                "HPV16;HPV18;HPV31;HPV33;HPV45;HPV52;HPV58;BPV"
            ),
            "primary_scale": "log2_raw_positive_value",
            "effect_measure": "log2_change_and_geometric_mean_ratio",
            "multiplicity_family": "igg_subclasses",
        },
        {
            "outcome_family": "Fc-receptor engagement",
            "features": "FcgR2A;FcgR2B;FcgR3A",
            "antigen_scope": (
                "HPV16;HPV18;HPV31;HPV33;HPV45;HPV52;HPV58;BPV"
            ),
            "primary_scale": "log2_raw_positive_value",
            "effect_measure": "log2_change_and_geometric_mean_ratio",
            "multiplicity_family": "fc_receptor_binding",
        },
        {
            "outcome_family": "Phagocytic function",
            "features": "ADCP",
            "antigen_scope": "HPV16;HPV18",
            "primary_scale": "log2_raw_positive_value",
            "effect_measure": "log2_change_and_geometric_mean_ratio",
            "multiplicity_family": "adcp",
        },
        {
            "outcome_family": "Neutralization",
            "features": "nAb",
            "antigen_scope": "HPV16;HPV18",
            "primary_scale": "log2_dilution_titer",
            "effect_measure": "log2_titer_change_and_geometric_mean_titer_ratio",
            "multiplicity_family": "neutralization",
        },
    ]

    outcome_registry = pd.DataFrame(outcome_rows)

    transformation_contract = transformations[
        [
            "antigen_target",
            "antigen_class",
            "feature",
            "assay_family",
            "minimum",
            "minimum_positive",
            "recommended_pseudocount",
            "transform_recommendation",
        ]
    ].copy()

    transformation_contract[
        "authoritative_phase1F_scale"
    ] = transformation_contract.apply(
        lambda row: (
            "log2_dilution_titer"
            if str(row["feature"]).lower() == "nab"
            else "log2_raw_positive_value"
        ),
        axis=1,
    )

    transformation_contract[
        "phase1F_note"
    ] = transformation_contract.apply(
        lambda row: (
            "No pseudocount required because observed titers are positive; "
            "retain assay-floor indicator."
            if str(row["feature"]).lower() == "nab"
            else (
                "Observed values are positive; use raw log2 transformation. "
                "Retain target-visit floor indicator and perform two-part "
                "sensitivity analysis for moderate or high floor severity."
            )
        ),
        axis=1,
    )

    analysis_decision = pd.DataFrame(
        [
            {
                "decision": "READY_FOR_PHASE2_MODELING",
                "phase1D_readiness": readiness_decision,
                "phase1E_context": context_decision,
                "unique_participants": 80,
                "dose0_primary_participants": 20,
                "dose1_recall_participants": 20,
                "dose2_recall_participants": 21,
                "dose3_recall_participants": 19,
                "primary_model_families": len(model_contract),
                "outcome_families": len(outcome_registry),
                "floor_policy": (
                    "none_below_5pct;low_5_to_lt15pct;"
                    "moderate_15_to_lt30pct;high_ge30pct"
                ),
                "multiplicity_policy": (
                    "Benjamini-Hochberg within predefined biological "
                    "outcome and contrast families"
                ),
            }
        ]
    )

    write_tsv(
        floor_summary,
        TABLE_DIR
        / "phase1F_fiji_floor_severity_summary.tsv",
    )

    write_tsv(
        floor_distribution,
        TABLE_DIR
        / "phase1F_fiji_floor_severity_registry.tsv",
    )

    write_tsv(
        model_contract,
        TABLE_DIR
        / "phase1F_fiji_model_family_contract.tsv",
    )

    write_tsv(
        outcome_registry,
        TABLE_DIR
        / "phase1F_fiji_outcome_registry.tsv",
    )

    write_tsv(
        transformation_contract,
        TABLE_DIR
        / "phase1F_fiji_transformation_contract.tsv",
    )

    write_tsv(
        analysis_decision,
        TABLE_DIR
        / "phase1F_fiji_analysis_contract_decision.tsv",
    )

    report_path = (
        PROTOCOL_DIR
        / "phase1F_fiji_statistical_analysis_contract.md"
    )

    severity_counts = (
        distribution["floor_severity"]
        .value_counts()
        .to_dict()
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as report:
        report.write(
            "# Phase 1F Fiji statistical analysis contract\n\n"
        )

        report.write("## Decision\n\n")
        report.write(
            "**READY_FOR_PHASE2_MODELING**\n\n"
        )

        report.write("## Biological estimands\n\n")
        report.write(
            "The Fiji cohort contains two distinct immunological "
            "trajectories that must not be combined under a uniform "
            "booster-response label:\n\n"
        )
        report.write(
            "1. **Primary induction:** unvaccinated baseline to the "
            "day-28 primary 2vHPV response among participants with "
            "zero previous 4vHPV doses.\n"
        )
        report.write(
            "2. **Persistence and recall:** six-year 4vHPV immunity "
            "to day-28 heterologous 2vHPV recall among participants "
            "with one, two or three previous 4vHPV doses.\n\n"
        )

        report.write("## Core modeling rules\n\n")
        report.write(
            "- Previous dose number is categorical in primary models.\n"
        )
        report.write(
            "- Participant identity is retained as a random intercept "
            "for repeated-measure models.\n"
        )
        report.write(
            "- Models are fitted separately by antibody feature or "
            "prespecified functional family; raw assay scales are not "
            "pooled indiscriminately.\n"
        )
        report.write(
            "- HPV16 and HPV18 are vaccine-target antigens.\n"
        )
        report.write(
            "- HPV31, HPV33, HPV45, HPV52 and HPV58 define the "
            "cross-reactive antigen layer.\n"
        )
        report.write(
            "- BPV is retained as a heterologous control and is not "
            "treated as an HPV vaccine-response antigen.\n"
        )
        report.write(
            "- ADCP and neutralization are analyzed only for HPV16 "
            "and HPV18.\n\n"
        )

        report.write("## Transformation rules\n\n")
        report.write(
            "- All observed assay measurements are positive; therefore "
            "primary continuous models use `log2(raw value)` without "
            "adding an arbitrary pseudocount.\n"
        )
        report.write(
            "- Neutralizing antibody is analyzed as a discrete dilution "
            "titer on the log2 scale.\n"
        )
        report.write(
            "- Effect sizes are reported as log2 differences and "
            "geometric-mean ratios.\n"
        )
        report.write(
            "- Original-scale summaries remain available for biological "
            "interpretation.\n\n"
        )

        report.write("## Assay-floor policy\n\n")
        report.write(
            "| Floor fraction | Severity | Primary treatment |\n"
        )
        report.write("|---:|---|---|\n")
        report.write(
            "| <5% | None | Standard log2 continuous model |\n"
        )
        report.write(
            "| 5%–<15% | Low | Continuous model plus floor summary |\n"
        )
        report.write(
            "| 15%–<30% | Moderate | Continuous model plus binary "
            "above-floor sensitivity analysis |\n"
        )
        report.write(
            "| ≥30% | High | Two-part sensitivity analysis: probability "
            "above floor and conditional response magnitude |\n\n"
        )

        report.write("Observed distribution counts:\n\n")
        for severity in [
            "none",
            "low",
            "moderate",
            "high",
        ]:
            report.write(
                f"- {severity}: "
                f"{severity_counts.get(severity, 0)}\n"
            )

        report.write("\n## Multiplicity control\n\n")
        report.write(
            "Benjamini–Hochberg false-discovery-rate correction is "
            "applied within prespecified biological families rather than "
            "across every numerical test in the project. Families include "
            "antibody abundance, IgG subclasses, Fc-receptor binding, "
            "ADCP, neutralization, persistence, primary induction, recall, "
            "dose interactions and cross-reactive breadth.\n\n"
        )

        report.write("## Reporting requirements\n\n")
        report.write(
            "Each reported comparison must include the number of "
            "participants, effect estimate, 95% confidence interval, "
            "raw P value, within-family adjusted P value, floor severity "
            "and biological context. Statistical significance alone will "
            "not substitute for effect magnitude or mechanistic coherence.\n\n"
        )

        report.write("## Excluded Fiji-specific covariates\n\n")
        report.write(
            "Age, weight, BMI, sex and ethnicity are not present in the "
            "open workbook and must not be imputed or inferred. These "
            "questions will be addressed through additional HPV cohorts, "
            "data requests or mechanistic comparator datasets.\n"
        )

    print("===== PHASE 1F COMPLETE =====")
    print("Decision: READY_FOR_PHASE2_MODELING")
    print(f"Model families: {len(model_contract)}")
    print(f"Outcome families: {len(outcome_registry)}")
    print(
        "Floor severity counts: "
        + ", ".join(
            f"{key}={severity_counts.get(key, 0)}"
            for key in [
                "none",
                "low",
                "moderate",
                "high",
            ]
        )
    )
    print(f"Contract: {report_path}")


if __name__ == "__main__":
    main()
