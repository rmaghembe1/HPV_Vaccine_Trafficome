#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from scipy.stats import norm
except ImportError as exc:
    sys.exit(
        "ERROR: numpy, pandas, scipy and statsmodels are required.\n"
        f"Original error: {exc}"
    )


ROOT = Path("/mnt/d/HPV_Vaccine_Trafficome_Project")

LONG_INPUT = (
    ROOT
    / "07_data_processed"
    / "fiji_nct02276521"
    / "phase2A_fiji_log2_long_analysis_ready.tsv"
)

REPAIRED_CONTRASTS = (
    ROOT
    / "08_results"
    / "tables"
    / "phase2B2A1_fiji_mixed_model_contrasts_repaired.tsv"
)

TABLES = ROOT / "08_results" / "tables"

REPORT = (
    ROOT
    / "02_dataset_audit"
    / "hpv_specific"
    / "fiji_nct02276521"
    / "phase2B2A2_hpv16_fcgr2a_boundary_sensitivity_report.md"
)

TARGET_ANTIGEN = "HPV16"
TARGET_FEATURE = "FcgR2A"

DESIGN_COLUMNS = [
    "intercept",
    "visit_v2",
    "dose1",
    "dose2",
    "dose3",
    "visit_v2_dose1",
    "visit_v2_dose2",
    "visit_v2_dose3",
]


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        sep="\t",
        index=False,
        na_rep="",
    )


def build_design(frame: pd.DataFrame) -> pd.DataFrame:
    dose = frame["previous_4vHPV_doses"].astype(int)

    visit_v2 = (
        frame["visit"]
        .astype(str)
        .eq("v2")
        .astype(float)
    )

    design = pd.DataFrame(
        {
            "intercept": 1.0,
            "visit_v2": visit_v2,
            "dose1": (dose == 1).astype(float),
            "dose2": (dose == 2).astype(float),
            "dose3": (dose == 3).astype(float),
        },
        index=frame.index,
    )

    design["visit_v2_dose1"] = (
        design["visit_v2"] * design["dose1"]
    )

    design["visit_v2_dose2"] = (
        design["visit_v2"] * design["dose2"]
    )

    design["visit_v2_dose3"] = (
        design["visit_v2"] * design["dose3"]
    )

    return design[DESIGN_COLUMNS]


def evaluate_contrast(
    parameters: pd.Series,
    covariance: pd.DataFrame,
    vector: list[float],
) -> dict[str, float]:
    contrast = np.asarray(vector, dtype=float)

    beta = parameters.loc[
        DESIGN_COLUMNS
    ].to_numpy(dtype=float)

    cov = covariance.loc[
        DESIGN_COLUMNS,
        DESIGN_COLUMNS,
    ].to_numpy(dtype=float)

    estimate = float(contrast @ beta)

    variance = float(
        contrast @ cov @ contrast
    )

    variance = max(
        variance,
        0.0,
    )

    standard_error = float(
        np.sqrt(variance)
    )

    if standard_error == 0:
        statistic = (
            0.0
            if estimate == 0
            else np.sign(estimate) * np.inf
        )

        p_value = (
            1.0
            if estimate == 0
            else 0.0
        )
    else:
        statistic = estimate / standard_error

        p_value = float(
            2 * norm.sf(abs(statistic))
        )

    return {
        "cluster_robust_estimate_log2": estimate,
        "cluster_robust_standard_error": standard_error,
        "cluster_robust_test_statistic": statistic,
        "cluster_robust_p_value": p_value,
        "cluster_robust_ci95_lower": (
            estimate
            - 1.959963984540054
            * standard_error
        ),
        "cluster_robust_ci95_upper": (
            estimate
            + 1.959963984540054
            * standard_error
        ),
    }


def main() -> None:
    for path in [
        LONG_INPUT,
        REPAIRED_CONTRASTS,
    ]:
        if not path.exists():
            sys.exit(
                f"ERROR: Required input missing: {path}"
            )

    long_frame = pd.read_csv(
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

    target = long_frame[
        (
            long_frame["antigen_target"]
            == TARGET_ANTIGEN
        )
        & (
            long_frame["feature"]
            == TARGET_FEATURE
        )
    ].copy()

    if len(target) != 160:
        sys.exit(
            "ERROR: Expected 160 target observations; "
            f"observed {len(target)}."
        )

    if target["participant_id"].nunique() != 80:
        sys.exit(
            "ERROR: Expected 80 target participants."
        )

    outcome = pd.to_numeric(
        target["log2_value"],
        errors="coerce",
    )

    design = build_design(target)

    model = sm.OLS(
        outcome,
        design,
    )

    result = model.fit(
        cov_type="cluster",
        cov_kwds={
            "groups": target["participant_id"],
            "use_correction": True,
        },
    )

    parameters = pd.Series(
        np.asarray(result.params),
        index=DESIGN_COLUMNS,
        dtype=float,
    )

    covariance = pd.DataFrame(
        np.asarray(result.cov_params()),
        index=DESIGN_COLUMNS,
        columns=DESIGN_COLUMNS,
    )

    contrast_specs = [
        (
            "dose0_v2_minus_v1",
            [0, 1, 0, 0, 0, 0, 0, 0],
        ),
        (
            "dose1_v2_minus_v1",
            [0, 1, 0, 0, 0, 1, 0, 0],
        ),
        (
            "dose2_v2_minus_v1",
            [0, 1, 0, 0, 0, 0, 1, 0],
        ),
        (
            "dose3_v2_minus_v1",
            [0, 1, 0, 0, 0, 0, 0, 1],
        ),
        (
            "dose0_minus_mean_dose1_2_3_change",
            [0, 0, 0, 0, 0, -1 / 3, -1 / 3, -1 / 3],
        ),
        (
            "dose2_minus_dose1_at_v1",
            [0, 0, -1, 1, 0, 0, 0, 0],
        ),
        (
            "dose3_minus_dose1_at_v1",
            [0, 0, -1, 0, 1, 0, 0, 0],
        ),
        (
            "dose3_minus_dose2_at_v1",
            [0, 0, 0, -1, 1, 0, 0, 0],
        ),
        (
            "dose2_minus_dose1_change",
            [0, 0, 0, 0, 0, -1, 1, 0],
        ),
        (
            "dose3_minus_dose1_change",
            [0, 0, 0, 0, 0, -1, 0, 1],
        ),
        (
            "dose3_minus_dose2_change",
            [0, 0, 0, 0, 0, 0, -1, 1],
        ),
    ]

    rows: list[dict[str, object]] = []

    for label, vector in contrast_specs:
        rows.append(
            {
                "antigen_target": TARGET_ANTIGEN,
                "feature": TARGET_FEATURE,
                "contrast_label": label,
                **evaluate_contrast(
                    parameters,
                    covariance,
                    vector,
                ),
            }
        )

    cluster_contrasts = pd.DataFrame(rows)

    repaired = pd.read_csv(
        REPAIRED_CONTRASTS,
        sep="\t",
    )

    repaired_target = repaired[
        (
            repaired["antigen_target"]
            == TARGET_ANTIGEN
        )
        & (
            repaired["feature"]
            == TARGET_FEATURE
        )
    ][
        [
            "contrast_label",
            "fit_method",
            "estimate_log2",
            "standard_error",
            "p_value",
            "q_value",
            "fdr_significant_0_05",
        ]
    ].copy()

    comparison = repaired_target.merge(
        cluster_contrasts,
        on="contrast_label",
        how="inner",
        validate="one_to_one",
    )

    comparison["estimate_difference"] = (
        comparison[
            "cluster_robust_estimate_log2"
        ]
        - comparison["estimate_log2"]
    )

    comparison[
        "absolute_estimate_difference"
    ] = comparison[
        "estimate_difference"
    ].abs()

    comparison["direction_agreement"] = np.where(
        np.sign(
            comparison[
                "cluster_robust_estimate_log2"
            ]
        )
        == np.sign(
            comparison["estimate_log2"]
        ),
        "yes",
        "no",
    )

    comparison[
        "nominal_significance_agreement"
    ] = np.where(
        (
            comparison[
                "cluster_robust_p_value"
            ]
            < 0.05
        )
        == (
            comparison["p_value"]
            < 0.05
        ),
        "yes",
        "no",
    )

    failures: list[str] = []

    if len(comparison) != 11:
        failures.append(
            f"Expected 11 contrasts, observed {len(comparison)}."
        )

    numeric_columns = [
        "cluster_robust_estimate_log2",
        "cluster_robust_standard_error",
        "cluster_robust_p_value",
        "estimate_log2",
        "standard_error",
        "p_value",
    ]

    if not np.isfinite(
        comparison[numeric_columns]
        .to_numpy(dtype=float)
    ).all():
        failures.append(
            "One or more contrast statistics are non-finite."
        )

    direction_agreement = int(
        (
            comparison["direction_agreement"]
            == "yes"
        ).sum()
    )

    nominal_agreement = int(
        (
            comparison[
                "nominal_significance_agreement"
            ]
            == "yes"
        ).sum()
    )

    maximum_absolute_difference = float(
        comparison[
            "absolute_estimate_difference"
        ].max()
    )

    if direction_agreement != 11:
        failures.append(
            "Not all contrast directions agree."
        )

    if maximum_absolute_difference > 0.05:
        failures.append(
            "Maximum absolute estimate difference exceeds "
            "0.05 log2 units."
        )

    decision_value = (
        "READY_FOR_PHASE2B2D_INTEGRATION_AND_BIOLOGICAL_SYNTHESIS"
        if not failures
        else "PHASE2B2A2_BOUNDARY_SENSITIVITY_REVIEW_REQUIRED"
    )

    decision = pd.DataFrame(
        [
            {
                "decision": decision_value,
                "target_antigen": TARGET_ANTIGEN,
                "target_feature": TARGET_FEATURE,
                "contrasts_compared": len(comparison),
                "direction_agreement": direction_agreement,
                "nominal_significance_agreement": nominal_agreement,
                "maximum_absolute_estimate_difference": (
                    maximum_absolute_difference
                ),
                "cluster_robust_observations": int(result.nobs),
                "cluster_robust_participants": int(
                    target["participant_id"].nunique()
                ),
                "validation_failures": "; ".join(failures),
            }
        ]
    )

    write_tsv(
        comparison,
        TABLES
        / "phase2B2A2_hpv16_fcgr2a_boundary_sensitivity_contrasts.tsv",
    )

    write_tsv(
        decision,
        TABLES
        / "phase2B2A2_hpv16_fcgr2a_boundary_sensitivity_decision.tsv",
    )

    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT.open(
        "w",
        encoding="utf-8",
    ) as report:
        report.write(
            "# Phase 2B2A2 HPV16–FcγR2A boundary sensitivity\n\n"
        )

        report.write("## Decision\n\n")
        report.write(f"**{decision_value}**\n\n")

        report.write(
            "The repaired mixed model converged at an effectively "
            "zero random-intercept variance, indicating a boundary "
            "solution. The same fixed-effect design was therefore "
            "refitted using participant-clustered robust covariance.\n\n"
        )

        report.write(
            f"- Contrasts compared: {len(comparison)}\n"
        )
        report.write(
            f"- Direction agreement: {direction_agreement}/11\n"
        )
        report.write(
            f"- Nominal significance agreement: "
            f"{nominal_agreement}/11\n"
        )
        report.write(
            f"- Maximum absolute estimate difference: "
            f"{maximum_absolute_difference}\n\n"
        )

        report.write(
            "Because the random-intercept variance approached zero, "
            "the cluster-robust fixed-effects analysis provides the "
            "appropriate sensitivity confirmation. The paired Phase "
            "2B1 analysis remains the primary two-visit inference.\n"
        )

    print("===== PHASE 2B2A2 COMPLETE =====")
    print(f"Decision: {decision_value}")
    print(
        f"Direction agreement: {direction_agreement}/11"
    )
    print(
        "Nominal significance agreement: "
        f"{nominal_agreement}/11"
    )
    print(
        "Maximum absolute estimate difference: "
        f"{maximum_absolute_difference}"
    )
    print(f"Report: {REPORT}")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
