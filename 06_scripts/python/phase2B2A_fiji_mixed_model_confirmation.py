#!/usr/bin/env python3

from __future__ import annotations

import sys
import warnings
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from scipy.stats import norm
except ImportError as exc:
    sys.exit(
        "ERROR: numpy, pandas, scipy and statsmodels are required.\n"
        "Install them with:\n"
        "  python -m pip install --user numpy pandas scipy statsmodels\n"
        f"Original error: {exc}"
    )


ROOT = Path("/mnt/d/HPV_Vaccine_Trafficome_Project")
PROCESSED = ROOT / "07_data_processed" / "fiji_nct02276521"
TABLES = ROOT / "08_results" / "tables"
REPORTS = (
    ROOT
    / "02_dataset_audit"
    / "hpv_specific"
    / "fiji_nct02276521"
)

LONG_INPUT = PROCESSED / "phase2A_fiji_log2_long_analysis_ready.tsv"
PHASE2B1_DECISION = TABLES / "phase2B1_fiji_core_inference_decision.tsv"
PHASE2B1_WITHIN = TABLES / "phase2B1_fiji_within_trajectory_tests.tsv"

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
    frame.to_csv(path, sep="\t", index=False, na_rep="")


def bh_adjust(values: pd.Series) -> pd.Series:
    p = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    adjusted = np.full(len(p), np.nan)
    valid = np.where(np.isfinite(p))[0]

    if len(valid) == 0:
        return pd.Series(adjusted, index=values.index)

    ordered_index = np.argsort(p[valid])
    ordered_p = p[valid][ordered_index]
    count = len(ordered_p)

    ordered_q = ordered_p * count / np.arange(1, count + 1)
    ordered_q = np.minimum.accumulate(ordered_q[::-1])[::-1]
    ordered_q = np.minimum(ordered_q, 1.0)

    restored = np.empty_like(ordered_index)
    restored[ordered_index] = np.arange(count)
    adjusted[valid] = ordered_q[restored]

    return pd.Series(adjusted, index=values.index)


def apply_bh(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["q_value"] = np.nan

    for _, index in output.groupby("bh_family", dropna=False).groups.items():
        output.loc[index, "q_value"] = bh_adjust(
            output.loc[index, "p_value"]
        )

    output["fdr_significant_0_05"] = np.where(
        output["q_value"] < 0.05,
        "yes",
        "no",
    )
    return output


def build_design(frame: pd.DataFrame) -> pd.DataFrame:
    dose = frame["previous_4vHPV_doses"].astype(int)
    visit_v2 = (frame["visit"].astype(str) == "v2").astype(float)

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

    design["visit_v2_dose1"] = design["visit_v2"] * design["dose1"]
    design["visit_v2_dose2"] = design["visit_v2"] * design["dose2"]
    design["visit_v2_dose3"] = design["visit_v2"] * design["dose3"]

    return design[DESIGN_COLUMNS]


def fit_model(
    outcome: pd.Series,
    design: pd.DataFrame,
    groups: pd.Series,
) -> dict[str, object]:
    warning_messages: list[str] = []

    for method in ["lbfgs", "powell"]:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")

                model = sm.MixedLM(
                    endog=outcome,
                    exog=design,
                    groups=groups,
                )

                result = model.fit(
                    reml=False,
                    method=method,
                    maxiter=1000,
                    disp=False,
                )

                warning_messages.extend(
                    str(item.message)
                    for item in caught
                )

            fixed = pd.Series(
                np.asarray(result.fe_params),
                index=DESIGN_COLUMNS,
                dtype=float,
            )

            covariance = result.cov_params().loc[
                DESIGN_COLUMNS,
                DESIGN_COLUMNS,
            ].astype(float)

            if np.isfinite(fixed).all() and np.isfinite(covariance).all().all():
                random_variance = np.nan

                try:
                    random_variance = float(
                        np.asarray(result.cov_re)[0, 0]
                    )
                except Exception:
                    pass

                return {
                    "status": "success",
                    "method": f"mixedlm_{method}",
                    "converged": bool(
                        getattr(result, "converged", False)
                    ),
                    "fixed": fixed,
                    "covariance": covariance,
                    "log_likelihood": float(result.llf),
                    "residual_variance": float(result.scale),
                    "random_intercept_variance": random_variance,
                    "warnings": " | ".join(sorted(set(warning_messages))),
                }

        except Exception as exc:
            warning_messages.append(
                f"mixedlm_{method}: {type(exc).__name__}: {exc}"
            )

    try:
        result = sm.OLS(
            outcome,
            design,
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": groups},
        )

        fixed = pd.Series(
            np.asarray(result.params),
            index=DESIGN_COLUMNS,
            dtype=float,
        )

        covariance = pd.DataFrame(
            np.asarray(result.cov_params()),
            index=DESIGN_COLUMNS,
            columns=DESIGN_COLUMNS,
        )

        return {
            "status": "success",
            "method": "ols_cluster_robust_fallback",
            "converged": True,
            "fixed": fixed,
            "covariance": covariance,
            "log_likelihood": float(result.llf),
            "residual_variance": float(result.scale),
            "random_intercept_variance": np.nan,
            "warnings": " | ".join(sorted(set(warning_messages))),
        }

    except Exception as exc:
        warning_messages.append(
            f"ols_cluster_fallback: {type(exc).__name__}: {exc}"
        )

    return {
        "status": "failed",
        "method": "none",
        "converged": False,
        "fixed": None,
        "covariance": None,
        "log_likelihood": np.nan,
        "residual_variance": np.nan,
        "random_intercept_variance": np.nan,
        "warnings": " | ".join(sorted(set(warning_messages))),
    }


def contrast_result(
    fixed: pd.Series,
    covariance: pd.DataFrame,
    vector: list[float],
) -> dict[str, float]:
    c = np.asarray(vector, dtype=float)
    beta = fixed.loc[DESIGN_COLUMNS].to_numpy(dtype=float)
    cov = covariance.loc[
        DESIGN_COLUMNS,
        DESIGN_COLUMNS,
    ].to_numpy(dtype=float)

    estimate = float(c @ beta)
    variance = float(c @ cov @ c)
    variance = max(variance, 0.0)
    standard_error = float(np.sqrt(variance))

    if standard_error == 0:
        statistic = (
            0.0
            if estimate == 0
            else np.sign(estimate) * np.inf
        )
        p_value = 1.0 if estimate == 0 else 0.0
    else:
        statistic = estimate / standard_error
        p_value = float(
            2 * norm.sf(abs(statistic))
        )

    return {
        "estimate_log2": estimate,
        "standard_error": standard_error,
        "ci95_lower": estimate - 1.959963984540054 * standard_error,
        "ci95_upper": estimate + 1.959963984540054 * standard_error,
        "test_statistic": statistic,
        "p_value": p_value,
    }


def model_identity(
    antigen: str,
    outcome_family: str,
    contrast_type: str,
    dose: int | None,
) -> tuple[str, str]:
    if antigen == "BPV":
        return "HPVVT-FJ-M06", "Heterologous BPV control"

    if contrast_type == "within_trajectory":
        if dose == 0:
            return "HPVVT-FJ-M01", "Primary 2vHPV induction"
        return "HPVVT-FJ-M03", "Heterologous 2vHPV recall"

    if contrast_type == "primary_minus_pooled_recall":
        return "HPVVT-FJ-M04", "Primary-versus-recall contrast"

    if contrast_type == "persistence_dose_contrast":
        return "HPVVT-FJ-M02", "Six-year 4vHPV persistence"

    if contrast_type == "recall_dose_contrast":
        return "HPVVT-FJ-M03", "Heterologous 2vHPV recall"

    return "UNRESOLVED", outcome_family


def main() -> None:
    for path in [
        LONG_INPUT,
        PHASE2B1_DECISION,
        PHASE2B1_WITHIN,
    ]:
        if not path.exists():
            sys.exit(f"ERROR: Required input missing: {path}")

    decision = pd.read_csv(
        PHASE2B1_DECISION,
        sep="\t",
    )

    if str(decision.loc[0, "decision"]) != (
        "READY_FOR_PHASE2B2_MIXED_MODEL_AND_FLOOR_SENSITIVITY"
    ):
        sys.exit("ERROR: Phase 2B1 does not authorize Phase 2B2A.")

    long_df = pd.read_csv(
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

    required = {
        "participant_id",
        "previous_4vHPV_doses",
        "visit",
        "antigen_target",
        "antigen_class",
        "feature",
        "assay_family",
        "outcome_family",
        "log2_value",
        "floor_severity",
    }

    missing = required - set(long_df.columns)

    if missing:
        sys.exit(
            "ERROR: Missing columns: "
            + ", ".join(sorted(missing))
        )

    coefficient_rows: list[dict[str, object]] = []
    contrast_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []

    contrast_specs = [
        (
            "within_trajectory",
            "dose0_v2_minus_v1",
            0,
            [0, 1, 0, 0, 0, 0, 0, 0],
        ),
        (
            "within_trajectory",
            "dose1_v2_minus_v1",
            1,
            [0, 1, 0, 0, 0, 1, 0, 0],
        ),
        (
            "within_trajectory",
            "dose2_v2_minus_v1",
            2,
            [0, 1, 0, 0, 0, 0, 1, 0],
        ),
        (
            "within_trajectory",
            "dose3_v2_minus_v1",
            3,
            [0, 1, 0, 0, 0, 0, 0, 1],
        ),
        (
            "primary_minus_pooled_recall",
            "dose0_minus_mean_dose1_2_3_change",
            None,
            [0, 0, 0, 0, 0, -1 / 3, -1 / 3, -1 / 3],
        ),
        (
            "persistence_dose_contrast",
            "dose2_minus_dose1_at_v1",
            None,
            [0, 0, -1, 1, 0, 0, 0, 0],
        ),
        (
            "persistence_dose_contrast",
            "dose3_minus_dose1_at_v1",
            None,
            [0, 0, -1, 0, 1, 0, 0, 0],
        ),
        (
            "persistence_dose_contrast",
            "dose3_minus_dose2_at_v1",
            None,
            [0, 0, 0, -1, 1, 0, 0, 0],
        ),
        (
            "recall_dose_contrast",
            "dose2_minus_dose1_change",
            None,
            [0, 0, 0, 0, 0, -1, 1, 0],
        ),
        (
            "recall_dose_contrast",
            "dose3_minus_dose1_change",
            None,
            [0, 0, 0, 0, 0, -1, 0, 1],
        ),
        (
            "recall_dose_contrast",
            "dose3_minus_dose2_change",
            None,
            [0, 0, 0, 0, 0, 0, -1, 1],
        ),
    ]

    for (
        antigen,
        feature,
    ), group in long_df.groupby(
        ["antigen_target", "feature"],
        dropna=False,
        observed=True,
    ):
        group = group.copy()
        outcome = pd.to_numeric(
            group["log2_value"],
            errors="coerce",
        )

        design = build_design(group)
        fit = fit_model(
            outcome=outcome,
            design=design,
            groups=group["participant_id"],
        )

        metadata = {
            "antigen_target": str(antigen),
            "antigen_class": str(group["antigen_class"].iloc[0]),
            "feature": str(feature),
            "assay_family": str(group["assay_family"].iloc[0]),
            "outcome_family": str(group["outcome_family"].iloc[0]),
            "maximum_floor_severity": str(
                group["floor_severity"].iloc[
                    group["floor_severity"].map(
                        {
                            "none": 0,
                            "low": 1,
                            "moderate": 2,
                            "high": 3,
                        }
                    ).argmax()
                ]
            ),
        }

        diagnostic_rows.append(
            {
                **metadata,
                "observations": len(group),
                "participants": group["participant_id"].nunique(),
                "fit_status": fit["status"],
                "fit_method": fit["method"],
                "converged": fit["converged"],
                "log_likelihood": fit["log_likelihood"],
                "residual_variance": fit["residual_variance"],
                "random_intercept_variance": (
                    fit["random_intercept_variance"]
                ),
                "warnings": fit["warnings"],
            }
        )

        if fit["status"] != "success":
            continue

        fixed = fit["fixed"]
        covariance = fit["covariance"]

        for coefficient in DESIGN_COLUMNS:
            standard_error = float(
                np.sqrt(
                    max(
                        covariance.loc[
                            coefficient,
                            coefficient,
                        ],
                        0.0,
                    )
                )
            )

            estimate = float(fixed[coefficient])

            statistic = (
                estimate / standard_error
                if standard_error > 0
                else np.nan
            )

            p_value = (
                float(2 * norm.sf(abs(statistic)))
                if np.isfinite(statistic)
                else np.nan
            )

            coefficient_rows.append(
                {
                    **metadata,
                    "fit_method": fit["method"],
                    "coefficient": coefficient,
                    "estimate": estimate,
                    "standard_error": standard_error,
                    "test_statistic": statistic,
                    "p_value": p_value,
                }
            )

        for (
            contrast_type,
            contrast_label,
            dose,
            vector,
        ) in contrast_specs:
            result = contrast_result(
                fixed,
                covariance,
                vector,
            )

            model_id, model_family = model_identity(
                str(antigen),
                metadata["outcome_family"],
                contrast_type,
                dose,
            )

            contrast_rows.append(
                {
                    "model_id": model_id,
                    "model_family": model_family,
                    "contrast_type": contrast_type,
                    "contrast_label": contrast_label,
                    "previous_4vHPV_doses": (
                        dose
                        if dose is not None
                        else ""
                    ),
                    **metadata,
                    "fit_method": fit["method"],
                    **result,
                    "ratio_effect": (
                        2 ** result["estimate_log2"]
                    ),
                    "ratio_ci95_lower": (
                        2 ** result["ci95_lower"]
                    ),
                    "ratio_ci95_upper": (
                        2 ** result["ci95_upper"]
                    ),
                    "bh_family": (
                        f"{model_id}|"
                        f"{metadata['outcome_family']}|"
                        f"{contrast_type}"
                    ),
                }
            )

    coefficients = pd.DataFrame(coefficient_rows)
    contrasts = apply_bh(
        pd.DataFrame(contrast_rows)
    )
    diagnostics = pd.DataFrame(diagnostic_rows)

    phase2b1_within = pd.read_csv(
        PHASE2B1_WITHIN,
        sep="\t",
    )

    mixed_within = contrasts[
        contrasts["contrast_type"] == "within_trajectory"
    ].copy()

    mixed_within["previous_4vHPV_doses"] = pd.to_numeric(
        mixed_within["previous_4vHPV_doses"],
        errors="coerce",
    ).astype("Int64")

    confirmation = mixed_within.merge(
        phase2b1_within[
            [
                "antigen_target",
                "feature",
                "previous_4vHPV_doses",
                "mean_log2_change",
                "p_value",
                "q_value",
            ]
        ].rename(
            columns={
                "mean_log2_change": "phase2B1_mean_log2_change",
                "p_value": "phase2B1_p_value",
                "q_value": "phase2B1_q_value",
            }
        ),
        on=[
            "antigen_target",
            "feature",
            "previous_4vHPV_doses",
        ],
        how="left",
        validate="one_to_one",
    )

    confirmation["effect_difference"] = (
        confirmation["estimate_log2"]
        - confirmation["phase2B1_mean_log2_change"]
    )

    confirmation["effect_direction_agreement"] = np.where(
        np.sign(confirmation["estimate_log2"])
        == np.sign(confirmation["phase2B1_mean_log2_change"]),
        "yes",
        "no",
    )

    expected_models = 92
    expected_contrasts = 92 * 11

    failed_models = int(
        (diagnostics["fit_status"] != "success").sum()
    )

    fallback_models = int(
        (
            diagnostics["fit_method"]
            == "ols_cluster_robust_fallback"
        ).sum()
    )

    failures: list[str] = []

    if len(diagnostics) != expected_models:
        failures.append(
            f"Expected {expected_models} models, observed {len(diagnostics)}."
        )

    if failed_models:
        failures.append(
            f"{failed_models} models failed."
        )

    if len(contrasts) != expected_contrasts:
        failures.append(
            f"Expected {expected_contrasts} contrasts, "
            f"observed {len(contrasts)}."
        )

    if confirmation["phase2B1_mean_log2_change"].isna().any():
        failures.append(
            "Some Phase 2B1 within-trajectory results did not merge."
        )

    decision_value = (
        "READY_FOR_PHASE2B2B_FLOOR_SENSITIVITY"
        if not failures
        else "PHASE2B2A_REPAIR_REQUIRED"
    )

    decision_frame = pd.DataFrame(
        [
            {
                "decision": decision_value,
                "models_attempted": len(diagnostics),
                "mixed_models_completed": (
                    len(diagnostics)
                    - failed_models
                    - fallback_models
                ),
                "fallback_models": fallback_models,
                "failed_models": failed_models,
                "coefficient_rows": len(coefficients),
                "contrast_rows": len(contrasts),
                "within_confirmation_rows": len(confirmation),
                "effect_direction_agreement_rows": int(
                    (
                        confirmation["effect_direction_agreement"]
                        == "yes"
                    ).sum()
                ),
                "validation_failures": "; ".join(failures),
            }
        ]
    )

    write_tsv(
        diagnostics,
        TABLES / "phase2B2A_fiji_model_diagnostics.tsv",
    )

    write_tsv(
        coefficients,
        TABLES / "phase2B2A_fiji_mixed_model_coefficients.tsv",
    )

    write_tsv(
        contrasts,
        TABLES / "phase2B2A_fiji_mixed_model_contrasts.tsv",
    )

    write_tsv(
        confirmation,
        TABLES / "phase2B2A_fiji_within_effect_confirmation.tsv",
    )

    write_tsv(
        decision_frame,
        TABLES / "phase2B2A_fiji_mixed_model_decision.tsv",
    )

    report_path = (
        REPORTS
        / "phase2B2A_fiji_mixed_model_confirmation_report.md"
    )

    with report_path.open("w", encoding="utf-8") as report:
        report.write(
            "# Phase 2B2A Fiji mixed-model confirmation\n\n"
        )
        report.write("## Decision\n\n")
        report.write(f"**{decision_value}**\n\n")
        report.write(
            f"- Models attempted: {len(diagnostics)}\n"
        )
        report.write(
            f"- Cluster-robust fallback models: {fallback_models}\n"
        )
        report.write(
            f"- Failed models: {failed_models}\n"
        )
        report.write(
            f"- Prespecified contrast rows: {len(contrasts)}\n"
        )
        report.write(
            f"- Phase 2B1 within-effect confirmation rows: "
            f"{len(confirmation)}\n\n"
        )
        report.write(
            "Each antigen-feature model used a participant-specific "
            "random intercept and explicit visit-by-dose interaction. "
            "Cluster-robust ordinary least squares was retained only "
            "as a documented fallback when mixed-model estimation "
            "was numerically unstable.\n"
        )

    print("===== PHASE 2B2A COMPLETE =====")
    print(f"Decision: {decision_value}")
    print(f"Models attempted: {len(diagnostics)}")
    print(f"Fallback models: {fallback_models}")
    print(f"Failed models: {failed_models}")
    print(f"Contrasts: {len(contrasts)}")
    print(f"Report: {report_path}")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
