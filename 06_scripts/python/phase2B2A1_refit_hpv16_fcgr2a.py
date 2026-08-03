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

LONG_INPUT = (
    PROCESSED
    / "phase2A_fiji_log2_long_analysis_ready.tsv"
)

ORIGINAL_DIAGNOSTICS = (
    TABLES
    / "phase2B2A_fiji_model_diagnostics.tsv"
)

ORIGINAL_COEFFICIENTS = (
    TABLES
    / "phase2B2A_fiji_mixed_model_coefficients.tsv"
)

ORIGINAL_CONTRASTS = (
    TABLES
    / "phase2B2A_fiji_mixed_model_contrasts.tsv"
)

PHASE2B1_WITHIN = (
    TABLES
    / "phase2B1_fiji_within_trajectory_tests.tsv"
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

OPTIMIZERS = [
    "powell",
    "bfgs",
    "cg",
    "nm",
    "lbfgs",
]


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


def bh_adjust(
    values: pd.Series,
) -> pd.Series:
    p_values = pd.to_numeric(
        values,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    adjusted = np.full(
        len(p_values),
        np.nan,
        dtype=float,
    )

    valid = np.where(
        np.isfinite(p_values)
    )[0]

    if len(valid) == 0:
        return pd.Series(
            adjusted,
            index=values.index,
        )

    ordering = np.argsort(
        p_values[valid]
    )

    ordered_p = p_values[
        valid
    ][ordering]

    count = len(
        ordered_p
    )

    ordered_q = (
        ordered_p
        * count
        / np.arange(
            1,
            count + 1,
        )
    )

    ordered_q = (
        np.minimum.accumulate(
            ordered_q[::-1]
        )[::-1]
    )

    ordered_q = np.minimum(
        ordered_q,
        1.0,
    )

    inverse = np.empty_like(
        ordering
    )

    inverse[ordering] = np.arange(
        count
    )

    adjusted[valid] = (
        ordered_q[inverse]
    )

    return pd.Series(
        adjusted,
        index=values.index,
    )


def apply_bh(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    output = frame.copy()

    output["q_value"] = np.nan

    for _, indices in output.groupby(
        "bh_family",
        dropna=False,
    ).groups.items():
        output.loc[
            indices,
            "q_value",
        ] = bh_adjust(
            output.loc[
                indices,
                "p_value",
            ]
        )

    output[
        "fdr_significant_0_05"
    ] = np.where(
        output["q_value"] < 0.05,
        "yes",
        "no",
    )

    return output


def build_design(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    dose = (
        frame[
            "previous_4vHPV_doses"
        ]
        .astype(int)
    )

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
            "dose1": (
                dose == 1
            ).astype(float),
            "dose2": (
                dose == 2
            ).astype(float),
            "dose3": (
                dose == 3
            ).astype(float),
        },
        index=frame.index,
    )

    design[
        "visit_v2_dose1"
    ] = (
        design["visit_v2"]
        * design["dose1"]
    )

    design[
        "visit_v2_dose2"
    ] = (
        design["visit_v2"]
        * design["dose2"]
    )

    design[
        "visit_v2_dose3"
    ] = (
        design["visit_v2"]
        * design["dose3"]
    )

    return design[
        DESIGN_COLUMNS
    ]


def fixed_covariance(
    result,
) -> pd.DataFrame:
    covariance = result.cov_params()

    if isinstance(
        covariance,
        pd.DataFrame,
    ):
        return covariance.loc[
            DESIGN_COLUMNS,
            DESIGN_COLUMNS,
        ].astype(float)

    array = np.asarray(
        covariance,
        dtype=float,
    )

    fixed_count = len(
        DESIGN_COLUMNS
    )

    return pd.DataFrame(
        array[
            :fixed_count,
            :fixed_count,
        ],
        index=DESIGN_COLUMNS,
        columns=DESIGN_COLUMNS,
    )


def fit_optimizer(
    outcome: pd.Series,
    design: pd.DataFrame,
    groups: pd.Series,
    optimizer: str,
) -> tuple[
    dict[str, object],
    object | None,
]:
    warning_messages: list[str] = []

    trial = {
        "optimizer": optimizer,
        "fit_completed": "no",
        "converged": "no",
        "acceptable": "no",
        "log_likelihood": np.nan,
        "gradient_warning": "",
        "minimum_fixed_covariance_eigenvalue": np.nan,
        "random_intercept_variance": np.nan,
        "residual_variance": np.nan,
        "warnings": "",
        "error": "",
    }

    try:
        with warnings.catch_warnings(
            record=True
        ) as caught:
            warnings.simplefilter(
                "always"
            )

            model = sm.MixedLM(
                endog=outcome,
                exog=design,
                groups=groups,
            )

            result = model.fit(
                reml=False,
                method=optimizer,
                maxiter=5000,
                full_output=True,
                disp=False,
            )

            warning_messages = [
                str(item.message)
                for item in caught
            ]

        fixed = pd.Series(
            np.asarray(
                result.fe_params
            ),
            index=DESIGN_COLUMNS,
            dtype=float,
        )

        covariance = fixed_covariance(
            result
        )

        eigenvalues = np.linalg.eigvalsh(
            covariance.to_numpy(
                dtype=float
            )
        )

        minimum_eigenvalue = float(
            np.min(eigenvalues)
        )

        random_variance = np.nan

        try:
            random_variance = float(
                np.asarray(
                    result.cov_re
                )[0, 0]
            )
        except Exception:
            pass

        converged = bool(
            getattr(
                result,
                "converged",
                False,
            )
        )

        finite_fixed = bool(
            np.isfinite(
                fixed
            ).all()
        )

        finite_covariance = bool(
            np.isfinite(
                covariance
            ).all().all()
        )

        covariance_acceptable = (
            minimum_eigenvalue
            >= -1e-8
        )

        acceptable = bool(
            converged
            and finite_fixed
            and finite_covariance
            and covariance_acceptable
        )

        trial.update(
            {
                "fit_completed": "yes",
                "converged": (
                    "yes"
                    if converged
                    else "no"
                ),
                "acceptable": (
                    "yes"
                    if acceptable
                    else "no"
                ),
                "log_likelihood": float(
                    result.llf
                ),
                "minimum_fixed_covariance_eigenvalue": (
                    minimum_eigenvalue
                ),
                "random_intercept_variance": (
                    random_variance
                ),
                "residual_variance": float(
                    result.scale
                ),
                "warnings": " | ".join(
                    sorted(
                        set(
                            warning_messages
                        )
                    )
                ),
            }
        )

        return trial, result

    except Exception as exc:
        trial["error"] = (
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        trial["warnings"] = " | ".join(
            sorted(
                set(
                    warning_messages
                )
            )
        )

        return trial, None


def contrast_result(
    fixed: pd.Series,
    covariance: pd.DataFrame,
    vector: list[float],
) -> dict[str, float]:
    contrast = np.asarray(
        vector,
        dtype=float,
    )

    beta = fixed.loc[
        DESIGN_COLUMNS
    ].to_numpy(
        dtype=float
    )

    cov = covariance.loc[
        DESIGN_COLUMNS,
        DESIGN_COLUMNS,
    ].to_numpy(
        dtype=float
    )

    estimate = float(
        contrast @ beta
    )

    variance = float(
        contrast
        @ cov
        @ contrast
    )

    variance = max(
        variance,
        0.0,
    )

    standard_error = float(
        np.sqrt(
            variance
        )
    )

    if standard_error == 0:
        statistic = (
            0.0
            if estimate == 0
            else np.sign(
                estimate
            )
            * np.inf
        )

        p_value = (
            1.0
            if estimate == 0
            else 0.0
        )

    else:
        statistic = (
            estimate
            / standard_error
        )

        p_value = float(
            2
            * norm.sf(
                abs(
                    statistic
                )
            )
        )

    return {
        "estimate_log2": estimate,
        "standard_error": standard_error,
        "ci95_lower": (
            estimate
            - 1.959963984540054
            * standard_error
        ),
        "ci95_upper": (
            estimate
            + 1.959963984540054
            * standard_error
        ),
        "test_statistic": statistic,
        "p_value": p_value,
        "ratio_effect": (
            2 ** estimate
        ),
        "ratio_ci95_lower": (
            2
            ** (
                estimate
                - 1.959963984540054
                * standard_error
            )
        ),
        "ratio_ci95_upper": (
            2
            ** (
                estimate
                + 1.959963984540054
                * standard_error
            )
        ),
    }


def main() -> None:
    required_files = [
        LONG_INPUT,
        ORIGINAL_DIAGNOSTICS,
        ORIGINAL_COEFFICIENTS,
        ORIGINAL_CONTRASTS,
        PHASE2B1_WITHIN,
    ]

    for required_file in required_files:
        if not required_file.exists():
            sys.exit(
                "ERROR: Missing required "
                f"input: {required_file}"
            )

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

    target = long_df[
        (
            long_df[
                "antigen_target"
            ]
            == TARGET_ANTIGEN
        )
        & (
            long_df[
                "feature"
            ]
            == TARGET_FEATURE
        )
    ].copy()

    if len(target) != 160:
        sys.exit(
            "ERROR: Expected 160 "
            "HPV16-FcgR2A observations; "
            f"observed {len(target)}."
        )

    if (
        target[
            "participant_id"
        ].nunique()
        != 80
    ):
        sys.exit(
            "ERROR: Expected 80 "
            "participants in target model."
        )

    outcome = pd.to_numeric(
        target["log2_value"],
        errors="coerce",
    )

    design = build_design(
        target
    )

    groups = target[
        "participant_id"
    ]

    trial_rows: list[
        dict[str, object]
    ] = []

    acceptable_results: list[
        tuple[
            dict[str, object],
            object,
        ]
    ] = []

    for optimizer in OPTIMIZERS:
        trial, result = fit_optimizer(
            outcome=outcome,
            design=design,
            groups=groups,
            optimizer=optimizer,
        )

        trial_rows.append(
            trial
        )

        if (
            trial[
                "acceptable"
            ]
            == "yes"
            and result is not None
        ):
            acceptable_results.append(
                (
                    trial,
                    result,
                )
            )

    trials = pd.DataFrame(
        trial_rows
    )

    write_tsv(
        trials,
        TABLES
        / "phase2B2A1_fiji_hpv16_fcgr2a_optimizer_trials.tsv",
    )

    if not acceptable_results:
        decision = pd.DataFrame(
            [
                {
                    "decision": (
                        "PHASE2B2A1_REPAIR_REQUIRED"
                    ),
                    "target_antigen": (
                        TARGET_ANTIGEN
                    ),
                    "target_feature": (
                        TARGET_FEATURE
                    ),
                    "acceptable_optimizers": 0,
                    "selected_optimizer": "",
                    "validation_failures": (
                        "No optimizer produced "
                        "an acceptable converged "
                        "mixed model."
                    ),
                }
            ]
        )

        write_tsv(
            decision,
            TABLES
            / "phase2B2A1_fiji_hpv16_fcgr2a_refit_decision.tsv",
        )

        sys.exit(
            "ERROR: No optimizer produced "
            "an acceptable converged model."
        )

    acceptable_results.sort(
        key=lambda item: float(
            item[0][
                "log_likelihood"
            ]
        ),
        reverse=True,
    )

    best_trial, best_result = (
        acceptable_results[0]
    )

    fixed = pd.Series(
        np.asarray(
            best_result.fe_params
        ),
        index=DESIGN_COLUMNS,
        dtype=float,
    )

    covariance = fixed_covariance(
        best_result
    )

    metadata = {
        "antigen_target": (
            TARGET_ANTIGEN
        ),
        "antigen_class": str(
            target[
                "antigen_class"
            ].iloc[0]
        ),
        "feature": (
            TARGET_FEATURE
        ),
        "assay_family": str(
            target[
                "assay_family"
            ].iloc[0]
        ),
        "outcome_family": str(
            target[
                "outcome_family"
            ].iloc[0]
        ),
        "maximum_floor_severity": (
            "low"
        ),
    }

    coefficient_rows: list[
        dict[str, object]
    ] = []

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

        estimate = float(
            fixed[
                coefficient
            ]
        )

        statistic = (
            estimate
            / standard_error
            if standard_error > 0
            else np.nan
        )

        p_value = (
            float(
                2
                * norm.sf(
                    abs(
                        statistic
                    )
                )
            )
            if np.isfinite(
                statistic
            )
            else np.nan
        )

        coefficient_rows.append(
            {
                **metadata,
                "fit_method": (
                    "mixedlm_"
                    + str(
                        best_trial[
                            "optimizer"
                        ]
                    )
                    + "_convergence_repair"
                ),
                "coefficient": (
                    coefficient
                ),
                "estimate": estimate,
                "standard_error": (
                    standard_error
                ),
                "test_statistic": (
                    statistic
                ),
                "p_value": p_value,
            }
        )

    replacement_coefficients = (
        pd.DataFrame(
            coefficient_rows
        )
    )

    contrast_specs = [
        (
            "HPVVT-FJ-M01",
            "Primary 2vHPV induction",
            "within_trajectory",
            "dose0_v2_minus_v1",
            0,
            [0, 1, 0, 0, 0, 0, 0, 0],
        ),
        (
            "HPVVT-FJ-M03",
            "Heterologous 2vHPV recall",
            "within_trajectory",
            "dose1_v2_minus_v1",
            1,
            [0, 1, 0, 0, 0, 1, 0, 0],
        ),
        (
            "HPVVT-FJ-M03",
            "Heterologous 2vHPV recall",
            "within_trajectory",
            "dose2_v2_minus_v1",
            2,
            [0, 1, 0, 0, 0, 0, 1, 0],
        ),
        (
            "HPVVT-FJ-M03",
            "Heterologous 2vHPV recall",
            "within_trajectory",
            "dose3_v2_minus_v1",
            3,
            [0, 1, 0, 0, 0, 0, 0, 1],
        ),
        (
            "HPVVT-FJ-M04",
            "Primary-versus-recall contrast",
            "primary_minus_pooled_recall",
            "dose0_minus_mean_dose1_2_3_change",
            None,
            [0, 0, 0, 0, 0, -1 / 3, -1 / 3, -1 / 3],
        ),
        (
            "HPVVT-FJ-M02",
            "Six-year 4vHPV persistence",
            "persistence_dose_contrast",
            "dose2_minus_dose1_at_v1",
            None,
            [0, 0, -1, 1, 0, 0, 0, 0],
        ),
        (
            "HPVVT-FJ-M02",
            "Six-year 4vHPV persistence",
            "persistence_dose_contrast",
            "dose3_minus_dose1_at_v1",
            None,
            [0, 0, -1, 0, 1, 0, 0, 0],
        ),
        (
            "HPVVT-FJ-M02",
            "Six-year 4vHPV persistence",
            "persistence_dose_contrast",
            "dose3_minus_dose2_at_v1",
            None,
            [0, 0, 0, -1, 1, 0, 0, 0],
        ),
        (
            "HPVVT-FJ-M03",
            "Heterologous 2vHPV recall",
            "recall_dose_contrast",
            "dose2_minus_dose1_change",
            None,
            [0, 0, 0, 0, 0, -1, 1, 0],
        ),
        (
            "HPVVT-FJ-M03",
            "Heterologous 2vHPV recall",
            "recall_dose_contrast",
            "dose3_minus_dose1_change",
            None,
            [0, 0, 0, 0, 0, -1, 0, 1],
        ),
        (
            "HPVVT-FJ-M03",
            "Heterologous 2vHPV recall",
            "recall_dose_contrast",
            "dose3_minus_dose2_change",
            None,
            [0, 0, 0, 0, 0, 0, -1, 1],
        ),
    ]

    contrast_rows: list[
        dict[str, object]
    ] = []

    for (
        model_id,
        model_family,
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

        contrast_rows.append(
            {
                "model_id": (
                    model_id
                ),
                "model_family": (
                    model_family
                ),
                "contrast_type": (
                    contrast_type
                ),
                "contrast_label": (
                    contrast_label
                ),
                "previous_4vHPV_doses": (
                    dose
                    if dose is not None
                    else np.nan
                ),
                **metadata,
                "fit_method": (
                    "mixedlm_"
                    + str(
                        best_trial[
                            "optimizer"
                        ]
                    )
                    + "_convergence_repair"
                ),
                **result,
                "bh_family": (
                    f"{model_id}|"
                    f"{metadata['outcome_family']}|"
                    f"{contrast_type}"
                ),
            }
        )

    replacement_contrasts = (
        pd.DataFrame(
            contrast_rows
        )
    )

    original_diagnostics = pd.read_csv(
        ORIGINAL_DIAGNOSTICS,
        sep="\t",
    )

    repaired_diagnostics = (
        original_diagnostics[
            ~(
                (
                    original_diagnostics[
                        "antigen_target"
                    ]
                    == TARGET_ANTIGEN
                )
                & (
                    original_diagnostics[
                        "feature"
                    ]
                    == TARGET_FEATURE
                )
            )
        ]
        .copy()
    )

    repaired_diagnostic_row = pd.DataFrame(
        [
            {
                **metadata,
                "observations": len(
                    target
                ),
                "participants": int(
                    target[
                        "participant_id"
                    ].nunique()
                ),
                "fit_status": (
                    "success"
                ),
                "fit_method": (
                    "mixedlm_"
                    + str(
                        best_trial[
                            "optimizer"
                        ]
                    )
                    + "_convergence_repair"
                ),
                "converged": True,
                "log_likelihood": float(
                    best_result.llf
                ),
                "residual_variance": float(
                    best_result.scale
                ),
                "random_intercept_variance": float(
                    np.asarray(
                        best_result.cov_re
                    )[0, 0]
                ),
                "warnings": str(
                    best_trial[
                        "warnings"
                    ]
                ),
            }
        ]
    )

    repaired_diagnostics = pd.concat(
        [
            repaired_diagnostics,
            repaired_diagnostic_row,
        ],
        ignore_index=True,
        sort=False,
    )

    repaired_diagnostics = (
        repaired_diagnostics.sort_values(
            [
                "antigen_target",
                "feature",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    original_coefficients = pd.read_csv(
        ORIGINAL_COEFFICIENTS,
        sep="\t",
    )

    repaired_coefficients = (
        original_coefficients[
            ~(
                (
                    original_coefficients[
                        "antigen_target"
                    ]
                    == TARGET_ANTIGEN
                )
                & (
                    original_coefficients[
                        "feature"
                    ]
                    == TARGET_FEATURE
                )
            )
        ]
        .copy()
    )

    repaired_coefficients = pd.concat(
        [
            repaired_coefficients,
            replacement_coefficients,
        ],
        ignore_index=True,
        sort=False,
    )

    repaired_coefficients = (
        repaired_coefficients.sort_values(
            [
                "antigen_target",
                "feature",
                "coefficient",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    original_contrasts = pd.read_csv(
        ORIGINAL_CONTRASTS,
        sep="\t",
    )

    repaired_contrasts = (
        original_contrasts[
            ~(
                (
                    original_contrasts[
                        "antigen_target"
                    ]
                    == TARGET_ANTIGEN
                )
                & (
                    original_contrasts[
                        "feature"
                    ]
                    == TARGET_FEATURE
                )
            )
        ]
        .copy()
    )

    repaired_contrasts = pd.concat(
        [
            repaired_contrasts,
            replacement_contrasts,
        ],
        ignore_index=True,
        sort=False,
    )

    repaired_contrasts = apply_bh(
        repaired_contrasts
    )

    repaired_contrasts = (
        repaired_contrasts.sort_values(
            [
                "antigen_target",
                "feature",
                "contrast_type",
                "contrast_label",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    phase2b1_within = pd.read_csv(
        PHASE2B1_WITHIN,
        sep="\t",
    )

    repaired_within = repaired_contrasts[
        repaired_contrasts[
            "contrast_type"
        ]
        == "within_trajectory"
    ].copy()

    repaired_within[
        "previous_4vHPV_doses"
    ] = pd.to_numeric(
        repaired_within[
            "previous_4vHPV_doses"
        ],
        errors="coerce",
    ).astype(
        "Int64"
    )

    phase2b1_within[
        "previous_4vHPV_doses"
    ] = pd.to_numeric(
        phase2b1_within[
            "previous_4vHPV_doses"
        ],
        errors="coerce",
    ).astype(
        "Int64"
    )

    repaired_confirmation = (
        repaired_within.merge(
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
                    "mean_log2_change": (
                        "phase2B1_mean_log2_change"
                    ),
                    "p_value": (
                        "phase2B1_p_value"
                    ),
                    "q_value": (
                        "phase2B1_q_value"
                    ),
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
    )

    repaired_confirmation[
        "effect_difference"
    ] = (
        repaired_confirmation[
            "estimate_log2"
        ]
        - repaired_confirmation[
            "phase2B1_mean_log2_change"
        ]
    )

    repaired_confirmation[
        "effect_direction_agreement"
    ] = np.where(
        np.sign(
            repaired_confirmation[
                "estimate_log2"
            ]
        )
        == np.sign(
            repaired_confirmation[
                "phase2B1_mean_log2_change"
            ]
        ),
        "yes",
        "no",
    )

    failures: list[str] = []

    if len(
        repaired_diagnostics
    ) != 92:
        failures.append(
            "Repaired diagnostics "
            "does not contain 92 rows."
        )

    remaining_nonconverged = int(
        (
            repaired_diagnostics[
                "converged"
            ]
            .astype(str)
            .str.lower()
            != "true"
        ).sum()
    )

    if remaining_nonconverged:
        failures.append(
            f"{remaining_nonconverged} "
            "models remain nonconverged."
        )

    if len(
        repaired_coefficients
    ) != 736:
        failures.append(
            "Repaired coefficient "
            "table does not contain "
            "736 rows."
        )

    if len(
        repaired_contrasts
    ) != 1012:
        failures.append(
            "Repaired contrast table "
            "does not contain 1012 rows."
        )

    if len(
        repaired_confirmation
    ) != 368:
        failures.append(
            "Repaired within-effect "
            "confirmation does not "
            "contain 368 rows."
        )

    if repaired_contrasts[
        "p_value"
    ].isna().any():
        failures.append(
            "Repaired contrasts contain "
            "missing P values."
        )

    if repaired_contrasts[
        "q_value"
    ].isna().any():
        failures.append(
            "Repaired contrasts contain "
            "missing q values."
        )

    decision_value = (
        "READY_FOR_PHASE2B2D_INTEGRATION_AND_BIOLOGICAL_SYNTHESIS"
        if not failures
        else "PHASE2B2A1_REPAIR_REQUIRED"
    )

    decision_frame = pd.DataFrame(
        [
            {
                "decision": (
                    decision_value
                ),
                "target_antigen": (
                    TARGET_ANTIGEN
                ),
                "target_feature": (
                    TARGET_FEATURE
                ),
                "optimizers_attempted": (
                    len(
                        trials
                    )
                ),
                "acceptable_optimizers": (
                    len(
                        acceptable_results
                    )
                ),
                "selected_optimizer": str(
                    best_trial[
                        "optimizer"
                    ]
                ),
                "selected_log_likelihood": float(
                    best_trial[
                        "log_likelihood"
                    ]
                ),
                "selected_random_intercept_variance": float(
                    best_trial[
                        "random_intercept_variance"
                    ]
                ),
                "remaining_nonconverged_models": (
                    remaining_nonconverged
                ),
                "repaired_coefficient_rows": (
                    len(
                        repaired_coefficients
                    )
                ),
                "repaired_contrast_rows": (
                    len(
                        repaired_contrasts
                    )
                ),
                "repaired_confirmation_rows": (
                    len(
                        repaired_confirmation
                    )
                ),
                "validation_failures": (
                    "; ".join(
                        failures
                    )
                ),
            }
        ]
    )

    write_tsv(
        repaired_diagnostics,
        TABLES
        / "phase2B2A1_fiji_model_diagnostics_repaired.tsv",
    )

    write_tsv(
        repaired_coefficients,
        TABLES
        / "phase2B2A1_fiji_mixed_model_coefficients_repaired.tsv",
    )

    write_tsv(
        repaired_contrasts,
        TABLES
        / "phase2B2A1_fiji_mixed_model_contrasts_repaired.tsv",
    )

    write_tsv(
        repaired_confirmation,
        TABLES
        / "phase2B2A1_fiji_within_effect_confirmation_repaired.tsv",
    )

    write_tsv(
        decision_frame,
        TABLES
        / "phase2B2A1_fiji_hpv16_fcgr2a_refit_decision.tsv",
    )

    report_path = (
        REPORTS
        / "phase2B2A1_fiji_hpv16_fcgr2a_convergence_repair_report.md"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as report:
        report.write(
            "# Phase 2B2A1 HPV16–FcγR2A convergence repair\n\n"
        )

        report.write(
            "## Decision\n\n"
        )

        report.write(
            f"**{decision_value}**\n\n"
        )

        report.write(
            "The original HPV16–FcγR2A L-BFGS model returned "
            "finite coefficients but did not satisfy the optimizer "
            "convergence criterion. It also produced singular "
            "random-effects and Hessian warnings.\n\n"
        )

        report.write(
            "The model was therefore refitted independently using "
            "multiple optimizers. Only models with explicit optimizer "
            "convergence, finite fixed effects, finite fixed-effect "
            "covariance and an acceptable covariance eigenstructure "
            "were eligible for selection.\n\n"
        )

        report.write(
            f"- Selected optimizer: "
            f"{best_trial['optimizer']}\n"
        )

        report.write(
            f"- Selected log likelihood: "
            f"{best_trial['log_likelihood']}\n"
        )

        report.write(
            f"- Acceptable optimizers: "
            f"{len(acceptable_results)}\n"
        )

        report.write(
            f"- Remaining nonconverged models: "
            f"{remaining_nonconverged}\n"
        )

        report.write(
            f"- Repaired contrast rows: "
            f"{len(repaired_contrasts)}\n\n"
        )

        report.write(
            "All Phase 2B2A multiplicity adjustments were recomputed "
            "after replacing the eleven HPV16–FcγR2A contrasts. "
            "The repaired tables are authoritative for subsequent "
            "integration and biological synthesis.\n"
        )

    print(
        "===== PHASE 2B2A1 COMPLETE ====="
    )

    print(
        f"Decision: {decision_value}"
    )

    print(
        "Selected optimizer: "
        f"{best_trial['optimizer']}"
    )

    print(
        "Acceptable optimizers: "
        f"{len(acceptable_results)}"
    )

    print(
        "Remaining nonconverged models: "
        f"{remaining_nonconverged}"
    )

    print(
        "Repaired contrasts: "
        f"{len(repaired_contrasts)}"
    )

    print(
        f"Report: {report_path}"
    )

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
