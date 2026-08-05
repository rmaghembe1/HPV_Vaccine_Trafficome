#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import (
    chi2,
    norm,
    pearsonr,
    rankdata,
    t as student_t,
)
from statsmodels.stats.multitest import multipletests


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

ATLAS_INPUT = (
    TABLES
    / "phase2C3A_fiji_functional_coupling_atlas.tsv"
)

WIDE_INPUT = (
    PROCESSED
    / "phase2C3A_fiji_functional_coupling_wide.tsv"
)

C3A_DECISION_INPUT = (
    TABLES
    / "phase2C3A_fiji_functional_coupling_decision.tsv"
)

EXPECTED_C3A_DECISION = (
    "READY_FOR_PHASE2C3B_CONTEXT_AND_PARTIAL_COUPLING_ANALYSIS"
)

ANTIGENS = [
    "HPV16",
    "HPV18",
]

PREDICTORS = [
    "IgG",
    "IgG1",
    "IgG3",
    "FcgR2A",
    "FcgR2B",
    "FcgR3A",
]

FUNCTIONS = [
    "ADCP",
    "nAb",
]

PREDICTOR_REPRESENTATIONS = [
    "raw_predictor",
    "bpv_calibrated_predictor",
]

DOSE_STRATA = [
    "recall_dose1",
    "recall_dose2",
    "recall_dose3",
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
                sorted(missing)
            )
        )


def clip_correlation(
    value: float,
) -> float:
    return float(
        np.clip(
            value,
            -0.999999,
            0.999999,
        )
    )


def fisher_context_difference(
    primary_rho: float,
    primary_n: int,
    recall_rho: float,
    recall_n: int,
) -> dict[str, float]:
    values = [
        primary_rho,
        recall_rho,
    ]

    if (
        primary_n <= 3
        or recall_n <= 3
        or not all(
            np.isfinite(value)
            for value in values
        )
    ):
        return {
            "rho_difference_recall_minus_primary": np.nan,
            "fisher_z_difference": np.nan,
            "standard_error": np.nan,
            "z_statistic": np.nan,
            "p_value": np.nan,
        }

    primary_clipped = clip_correlation(
        primary_rho
    )

    recall_clipped = clip_correlation(
        recall_rho
    )

    primary_z = np.arctanh(
        primary_clipped
    )

    recall_z = np.arctanh(
        recall_clipped
    )

    standard_error = np.sqrt(
        1
        / (
            primary_n
            - 3
        )
        + 1
        / (
            recall_n
            - 3
        )
    )

    fisher_difference = (
        recall_z
        - primary_z
    )

    z_statistic = (
        fisher_difference
        / standard_error
    )

    p_value = (
        2
        * norm.sf(
            abs(
                z_statistic
            )
        )
    )

    return {
        "rho_difference_recall_minus_primary": float(
            recall_rho
            - primary_rho
        ),
        "fisher_z_difference": float(
            fisher_difference
        ),
        "standard_error": float(
            standard_error
        ),
        "z_statistic": float(
            z_statistic
        ),
        "p_value": float(
            p_value
        ),
    }


def add_family_fdr(
    frame: pd.DataFrame,
    p_column: str,
) -> pd.DataFrame:
    output = frame.copy()

    output[
        "bh_q_value"
    ] = np.nan

    predictor_rows = output[
        output[
            "analysis_layer"
        ]
        == "predictor_function"
    ]

    groups = predictor_rows.groupby(
        [
            "predictor_representation",
            "antigen_target",
            "functional_outcome",
        ],
        observed=True,
        dropna=False,
    ).groups

    for _, indices in groups.items():
        index_list = list(indices)

        p_values = pd.to_numeric(
            output.loc[
                index_list,
                p_column,
            ],
            errors="coerce",
        )

        valid = p_values.notna()

        if not valid.any():
            continue

        adjusted = multipletests(
            p_values.loc[
                valid
            ],
            method="fdr_bh",
        )[1]

        output.loc[
            p_values.loc[
                valid
            ].index,
            "bh_q_value",
        ] = adjusted

    function_rows = (
        output[
            "analysis_layer"
        ]
        == "function_function"
    )

    output.loc[
        function_rows,
        "bh_q_value",
    ] = output.loc[
        function_rows,
        p_column,
    ]

    output[
        "fdr_significant"
    ] = (
        output[
            "bh_q_value"
        ]
        < 0.05
    )

    return output


def construct_context_tests(
    atlas: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "analysis_layer",
        "predictor_representation",
        "antigen_target",
        "predictor_feature",
        "functional_outcome",
        "feature_pair",
        "pair_maximum_floor_severity",
        "floor_sensitive_pair",
    ]

    primary = atlas[
        atlas[
            "analysis_stratum"
        ]
        == "primary_dose0"
    ][
        keys
        + [
            "participants",
            "spearman_rho",
            "spearman_p_value",
            "bh_q_value",
        ]
    ].rename(
        columns={
            "participants": (
                "primary_participants"
            ),
            "spearman_rho": (
                "primary_spearman_rho"
            ),
            "spearman_p_value": (
                "primary_p_value"
            ),
            "bh_q_value": (
                "primary_q_value"
            ),
        }
    )

    recall = atlas[
        atlas[
            "analysis_stratum"
        ]
        == "recall_all_doses"
    ][
        keys
        + [
            "participants",
            "spearman_rho",
            "spearman_p_value",
            "bh_q_value",
        ]
    ].rename(
        columns={
            "participants": (
                "recall_participants"
            ),
            "spearman_rho": (
                "recall_spearman_rho"
            ),
            "spearman_p_value": (
                "recall_p_value"
            ),
            "bh_q_value": (
                "recall_q_value"
            ),
        }
    )

    tests = primary.merge(
        recall,
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    statistics = tests.apply(
        lambda row: fisher_context_difference(
            primary_rho=float(
                row[
                    "primary_spearman_rho"
                ]
            ),
            primary_n=int(
                row[
                    "primary_participants"
                ]
            ),
            recall_rho=float(
                row[
                    "recall_spearman_rho"
                ]
            ),
            recall_n=int(
                row[
                    "recall_participants"
                ]
            ),
        ),
        axis=1,
        result_type="expand",
    )

    tests = pd.concat(
        [
            tests,
            statistics,
        ],
        axis=1,
    )

    tests[
        "context_pattern"
    ] = np.select(
        [
            (
                tests[
                    "primary_spearman_rho"
                ]
                <= 0
            )
            & (
                tests[
                    "recall_spearman_rho"
                ]
                > 0
            ),
            (
                tests[
                    "primary_spearman_rho"
                ]
                > 0
            )
            & (
                tests[
                    "recall_spearman_rho"
                ]
                > tests[
                    "primary_spearman_rho"
                ]
            ),
            (
                tests[
                    "primary_spearman_rho"
                ]
                > 0
            )
            & (
                tests[
                    "recall_spearman_rho"
                ]
                < tests[
                    "primary_spearman_rho"
                ]
            ),
        ],
        [
            "emerges_or_reverses_positive_in_recall",
            "stronger_positive_coupling_in_recall",
            "weaker_positive_coupling_in_recall",
        ],
        default="other_context_pattern",
    )

    return add_family_fdr(
        tests,
        p_column="p_value",
    )


def residualize_on_dose(
    values: np.ndarray,
    doses: np.ndarray,
) -> np.ndarray:
    dose2 = (
        doses
        == 2
    ).astype(float)

    dose3 = (
        doses
        == 3
    ).astype(float)

    design = np.column_stack(
        [
            np.ones(
                len(values)
            ),
            dose2,
            dose3,
        ]
    )

    coefficients = np.linalg.lstsq(
        design,
        values,
        rcond=None,
    )[0]

    fitted = (
        design
        @ coefficients
    )

    return (
        values
        - fitted
    )


def partial_spearman_adjusted_for_dose(
    x: pd.Series,
    y: pd.Series,
    doses: pd.Series,
) -> dict[str, object]:
    x_numeric = pd.to_numeric(
        x,
        errors="coerce",
    )

    y_numeric = pd.to_numeric(
        y,
        errors="coerce",
    )

    dose_numeric = pd.to_numeric(
        doses,
        errors="coerce",
    )

    valid = (
        x_numeric.notna()
        & y_numeric.notna()
        & dose_numeric.notna()
    )

    x_valid = x_numeric.loc[
        valid
    ].to_numpy(
        dtype=float
    )

    y_valid = y_numeric.loc[
        valid
    ].to_numpy(
        dtype=float
    )

    dose_valid = dose_numeric.loc[
        valid
    ].to_numpy(
        dtype=int
    )

    n = len(
        x_valid
    )

    if n < 8:
        return {
            "participants": n,
            "partial_spearman_rho": np.nan,
            "t_statistic": np.nan,
            "degrees_of_freedom": np.nan,
            "p_value": np.nan,
            "partial_status": (
                "insufficient_participants"
            ),
        }

    x_rank = rankdata(
        x_valid,
        method="average",
    )

    y_rank = rankdata(
        y_valid,
        method="average",
    )

    x_residual = residualize_on_dose(
        x_rank,
        dose_valid,
    )

    y_residual = residualize_on_dose(
        y_rank,
        dose_valid,
    )

    if (
        np.std(
            x_residual,
            ddof=1,
        )
        <= 1e-15
        or np.std(
            y_residual,
            ddof=1,
        )
        <= 1e-15
    ):
        return {
            "participants": n,
            "partial_spearman_rho": np.nan,
            "t_statistic": np.nan,
            "degrees_of_freedom": np.nan,
            "p_value": np.nan,
            "partial_status": (
                "constant_residual"
            ),
        }

    result = pearsonr(
        x_residual,
        y_residual,
    )

    rho = float(
        result.statistic
    )

    control_parameters = 2

    degrees_of_freedom = (
        n
        - control_parameters
        - 2
    )

    denominator = max(
        1
        - rho ** 2,
        1e-15,
    )

    t_statistic = (
        rho
        * np.sqrt(
            degrees_of_freedom
            / denominator
        )
    )

    p_value = (
        2
        * student_t.sf(
            abs(
                t_statistic
            ),
            df=degrees_of_freedom,
        )
    )

    return {
        "participants": n,
        "partial_spearman_rho": rho,
        "t_statistic": float(
            t_statistic
        ),
        "degrees_of_freedom": int(
            degrees_of_freedom
        ),
        "p_value": float(
            p_value
        ),
        "partial_status": "estimated",
    }


def construct_partial_tests(
    wide: pd.DataFrame,
    atlas: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    atlas_floor = (
        atlas[
            atlas[
                "analysis_stratum"
            ]
            == "recall_all_doses"
        ][
            [
                "analysis_layer",
                "predictor_representation",
                "antigen_target",
                "predictor_feature",
                "functional_outcome",
                "feature_pair",
                "pair_maximum_floor_severity",
                "floor_sensitive_pair",
                "spearman_rho",
                "bh_q_value",
            ]
        ]
        .rename(
            columns={
                "spearman_rho": (
                    "zero_order_spearman_rho"
                ),
                "bh_q_value": (
                    "zero_order_q_value"
                ),
            }
        )
    )

    for representation in PREDICTOR_REPRESENTATIONS:
        representation_frame = wide[
            wide[
                "predictor_representation"
            ]
            == representation
        ]

        for antigen in ANTIGENS:
            recall = representation_frame[
                (
                    representation_frame[
                        "antigen_target"
                    ]
                    == antigen
                )
                & (
                    pd.to_numeric(
                        representation_frame[
                            "previous_4vHPV_doses"
                        ],
                        errors="coerce",
                    )
                    > 0
                )
            ]

            for outcome in FUNCTIONS:
                for predictor in PREDICTORS:
                    statistics = (
                        partial_spearman_adjusted_for_dose(
                            recall[
                                predictor
                            ],
                            recall[
                                outcome
                            ],
                            recall[
                                "previous_4vHPV_doses"
                            ],
                        )
                    )

                    rows.append(
                        {
                            "analysis_layer": (
                                "predictor_function"
                            ),
                            "predictor_representation": (
                                representation
                            ),
                            "antigen_target": antigen,
                            "predictor_feature": predictor,
                            "functional_outcome": outcome,
                            "feature_pair": (
                                f"{predictor}__{outcome}"
                            ),
                            **statistics,
                        }
                    )

    raw_recall = wide[
        (
            wide[
                "predictor_representation"
            ]
            == "raw_predictor"
        )
        & (
            pd.to_numeric(
                wide[
                    "previous_4vHPV_doses"
                ],
                errors="coerce",
            )
            > 0
        )
    ]

    for antigen in ANTIGENS:
        recall = raw_recall[
            raw_recall[
                "antigen_target"
            ]
            == antigen
        ]

        statistics = (
            partial_spearman_adjusted_for_dose(
                recall["ADCP"],
                recall["nAb"],
                recall[
                    "previous_4vHPV_doses"
                ],
            )
        )

        rows.append(
            {
                "analysis_layer": (
                    "function_function"
                ),
                "predictor_representation": (
                    "raw_functions"
                ),
                "antigen_target": antigen,
                "predictor_feature": "ADCP",
                "functional_outcome": "nAb",
                "feature_pair": "ADCP__nAb",
                **statistics,
            }
        )

    partial = pd.DataFrame(
        rows
    )

    partial = partial.merge(
        atlas_floor,
        on=[
            "analysis_layer",
            "predictor_representation",
            "antigen_target",
            "predictor_feature",
            "functional_outcome",
            "feature_pair",
        ],
        how="left",
        validate="one_to_one",
    )

    partial[
        "rho_change_after_dose_adjustment"
    ] = (
        partial[
            "partial_spearman_rho"
        ]
        - partial[
            "zero_order_spearman_rho"
        ]
    )

    return add_family_fdr(
        partial,
        p_column="p_value",
    )


def correlation_heterogeneity(
    correlations: list[float],
    sample_sizes: list[int],
) -> dict[str, float]:
    valid = [
        (
            float(rho),
            int(n),
        )
        for rho, n in zip(
            correlations,
            sample_sizes,
        )
        if np.isfinite(rho)
        and n > 3
    ]

    if len(valid) < 2:
        return {
            "weighted_fisher_z_mean": np.nan,
            "heterogeneity_q": np.nan,
            "degrees_of_freedom": np.nan,
            "p_value": np.nan,
            "i_squared_percent": np.nan,
        }

    rho_values = np.array(
        [
            clip_correlation(
                rho
            )
            for rho, _ in valid
        ],
        dtype=float,
    )

    n_values = np.array(
        [
            n
            for _, n in valid
        ],
        dtype=float,
    )

    fisher_z = np.arctanh(
        rho_values
    )

    weights = (
        n_values
        - 3
    )

    weighted_mean = np.sum(
        weights
        * fisher_z
    ) / np.sum(
        weights
    )

    q_statistic = np.sum(
        weights
        * (
            fisher_z
            - weighted_mean
        ) ** 2
    )

    degrees_of_freedom = (
        len(valid)
        - 1
    )

    p_value = chi2.sf(
        q_statistic,
        degrees_of_freedom,
    )

    if q_statistic > 0:
        i_squared = max(
            (
                q_statistic
                - degrees_of_freedom
            )
            / q_statistic,
            0.0,
        ) * 100
    else:
        i_squared = 0.0

    return {
        "weighted_fisher_z_mean": float(
            weighted_mean
        ),
        "heterogeneity_q": float(
            q_statistic
        ),
        "degrees_of_freedom": int(
            degrees_of_freedom
        ),
        "p_value": float(
            p_value
        ),
        "i_squared_percent": float(
            i_squared
        ),
    }


def construct_dose_heterogeneity_tests(
    atlas: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "analysis_layer",
        "predictor_representation",
        "antigen_target",
        "predictor_feature",
        "functional_outcome",
        "feature_pair",
        "pair_maximum_floor_severity",
        "floor_sensitive_pair",
    ]

    recall_doses = atlas[
        atlas[
            "analysis_stratum"
        ].isin(
            DOSE_STRATA
        )
    ].copy()

    rows: list[dict[str, object]] = []

    for keys_value, group in recall_doses.groupby(
        keys,
        observed=True,
        dropna=False,
    ):
        metadata = dict(
            zip(
                keys,
                keys_value,
            )
        )

        by_stratum = {
            str(row.analysis_stratum): row
            for row in group.itertuples(
                index=False
            )
        }

        correlations = []
        sample_sizes = []

        output = dict(
            metadata
        )

        for stratum in DOSE_STRATA:
            row = by_stratum.get(
                stratum
            )

            if row is None:
                rho = np.nan
                participants = np.nan
                q_value = np.nan
            else:
                rho = float(
                    row.spearman_rho
                )

                participants = int(
                    row.participants
                )

                q_value = float(
                    row.bh_q_value
                )

            output[
                f"{stratum}_rho"
            ] = rho

            output[
                f"{stratum}_participants"
            ] = participants

            output[
                f"{stratum}_q_value"
            ] = q_value

            correlations.append(
                rho
            )

            sample_sizes.append(
                participants
            )

        statistics = correlation_heterogeneity(
            correlations,
            sample_sizes,
        )

        output.update(
            statistics
        )

        rows.append(
            output
        )

    heterogeneity = pd.DataFrame(
        rows
    )

    return add_family_fdr(
        heterogeneity,
        p_column="p_value",
    )


def build_summary(
    context: pd.DataFrame,
    partial: pd.DataFrame,
    heterogeneity: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for analysis_type, frame in [
        (
            "primary_vs_recall_context_difference",
            context,
        ),
        (
            "recall_partial_adjusted_for_dose",
            partial,
        ),
        (
            "recall_dose_correlation_heterogeneity",
            heterogeneity,
        ),
    ]:
        for representation, subset in frame.groupby(
            "predictor_representation",
            observed=True,
        ):
            rows.append(
                {
                    "analysis_type": analysis_type,
                    "predictor_representation": representation,
                    "tests": len(
                        subset
                    ),
                    "fdr_significant_tests": int(
                        subset[
                            "fdr_significant"
                        ].sum()
                    ),
                    "minimum_q_value": float(
                        pd.to_numeric(
                            subset[
                                "bh_q_value"
                            ],
                            errors="coerce",
                        ).min()
                    ),
                    "floor_sensitive_tests": int(
                        subset[
                            "floor_sensitive_pair"
                        ].sum()
                    ),
                }
            )

    return pd.DataFrame(
        rows
    ).sort_values(
        [
            "analysis_type",
            "predictor_representation",
        ]
    ).reset_index(
        drop=True
    )


def main() -> None:
    for path in [
        ATLAS_INPUT,
        WIDE_INPUT,
        C3A_DECISION_INPUT,
    ]:
        if not path.exists():
            sys.exit(
                f"ERROR: Required input missing: {path}"
            )

    decision = pd.read_csv(
        C3A_DECISION_INPUT,
        sep="\t",
    )

    observed_decision = str(
        decision.loc[
            0,
            "decision",
        ]
    )

    if observed_decision != EXPECTED_C3A_DECISION:
        sys.exit(
            "ERROR: Phase 2C3A decision is "
            f"{observed_decision}; expected "
            f"{EXPECTED_C3A_DECISION}."
        )

    atlas = pd.read_csv(
        ATLAS_INPUT,
        sep="\t",
    )

    wide = pd.read_csv(
        WIDE_INPUT,
        sep="\t",
        dtype={
            "participant_id": "string",
        },
    )

    require_columns(
        atlas,
        {
            "analysis_layer",
            "predictor_representation",
            "antigen_target",
            "analysis_stratum",
            "predictor_feature",
            "functional_outcome",
            "feature_pair",
            "participants",
            "spearman_rho",
            "spearman_p_value",
            "bh_q_value",
            "pair_maximum_floor_severity",
            "floor_sensitive_pair",
        },
        "Functional-coupling atlas",
    )

    require_columns(
        wide,
        {
            "participant_id",
            "previous_4vHPV_doses",
            "antigen_target",
            "predictor_representation",
            *PREDICTORS,
            *FUNCTIONS,
        },
        "Functional-coupling wide table",
    )

    context = construct_context_tests(
        atlas
    )

    partial = construct_partial_tests(
        wide,
        atlas,
    )

    heterogeneity = (
        construct_dose_heterogeneity_tests(
            atlas
        )
    )

    summary = build_summary(
        context,
        partial,
        heterogeneity,
    )

    failures: list[str] = []

    expected_counts = {
        "context": 50,
        "partial": 50,
        "heterogeneity": 50,
        "summary": 9,
    }

    observed_counts = {
        "context": len(
            context
        ),
        "partial": len(
            partial
        ),
        "heterogeneity": len(
            heterogeneity
        ),
        "summary": len(
            summary
        ),
    }

    for label, expected in expected_counts.items():
        observed = observed_counts[
            label
        ]

        if observed != expected:
            failures.append(
                f"{label}: expected {expected}, "
                f"observed {observed}."
            )

    if not (
        context[
            "primary_participants"
        ]
        == 20
    ).all():
        failures.append(
            "Some context comparisons do not contain "
            "20 primary participants."
        )

    if not (
        context[
            "recall_participants"
        ]
        == 60
    ).all():
        failures.append(
            "Some context comparisons do not contain "
            "60 recall participants."
        )

    if not (
        partial[
            "participants"
        ]
        == 60
    ).all():
        failures.append(
            "Some partial correlations do not contain "
            "60 recall participants."
        )

    expected_dose_n = {
        "recall_dose1_participants": 20,
        "recall_dose2_participants": 21,
        "recall_dose3_participants": 19,
    }

    for column, expected in expected_dose_n.items():
        values = pd.to_numeric(
            heterogeneity[
                column
            ],
            errors="coerce",
        )

        if not (
            values
            == expected
        ).all():
            failures.append(
                f"{column}: expected {expected} for every test."
            )

    for label, frame in [
        ("context", context),
        ("partial", partial),
        ("heterogeneity", heterogeneity),
    ]:
        q_values = pd.to_numeric(
            frame[
                "bh_q_value"
            ],
            errors="coerce",
        )

        if q_values.isna().any():
            failures.append(
                f"{label}: missing BH q-values."
            )

        if not q_values.between(
            0,
            1,
        ).all():
            failures.append(
                f"{label}: invalid BH q-values."
            )

    undefined_partial = int(
        (
            partial[
                "partial_status"
            ]
            != "estimated"
        ).sum()
    )

    if undefined_partial:
        failures.append(
            f"{undefined_partial} partial correlations "
            "were not estimable."
        )

    decision_value = (
        "READY_FOR_PHASE2C3C_FUNCTIONAL_COUPLING_SYNTHESIS"
        if not failures
        else "PHASE2C3B_REPAIR_REQUIRED"
    )

    context_output = (
        TABLES
        / "phase2C3B_fiji_primary_vs_recall_correlation_differences.tsv"
    )

    partial_output = (
        TABLES
        / "phase2C3B_fiji_recall_partial_coupling_tests.tsv"
    )

    heterogeneity_output = (
        TABLES
        / "phase2C3B_fiji_recall_dose_correlation_heterogeneity.tsv"
    )

    summary_output = (
        TABLES
        / "phase2C3B_fiji_functional_coupling_inference_summary.tsv"
    )

    decision_output = (
        TABLES
        / "phase2C3B_fiji_functional_coupling_inference_decision.tsv"
    )

    write_tsv(
        context,
        context_output,
    )

    write_tsv(
        partial,
        partial_output,
    )

    write_tsv(
        heterogeneity,
        heterogeneity_output,
    )

    write_tsv(
        summary,
        summary_output,
    )

    decision_frame = pd.DataFrame(
        [
            {
                "decision": decision_value,
                "context_difference_tests": len(
                    context
                ),
                "context_difference_fdr_findings": int(
                    context[
                        "fdr_significant"
                    ].sum()
                ),
                "recall_partial_tests": len(
                    partial
                ),
                "recall_partial_fdr_findings": int(
                    partial[
                        "fdr_significant"
                    ].sum()
                ),
                "dose_heterogeneity_tests": len(
                    heterogeneity
                ),
                "dose_heterogeneity_fdr_findings": int(
                    heterogeneity[
                        "fdr_significant"
                    ].sum()
                ),
                "undefined_partial_correlations": (
                    undefined_partial
                ),
                "summary_rows": len(
                    summary
                ),
                "validation_failures": "; ".join(
                    failures
                ),
            }
        ]
    )

    write_tsv(
        decision_frame,
        decision_output,
    )

    report_path = (
        REPORTS
        / "phase2C3B_fiji_context_partial_coupling_report.md"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as report:
        report.write(
            "# Phase 2C3B Fiji context and partial coupling\n\n"
        )

        report.write(
            "## Decision\n\n"
        )

        report.write(
            f"**{decision_value}**\n\n"
        )

        report.write(
            f"- Primary-versus-recall correlation comparisons: "
            f"{len(context)}\n"
        )

        report.write(
            f"- Recall partial correlations adjusted for categorical "
            f"previous-dose group: {len(partial)}\n"
        )

        report.write(
            f"- Recall-dose correlation-heterogeneity tests: "
            f"{len(heterogeneity)}\n\n"
        )

        report.write(
            "Primary-versus-recall differences are evaluated with an "
            "approximate Fisher-z comparison of independent Spearman "
            "correlations. Recall partial associations are calculated by "
            "rank-transforming each variable and residualizing both ranks "
            "on categorical previous-dose-group indicators before "
            "correlation. Dose-group heterogeneity is evaluated across the "
            "one-, two- and three-dose recall strata using inverse-variance "
            "weighted Fisher-z heterogeneity statistics.\n\n"
        )

        report.write(
            "These analyses distinguish coupling associated with the "
            "primary-versus-memory immunological state from coupling that "
            "could be explained by previous-dose-group composition. "
            "Dose-group comparisons remain between randomized schedule "
            "groups and are not within-person measures of waning.\n"
        )

    print(
        "===== PHASE 2C3B COMPLETE ====="
    )

    print(
        f"Decision: {decision_value}"
    )

    print(
        f"Context-difference tests: {len(context)}"
    )

    print(
        "Context-difference FDR findings: "
        f"{int(context['fdr_significant'].sum())}"
    )

    print(
        f"Recall partial tests: {len(partial)}"
    )

    print(
        "Recall partial FDR findings: "
        f"{int(partial['fdr_significant'].sum())}"
    )

    print(
        "Dose-heterogeneity tests: "
        f"{len(heterogeneity)}"
    )

    print(
        "Dose-heterogeneity FDR findings: "
        f"{int(heterogeneity['fdr_significant'].sum())}"
    )

    print(
        "Undefined partial correlations: "
        f"{undefined_partial}"
    )

    print(
        f"Report: {report_path}"
    )

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
