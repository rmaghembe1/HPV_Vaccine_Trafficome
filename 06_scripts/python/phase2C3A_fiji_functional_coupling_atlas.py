#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
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

PAIRED_INPUT = (
    PROCESSED
    / "phase2A_fiji_paired_effects_analysis_ready.tsv"
)

CALIBRATED_INPUT = (
    PROCESSED
    / "phase2C1_fiji_bpv_calibrated_effects_long.tsv"
)

METADATA_INPUT = (
    PROCESSED
    / "phase2C1_fiji_participant_metadata.tsv"
)

C2C_DECISION_INPUT = (
    TABLES
    / "phase2C2C_fiji_breadth_synthesis_decision.tsv"
)

EXPECTED_C2C_DECISION = (
    "READY_FOR_PHASE2C2_COMMIT_AND_PHASE2C3_FUNCTIONAL_COUPLING"
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

ALL_FEATURES = (
    PREDICTORS
    + FUNCTIONS
)

STRATA = [
    "all_participants",
    "primary_dose0",
    "recall_all_doses",
    "recall_dose1",
    "recall_dose2",
    "recall_dose3",
]

EXPECTED_STRATUM_N = {
    "all_participants": 80,
    "primary_dose0": 20,
    "recall_all_doses": 60,
    "recall_dose1": 20,
    "recall_dose2": 21,
    "recall_dose3": 19,
}

SEVERITY_ORDER = {
    "none": 0,
    "low": 1,
    "moderate": 2,
    "high": 3,
    "unresolved": -1,
}


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


def maximum_severity(
    values: pd.Series,
) -> str:
    valid = [
        str(value)
        for value in values
        if pd.notna(value)
        and str(value) in SEVERITY_ORDER
    ]

    if not valid:
        return "unresolved"

    return max(
        valid,
        key=lambda value: SEVERITY_ORDER[value],
    )


def combine_severity(
    severity_a: str,
    severity_b: str,
) -> str:
    values = [
        str(severity_a),
        str(severity_b),
    ]

    valid = [
        value
        for value in values
        if value in SEVERITY_ORDER
    ]

    if not valid:
        return "unresolved"

    return max(
        valid,
        key=lambda value: SEVERITY_ORDER[value],
    )


def correlation_strength(
    rho: float,
) -> str:
    if not np.isfinite(rho):
        return "undefined"

    absolute = abs(rho)

    if absolute < 0.20:
        return "very_weak"

    if absolute < 0.40:
        return "weak"

    if absolute < 0.60:
        return "moderate"

    if absolute < 0.80:
        return "strong"

    return "very_strong"


def stratum_mask(
    frame: pd.DataFrame,
    stratum: str,
) -> pd.Series:
    dose = pd.to_numeric(
        frame[
            "previous_4vHPV_doses"
        ],
        errors="coerce",
    )

    if stratum == "all_participants":
        return frame[
            "participant_id"
        ].notna()

    if stratum == "primary_dose0":
        return dose == 0

    if stratum == "recall_all_doses":
        return dose > 0

    if stratum == "recall_dose1":
        return dose == 1

    if stratum == "recall_dose2":
        return dose == 2

    if stratum == "recall_dose3":
        return dose == 3

    raise ValueError(
        f"Unknown stratum: {stratum}"
    )


def safe_spearman(
    x: pd.Series,
    y: pd.Series,
) -> dict[str, object]:
    x_numeric = pd.to_numeric(
        x,
        errors="coerce",
    )

    y_numeric = pd.to_numeric(
        y,
        errors="coerce",
    )

    valid = (
        x_numeric.notna()
        & y_numeric.notna()
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

    n = len(x_valid)

    if n < 4:
        return {
            "participants": n,
            "spearman_rho": np.nan,
            "spearman_p_value": np.nan,
            "correlation_status": (
                "insufficient_participants"
            ),
        }

    if (
        np.nanstd(
            x_valid,
            ddof=1,
        )
        <= 1e-15
        or np.nanstd(
            y_valid,
            ddof=1,
        )
        <= 1e-15
    ):
        return {
            "participants": n,
            "spearman_rho": np.nan,
            "spearman_p_value": np.nan,
            "correlation_status": (
                "constant_input"
            ),
        }

    result = spearmanr(
        x_valid,
        y_valid,
        nan_policy="omit",
    )

    rho = float(
        result.statistic
    )

    p_value = float(
        result.pvalue
    )

    if (
        not np.isfinite(rho)
        or not np.isfinite(p_value)
    ):
        status = (
            "undefined_correlation"
        )
    else:
        status = "estimated"

    return {
        "participants": n,
        "spearman_rho": rho,
        "spearman_p_value": p_value,
        "correlation_status": status,
    }


def add_family_fdr(
    atlas: pd.DataFrame,
) -> pd.DataFrame:
    output = atlas.copy()

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
            "analysis_stratum",
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
                "spearman_p_value",
            ],
            errors="coerce",
        )

        valid = p_values.notna()

        if not valid.any():
            continue

        adjusted = multipletests(
            p_values.loc[valid],
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
        "spearman_p_value",
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


def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    for path in [
        PAIRED_INPUT,
        CALIBRATED_INPUT,
        METADATA_INPUT,
        C2C_DECISION_INPUT,
    ]:
        if not path.exists():
            sys.exit(
                f"ERROR: Required input missing: {path}"
            )

    decision = pd.read_csv(
        C2C_DECISION_INPUT,
        sep="\t",
    )

    observed_decision = str(
        decision.loc[
            0,
            "decision",
        ]
    )

    if observed_decision != EXPECTED_C2C_DECISION:
        sys.exit(
            "ERROR: Phase 2C2C decision is "
            f"{observed_decision}; expected "
            f"{EXPECTED_C2C_DECISION}."
        )

    paired = pd.read_csv(
        PAIRED_INPUT,
        sep="\t",
        dtype={
            "participant_id": "string",
        },
    )

    calibrated = pd.read_csv(
        CALIBRATED_INPUT,
        sep="\t",
        dtype={
            "participant_id": "string",
        },
    )

    metadata = pd.read_csv(
        METADATA_INPUT,
        sep="\t",
        dtype={
            "participant_id": "string",
        },
    )

    require_columns(
        paired,
        {
            "participant_id",
            "previous_4vHPV_doses",
            "antigen_target",
            "feature",
            "log2_change_authoritative",
            "paired_floor_severity",
        },
        "Paired-effect table",
    )

    require_columns(
        calibrated,
        {
            "participant_id",
            "previous_4vHPV_doses",
            "antigen_target",
            "feature",
            "bpv_calibrated_log2_change",
        },
        "BPV-calibrated effect table",
    )

    require_columns(
        metadata,
        {
            "participant_id",
            "previous_4vHPV_doses",
            "analysis_context",
        },
        "Participant metadata",
    )

    return (
        paired,
        calibrated,
        metadata,
    )


def build_floor_manifest(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    return (
        paired.groupby(
            [
                "antigen_target",
                "feature",
            ],
            observed=True,
        )
        .agg(
            maximum_floor_severity=(
                "paired_floor_severity",
                maximum_severity,
            )
        )
        .reset_index()
    )


def floor_lookup(
    manifest: pd.DataFrame,
) -> dict[tuple[str, str], str]:
    return {
        (
            str(row.antigen_target),
            str(row.feature),
        ): str(
            row.maximum_floor_severity
        )
        for row in manifest.itertuples(
            index=False
        )
    }


def build_raw_wide(
    paired: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    subset = paired[
        paired[
            "antigen_target"
        ].isin(
            ANTIGENS
        )
        & paired[
            "feature"
        ].isin(
            ALL_FEATURES
        )
    ].copy()

    subset[
        "previous_4vHPV_doses"
    ] = pd.to_numeric(
        subset[
            "previous_4vHPV_doses"
        ],
        errors="raise",
    ).astype(int)

    duplicate_count = int(
        subset.duplicated(
            subset=[
                "participant_id",
                "antigen_target",
                "feature",
            ],
            keep=False,
        ).sum()
    )

    if duplicate_count:
        sys.exit(
            "ERROR: Duplicate raw functional-coupling "
            "records detected."
        )

    wide = subset.pivot(
        index=[
            "participant_id",
            "previous_4vHPV_doses",
            "antigen_target",
        ],
        columns="feature",
        values="log2_change_authoritative",
    ).reset_index()

    wide.columns.name = None

    metadata_small = metadata[
        [
            "participant_id",
            "previous_4vHPV_doses",
            "analysis_context",
        ]
    ].drop_duplicates()

    metadata_small[
        "previous_4vHPV_doses"
    ] = pd.to_numeric(
        metadata_small[
            "previous_4vHPV_doses"
        ],
        errors="raise",
    ).astype(int)

    wide = wide.merge(
        metadata_small,
        on=[
            "participant_id",
            "previous_4vHPV_doses",
        ],
        how="left",
        validate="many_to_one",
    )

    wide[
        "predictor_representation"
    ] = "raw_predictor"

    return wide


def build_calibrated_wide(
    calibrated: pd.DataFrame,
    raw_wide: pd.DataFrame,
) -> pd.DataFrame:
    subset = calibrated[
        calibrated[
            "antigen_target"
        ].isin(
            ANTIGENS
        )
        & calibrated[
            "feature"
        ].isin(
            PREDICTORS
        )
    ].copy()

    subset[
        "previous_4vHPV_doses"
    ] = pd.to_numeric(
        subset[
            "previous_4vHPV_doses"
        ],
        errors="raise",
    ).astype(int)

    duplicate_count = int(
        subset.duplicated(
            subset=[
                "participant_id",
                "antigen_target",
                "feature",
            ],
            keep=False,
        ).sum()
    )

    if duplicate_count:
        sys.exit(
            "ERROR: Duplicate calibrated predictor "
            "records detected."
        )

    predictors = subset.pivot(
        index=[
            "participant_id",
            "previous_4vHPV_doses",
            "antigen_target",
        ],
        columns="feature",
        values="bpv_calibrated_log2_change",
    ).reset_index()

    predictors.columns.name = None

    functions = raw_wide[
        [
            "participant_id",
            "previous_4vHPV_doses",
            "antigen_target",
            "analysis_context",
            "ADCP",
            "nAb",
        ]
    ]

    wide = predictors.merge(
        functions,
        on=[
            "participant_id",
            "previous_4vHPV_doses",
            "antigen_target",
        ],
        how="inner",
        validate="one_to_one",
    )

    wide[
        "predictor_representation"
    ] = (
        "bpv_calibrated_predictor"
    )

    return wide


def construct_coupling_atlas(
    raw_wide: pd.DataFrame,
    calibrated_wide: pd.DataFrame,
    severity: dict[tuple[str, str], str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    representation_frames = {
        "raw_predictor": raw_wide,
        "bpv_calibrated_predictor": (
            calibrated_wide
        ),
    }

    bpv_severity = {
        predictor: severity.get(
            (
                "BPV",
                predictor,
            ),
            "unresolved",
        )
        for predictor in PREDICTORS
    }

    for representation, frame in representation_frames.items():
        for antigen in ANTIGENS:
            antigen_frame = frame[
                frame[
                    "antigen_target"
                ]
                == antigen
            ]

            for stratum in STRATA:
                subset = antigen_frame.loc[
                    stratum_mask(
                        antigen_frame,
                        stratum,
                    )
                ]

                for outcome in FUNCTIONS:
                    for predictor in PREDICTORS:
                        statistics = safe_spearman(
                            subset[
                                predictor
                            ],
                            subset[
                                outcome
                            ],
                        )

                        hpv_predictor_floor = severity.get(
                            (
                                antigen,
                                predictor,
                            ),
                            "unresolved",
                        )

                        if representation == "bpv_calibrated_predictor":
                            predictor_floor = combine_severity(
                                hpv_predictor_floor,
                                bpv_severity[
                                    predictor
                                ],
                            )
                        else:
                            predictor_floor = hpv_predictor_floor

                        outcome_floor = severity.get(
                            (
                                antigen,
                                outcome,
                            ),
                            "unresolved",
                        )

                        pair_floor = combine_severity(
                            predictor_floor,
                            outcome_floor,
                        )

                        rho = statistics[
                            "spearman_rho"
                        ]

                        rows.append(
                            {
                                "analysis_layer": (
                                    "predictor_function"
                                ),
                                "predictor_representation": (
                                    representation
                                ),
                                "antigen_target": antigen,
                                "analysis_stratum": stratum,
                                "predictor_feature": predictor,
                                "functional_outcome": outcome,
                                "feature_pair": (
                                    f"{predictor}__{outcome}"
                                ),
                                **statistics,
                                "correlation_direction": (
                                    "positive"
                                    if np.isfinite(rho)
                                    and rho > 0
                                    else (
                                        "negative"
                                        if np.isfinite(rho)
                                        and rho < 0
                                        else "undefined"
                                    )
                                ),
                                "correlation_strength": (
                                    correlation_strength(
                                        rho
                                    )
                                ),
                                "predictor_floor_severity": (
                                    predictor_floor
                                ),
                                "outcome_floor_severity": (
                                    outcome_floor
                                ),
                                "pair_maximum_floor_severity": (
                                    pair_floor
                                ),
                                "floor_sensitive_pair": (
                                    pair_floor
                                    in {
                                        "moderate",
                                        "high",
                                    }
                                ),
                            }
                        )

    for antigen in ANTIGENS:
        antigen_frame = raw_wide[
            raw_wide[
                "antigen_target"
            ]
            == antigen
        ]

        for stratum in STRATA:
            subset = antigen_frame.loc[
                stratum_mask(
                    antigen_frame,
                    stratum,
                )
            ]

            statistics = safe_spearman(
                subset["ADCP"],
                subset["nAb"],
            )

            adcp_floor = severity.get(
                (
                    antigen,
                    "ADCP",
                ),
                "unresolved",
            )

            nab_floor = severity.get(
                (
                    antigen,
                    "nAb",
                ),
                "unresolved",
            )

            pair_floor = combine_severity(
                adcp_floor,
                nab_floor,
            )

            rho = statistics[
                "spearman_rho"
            ]

            rows.append(
                {
                    "analysis_layer": (
                        "function_function"
                    ),
                    "predictor_representation": (
                        "raw_functions"
                    ),
                    "antigen_target": antigen,
                    "analysis_stratum": stratum,
                    "predictor_feature": "ADCP",
                    "functional_outcome": "nAb",
                    "feature_pair": "ADCP__nAb",
                    **statistics,
                    "correlation_direction": (
                        "positive"
                        if np.isfinite(rho)
                        and rho > 0
                        else (
                            "negative"
                            if np.isfinite(rho)
                            and rho < 0
                            else "undefined"
                        )
                    ),
                    "correlation_strength": (
                        correlation_strength(
                            rho
                        )
                    ),
                    "predictor_floor_severity": (
                        adcp_floor
                    ),
                    "outcome_floor_severity": (
                        nab_floor
                    ),
                    "pair_maximum_floor_severity": (
                        pair_floor
                    ),
                    "floor_sensitive_pair": (
                        pair_floor
                        in {
                            "moderate",
                            "high",
                        }
                    ),
                }
            )

    atlas = pd.DataFrame(
        rows
    )

    return add_family_fdr(
        atlas
    )


def build_calibration_registry(
    atlas: pd.DataFrame,
) -> pd.DataFrame:
    predictor_atlas = atlas[
        atlas[
            "analysis_layer"
        ]
        == "predictor_function"
    ].copy()

    keys = [
        "antigen_target",
        "analysis_stratum",
        "predictor_feature",
        "functional_outcome",
        "feature_pair",
    ]

    raw = predictor_atlas[
        predictor_atlas[
            "predictor_representation"
        ]
        == "raw_predictor"
    ][
        keys
        + [
            "participants",
            "spearman_rho",
            "spearman_p_value",
            "bh_q_value",
            "pair_maximum_floor_severity",
        ]
    ].rename(
        columns={
            "participants": (
                "raw_participants"
            ),
            "spearman_rho": (
                "raw_spearman_rho"
            ),
            "spearman_p_value": (
                "raw_p_value"
            ),
            "bh_q_value": (
                "raw_q_value"
            ),
            "pair_maximum_floor_severity": (
                "raw_pair_floor_severity"
            ),
        }
    )

    calibrated = predictor_atlas[
        predictor_atlas[
            "predictor_representation"
        ]
        == "bpv_calibrated_predictor"
    ][
        keys
        + [
            "participants",
            "spearman_rho",
            "spearman_p_value",
            "bh_q_value",
            "pair_maximum_floor_severity",
        ]
    ].rename(
        columns={
            "participants": (
                "bpv_calibrated_participants"
            ),
            "spearman_rho": (
                "bpv_calibrated_spearman_rho"
            ),
            "spearman_p_value": (
                "bpv_calibrated_p_value"
            ),
            "bh_q_value": (
                "bpv_calibrated_q_value"
            ),
            "pair_maximum_floor_severity": (
                "bpv_calibrated_pair_floor_severity"
            ),
        }
    )

    registry = raw.merge(
        calibrated,
        on=keys,
        how="outer",
        validate="one_to_one",
    )

    statuses: list[str] = []

    for row in registry.itertuples(
        index=False
    ):
        raw_rho = float(
            row.raw_spearman_rho
        )

        calibrated_rho = float(
            row.bpv_calibrated_spearman_rho
        )

        raw_q = float(
            row.raw_q_value
        )

        calibrated_q = float(
            row.bpv_calibrated_q_value
        )

        if (
            np.isfinite(raw_rho)
            and np.isfinite(
                calibrated_rho
            )
            and raw_rho
            * calibrated_rho
            < 0
        ):
            status = (
                "direction_changed_after_bpv_calibration"
            )
        elif (
            raw_q < 0.05
            and calibrated_q < 0.05
        ):
            status = (
                "supported_raw_and_bpv_calibrated"
            )
        elif (
            raw_q < 0.05
            and calibrated_q >= 0.05
        ):
            status = (
                "attenuated_after_bpv_calibration"
            )
        elif (
            raw_q >= 0.05
            and calibrated_q < 0.05
        ):
            status = (
                "emerges_after_bpv_calibration"
            )
        else:
            status = (
                "not_fdr_significant"
            )

        statuses.append(status)

    registry[
        "bpv_calibration_status"
    ] = statuses

    registry[
        "absolute_rho_difference"
    ] = (
        registry[
            "bpv_calibrated_spearman_rho"
        ]
        - registry[
            "raw_spearman_rho"
        ]
    ).abs()

    return registry


def build_summary(
    atlas: pd.DataFrame,
) -> pd.DataFrame:
    return (
        atlas.groupby(
            [
                "analysis_layer",
                "predictor_representation",
                "antigen_target",
                "analysis_stratum",
                "functional_outcome",
            ],
            observed=True,
            dropna=False,
        )
        .agg(
            tested_pairs=(
                "feature_pair",
                "size",
            ),
            estimated_pairs=(
                "correlation_status",
                lambda values: int(
                    (
                        pd.Series(values)
                        == "estimated"
                    ).sum()
                ),
            ),
            positive_correlations=(
                "spearman_rho",
                lambda values: int(
                    (
                        pd.to_numeric(
                            values,
                            errors="coerce",
                        )
                        > 0
                    ).sum()
                ),
            ),
            negative_correlations=(
                "spearman_rho",
                lambda values: int(
                    (
                        pd.to_numeric(
                            values,
                            errors="coerce",
                        )
                        < 0
                    ).sum()
                ),
            ),
            nominal_p_below_0_05=(
                "spearman_p_value",
                lambda values: int(
                    (
                        pd.to_numeric(
                            values,
                            errors="coerce",
                        )
                        < 0.05
                    ).sum()
                ),
            ),
            fdr_significant_pairs=(
                "fdr_significant",
                "sum",
            ),
            median_absolute_rho=(
                "spearman_rho",
                lambda values: float(
                    pd.to_numeric(
                        values,
                        errors="coerce",
                    )
                    .abs()
                    .median()
                ),
            ),
            maximum_absolute_rho=(
                "spearman_rho",
                lambda values: float(
                    pd.to_numeric(
                        values,
                        errors="coerce",
                    )
                    .abs()
                    .max()
                ),
            ),
            floor_sensitive_pairs=(
                "floor_sensitive_pair",
                "sum",
            ),
        )
        .reset_index()
    )


def main() -> None:
    (
        paired,
        calibrated,
        metadata,
    ) = load_inputs()

    paired[
        "previous_4vHPV_doses"
    ] = pd.to_numeric(
        paired[
            "previous_4vHPV_doses"
        ],
        errors="raise",
    ).astype(int)

    floor_manifest = build_floor_manifest(
        paired
    )

    severity = floor_lookup(
        floor_manifest
    )

    raw_wide = build_raw_wide(
        paired,
        metadata,
    )

    calibrated_wide = build_calibrated_wide(
        calibrated,
        raw_wide,
    )

    combined_wide = pd.concat(
        [
            raw_wide,
            calibrated_wide,
        ],
        ignore_index=True,
        sort=False,
    )

    atlas = construct_coupling_atlas(
        raw_wide,
        calibrated_wide,
        severity,
    )

    calibration_registry = (
        build_calibration_registry(
            atlas
        )
    )

    summary = build_summary(
        atlas
    )

    failures: list[str] = []

    requested_columns = ALL_FEATURES

    if len(raw_wide) != 160:
        failures.append(
            "Raw wide table: expected 160 rows, "
            f"observed {len(raw_wide)}."
        )

    if len(calibrated_wide) != 160:
        failures.append(
            "Calibrated wide table: expected 160 rows, "
            f"observed {len(calibrated_wide)}."
        )

    if len(combined_wide) != 320:
        failures.append(
            "Combined wide table: expected 320 rows, "
            f"observed {len(combined_wide)}."
        )

    raw_missing = int(
        raw_wide[
            requested_columns
        ].isna().sum().sum()
    )

    calibrated_missing = int(
        calibrated_wide[
            requested_columns
        ].isna().sum().sum()
    )

    if raw_missing:
        failures.append(
            f"Raw wide table contains {raw_missing} "
            "missing requested values."
        )

    if calibrated_missing:
        failures.append(
            f"Calibrated wide table contains {calibrated_missing} "
            "missing requested values."
        )

    predictor_rows = atlas[
        atlas[
            "analysis_layer"
        ]
        == "predictor_function"
    ]

    function_rows = atlas[
        atlas[
            "analysis_layer"
        ]
        == "function_function"
    ]

    if len(predictor_rows) != 288:
        failures.append(
            "Expected 288 predictor-function correlations, "
            f"observed {len(predictor_rows)}."
        )

    if len(function_rows) != 12:
        failures.append(
            "Expected 12 ADCP-neutralization correlations, "
            f"observed {len(function_rows)}."
        )

    if len(atlas) != 300:
        failures.append(
            "Expected 300 total coupling rows, "
            f"observed {len(atlas)}."
        )

    if len(calibration_registry) != 144:
        failures.append(
            "Expected 144 raw-versus-calibrated rows, "
            f"observed {len(calibration_registry)}."
        )

    if len(summary) != 60:
        failures.append(
            "Expected 60 summary rows, "
            f"observed {len(summary)}."
        )

    for stratum, expected_n in EXPECTED_STRATUM_N.items():
        observed_n = set(
            atlas.loc[
                atlas[
                    "analysis_stratum"
                ]
                == stratum,
                "participants",
            ]
            .dropna()
            .astype(int)
            .unique()
        )

        if observed_n != {
            expected_n
        }:
            failures.append(
                f"{stratum}: expected correlation n={expected_n}, "
                f"observed {sorted(observed_n)}."
            )

    estimated = atlas[
        atlas[
            "correlation_status"
        ]
        == "estimated"
    ]

    valid_q = pd.to_numeric(
        estimated[
            "bh_q_value"
        ],
        errors="coerce",
    )

    if valid_q.isna().any():
        failures.append(
            "Some estimated correlations lack BH q-values."
        )

    if not valid_q.between(
        0,
        1,
    ).all():
        failures.append(
            "Some estimated correlations have invalid BH q-values."
        )

    undefined_correlations = int(
        (
            atlas[
                "correlation_status"
            ]
            != "estimated"
        ).sum()
    )

    decision_value = (
        "READY_FOR_PHASE2C3B_CONTEXT_AND_PARTIAL_COUPLING_ANALYSIS"
        if not failures
        else "PHASE2C3A_REPAIR_REQUIRED"
    )

    wide_output = (
        PROCESSED
        / "phase2C3A_fiji_functional_coupling_wide.tsv"
    )

    atlas_output = (
        TABLES
        / "phase2C3A_fiji_functional_coupling_atlas.tsv"
    )

    calibration_output = (
        TABLES
        / "phase2C3A_fiji_raw_vs_bpv_coupling_registry.tsv"
    )

    summary_output = (
        TABLES
        / "phase2C3A_fiji_functional_coupling_summary.tsv"
    )

    decision_output = (
        TABLES
        / "phase2C3A_fiji_functional_coupling_decision.tsv"
    )

    write_tsv(
        combined_wide,
        wide_output,
    )

    write_tsv(
        atlas,
        atlas_output,
    )

    write_tsv(
        calibration_registry,
        calibration_output,
    )

    write_tsv(
        summary,
        summary_output,
    )

    status_counts = (
        calibration_registry[
            "bpv_calibration_status"
        ]
        .value_counts()
        .to_dict()
    )

    decision = pd.DataFrame(
        [
            {
                "decision": decision_value,
                "raw_wide_rows": len(
                    raw_wide
                ),
                "bpv_calibrated_wide_rows": len(
                    calibrated_wide
                ),
                "combined_wide_rows": len(
                    combined_wide
                ),
                "predictor_function_correlations": len(
                    predictor_rows
                ),
                "function_function_correlations": len(
                    function_rows
                ),
                "total_coupling_rows": len(
                    atlas
                ),
                "raw_vs_bpv_registry_rows": len(
                    calibration_registry
                ),
                "summary_rows": len(
                    summary
                ),
                "undefined_correlations": (
                    undefined_correlations
                ),
                "supported_raw_and_bpv_calibrated": int(
                    status_counts.get(
                        "supported_raw_and_bpv_calibrated",
                        0,
                    )
                ),
                "attenuated_after_bpv_calibration": int(
                    status_counts.get(
                        "attenuated_after_bpv_calibration",
                        0,
                    )
                ),
                "emerges_after_bpv_calibration": int(
                    status_counts.get(
                        "emerges_after_bpv_calibration",
                        0,
                    )
                ),
                "direction_changed_after_bpv_calibration": int(
                    status_counts.get(
                        "direction_changed_after_bpv_calibration",
                        0,
                    )
                ),
                "validation_failures": "; ".join(
                    failures
                ),
            }
        ]
    )

    write_tsv(
        decision,
        decision_output,
    )

    report_path = (
        REPORTS
        / "phase2C3A_fiji_functional_coupling_atlas_report.md"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as report:
        report.write(
            "# Phase 2C3A Fiji functional-coupling atlas\n\n"
        )

        report.write("## Decision\n\n")
        report.write(f"**{decision_value}**\n\n")

        report.write(
            "- Antigens: HPV16 and HPV18\n"
        )
        report.write(
            "- Predictors: IgG, IgG1, IgG3, FcγR2A, "
            "FcγR2B and FcγR3A\n"
        )
        report.write(
            "- Functional outcomes: ADCP and neutralization\n"
        )
        report.write(
            "- Predictor representations: raw HPV change and "
            "matched BPV-calibrated HPV change\n"
        )
        report.write(
            f"- Predictor-function correlations: "
            f"{len(predictor_rows)}\n"
        )
        report.write(
            f"- ADCP-neutralization correlations: "
            f"{len(function_rows)}\n"
        )
        report.write(
            f"- Undefined or constant-input correlations: "
            f"{undefined_correlations}\n\n"
        )

        report.write(
            "Spearman correlations provide the zero-order functional "
            "coupling atlas. Benjamini-Hochberg correction is applied "
            "within antigen, immunization stratum, functional outcome "
            "and predictor representation across the six prespecified "
            "antibody or FcγR predictors. Matched BPV-calibrated "
            "predictors are analyzed separately because ADCP and "
            "neutralization do not have matched BPV assays.\n\n"
        )

        report.write(
            "Phase 2C3B should compare primary-induction and recall "
            "correlations, evaluate recall-dose heterogeneity and fit "
            "partial associations that account for previous-dose group "
            "within the recall population. Moderate- and high-floor "
            "pairs remain explicitly flagged in the atlas.\n"
        )

    print("===== PHASE 2C3A COMPLETE =====")
    print(f"Decision: {decision_value}")
    print(
        f"Raw wide rows: {len(raw_wide)}"
    )
    print(
        "BPV-calibrated wide rows: "
        f"{len(calibrated_wide)}"
    )
    print(
        "Predictor-function correlations: "
        f"{len(predictor_rows)}"
    )
    print(
        "Function-function correlations: "
        f"{len(function_rows)}"
    )
    print(
        f"Total coupling rows: {len(atlas)}"
    )
    print(
        "Raw-vs-BPV registry rows: "
        f"{len(calibration_registry)}"
    )
    print(
        f"Summary rows: {len(summary)}"
    )
    print(
        "Undefined correlations: "
        f"{undefined_correlations}"
    )
    print(f"Report: {report_path}")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
