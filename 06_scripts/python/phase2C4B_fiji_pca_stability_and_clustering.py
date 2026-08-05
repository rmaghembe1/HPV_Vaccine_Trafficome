#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    pairwise_distances_argmin_min,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


ROOT = Path(
    "/mnt/d/HPV_Vaccine_Trafficome_Project"
)

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

RAW_INPUT = (
    PROCESSED
    / "phase2C1_fiji_raw_log2_change_matrix.tsv"
)

BPV_INPUT = (
    PROCESSED
    / "phase2C1_fiji_bpv_calibrated_log2_change_matrix.tsv"
)

METADATA_INPUT = (
    PROCESSED
    / "phase2C1_fiji_participant_metadata.tsv"
)

C4A_DECISION_INPUT = (
    TABLES
    / "phase2C4A_fiji_pca_architecture_decision.tsv"
)

EXPECTED_C4A_DECISION = (
    "READY_FOR_PHASE2C4B_PCA_STABILITY_AND_CLUSTERING"
)

REPRESENTATIONS = {
    "raw_log2_change": RAW_INPUT,
    "bpv_calibrated_log2_change": BPV_INPUT,
}

EXPECTED_FAMILIES = {
    "raw_log2_change": {
        "binding_antibody_abundance",
        "igg_subclass_architecture",
        "fc_receptor_communication",
        "phagocytic_function",
        "neutralization",
    },
    "bpv_calibrated_log2_change": {
        "binding_antibody_abundance",
        "igg_subclass_architecture",
        "fc_receptor_communication",
    },
}

ANTIGENS = {
    "HPV16",
    "HPV18",
    "HPV31",
    "HPV33",
    "HPV45",
    "HPV52",
    "HPV58",
    "BPV",
}

TOP_STABILITY_PCS = 5

BOOTSTRAP_REPLICATES = 300

CLUSTER_K_VALUES = [
    2,
    3,
    4,
    5,
    6,
]

CLUSTER_SUBSAMPLE_REPLICATES = 100

CLUSTER_SUBSAMPLE_FRACTION = 0.80

RANDOM_SEED = 20260805


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


def require_file(
    path: Path,
) -> None:
    if not path.exists():
        sys.exit(
            f"ERROR: Required input is missing: {path}"
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
            f"ERROR: {label} is missing required columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )


def parse_feature(
    feature_column: str,
) -> tuple[str, str]:
    text = str(
        feature_column
    )

    if "__" not in text:
        return (
            "unresolved",
            text,
        )

    antigen, assay_feature = text.split(
        "__",
        maxsplit=1,
    )

    if antigen not in ANTIGENS:
        antigen = "unresolved"

    return (
        antigen,
        assay_feature,
    )


def classify_feature_family(
    assay_feature: str,
) -> str:
    feature = str(
        assay_feature
    )

    if feature in {
        "IgG",
        "IgM",
        "IgA1",
        "IgA2",
    }:
        return (
            "binding_antibody_abundance"
        )

    if feature in {
        "IgG1",
        "IgG2",
        "IgG3",
        "IgG4",
    }:
        return (
            "igg_subclass_architecture"
        )

    if feature in {
        "FcgR2A",
        "FcgR2B",
        "FcgR3A",
    }:
        return (
            "fc_receptor_communication"
        )

    if feature == "ADCP":
        return "phagocytic_function"

    if feature == "nAb":
        return "neutralization"

    return "unresolved"


def annotate_features(
    feature_columns: list[str],
) -> pd.DataFrame:
    rows = []

    for feature_column in feature_columns:
        antigen, assay_feature = parse_feature(
            feature_column
        )

        rows.append(
            {
                "feature_column": feature_column,
                "antigen_target": antigen,
                "assay_feature": assay_feature,
                "feature_family": (
                    classify_feature_family(
                        assay_feature
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def fit_reference_pca(
    numeric: pd.DataFrame,
) -> dict[str, object]:
    scaler = StandardScaler(
        with_mean=True,
        with_std=True,
    )

    standardized = scaler.fit_transform(
        numeric.to_numpy(
            dtype=float
        )
    )

    n_components = min(
        standardized.shape[0] - 1,
        standardized.shape[1],
    )

    pca = PCA(
        n_components=n_components,
        svd_solver="full",
    )

    scores = pca.fit_transform(
        standardized
    )

    cumulative_variance = np.cumsum(
        pca.explained_variance_ratio_
    )

    components_for_80_percent = int(
        np.flatnonzero(
            cumulative_variance >= 0.80
        )[0]
        + 1
    )

    return {
        "scaler": scaler,
        "standardized": standardized,
        "pca": pca,
        "scores": scores,
        "components_for_80_percent": (
            components_for_80_percent
        ),
    }


def safe_correlation(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    x_array = np.asarray(
        x,
        dtype=float,
    )

    y_array = np.asarray(
        y,
        dtype=float,
    )

    if (
        len(x_array) != len(y_array)
        or len(x_array) < 3
    ):
        return np.nan

    if (
        np.std(
            x_array,
            ddof=1,
        )
        <= 1e-15
        or np.std(
            y_array,
            ddof=1,
        )
        <= 1e-15
    ):
        return np.nan

    result = pearsonr(
        x_array,
        y_array,
    )

    return float(
        result.statistic
    )


def match_components(
    reference_loadings: np.ndarray,
    candidate_loadings: np.ndarray,
) -> list[dict[str, object]]:
    reference = np.asarray(
        reference_loadings,
        dtype=float,
    )

    candidate = np.asarray(
        candidate_loadings,
        dtype=float,
    )

    if reference.shape[0] != candidate.shape[0]:
        raise ValueError(
            "Reference and candidate component counts differ."
        )

    component_count = reference.shape[0]

    correlation_matrix = np.zeros(
        (
            component_count,
            component_count,
        ),
        dtype=float,
    )

    for reference_index in range(
        component_count
    ):
        for candidate_index in range(
            component_count
        ):
            correlation = safe_correlation(
                reference[
                    reference_index,
                    :
                ],
                candidate[
                    candidate_index,
                    :
                ],
            )

            if not np.isfinite(
                correlation
            ):
                correlation = 0.0

            correlation_matrix[
                reference_index,
                candidate_index,
            ] = correlation

    reference_indices, candidate_indices = (
        linear_sum_assignment(
            -np.abs(
                correlation_matrix
            )
        )
    )

    matches = []

    for reference_index, candidate_index in zip(
        reference_indices,
        candidate_indices,
    ):
        signed_correlation = float(
            correlation_matrix[
                reference_index,
                candidate_index,
            ]
        )

        sign_multiplier = (
            1.0
            if signed_correlation >= 0
            else -1.0
        )

        matches.append(
            {
                "reference_component_index": int(
                    reference_index
                ),
                "candidate_component_index": int(
                    candidate_index
                ),
                "signed_loading_correlation": (
                    signed_correlation
                ),
                "absolute_loading_correlation": abs(
                    signed_correlation
                ),
                "sign_multiplier": (
                    sign_multiplier
                ),
            }
        )

    return sorted(
        matches,
        key=lambda row: row[
            "reference_component_index"
        ],
    )


def leave_one_family_out_stability(
    representation: str,
    numeric: pd.DataFrame,
    annotation: pd.DataFrame,
    reference: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_pca = reference[
        "pca"
    ]

    reference_scores = np.asarray(
        reference[
            "scores"
        ],
        dtype=float,
    )

    reference_feature_columns = list(
        numeric.columns
    )

    reference_column_index = {
        column: index
        for index, column in enumerate(
            reference_feature_columns
        )
    }

    observed_families = set(
        annotation[
            "feature_family"
        ].astype(str)
    )

    expected_families = EXPECTED_FAMILIES[
        representation
    ]

    if observed_families != expected_families:
        raise RuntimeError(
            f"{representation}: observed feature families "
            f"{sorted(observed_families)} do not match expected "
            f"{sorted(expected_families)}."
        )

    stability_rows = []

    manifest_rows = []

    for omitted_family in sorted(
        observed_families
    ):
        retained_annotation = annotation[
            annotation[
                "feature_family"
            ]
            != omitted_family
        ].copy()

        retained_columns = retained_annotation[
            "feature_column"
        ].astype(str).tolist()

        reduced_numeric = numeric[
            retained_columns
        ]

        reduced_reference = fit_reference_pca(
            reduced_numeric
        )

        candidate_pca = reduced_reference[
            "pca"
        ]

        candidate_scores = np.asarray(
            reduced_reference[
                "scores"
            ],
            dtype=float,
        )

        retained_indices = [
            reference_column_index[
                column
            ]
            for column in retained_columns
        ]

        restricted_reference_loadings = (
            reference_pca.components_[
                :TOP_STABILITY_PCS,
                :
            ][
                :,
                retained_indices,
            ]
        )

        candidate_loadings = (
            candidate_pca.components_[
                :TOP_STABILITY_PCS,
                :
            ]
        )

        matches = match_components(
            restricted_reference_loadings,
            candidate_loadings,
        )

        omitted_features = int(
            (
                annotation[
                    "feature_family"
                ]
                == omitted_family
            ).sum()
        )

        manifest_rows.append(
            {
                "matrix_representation": (
                    representation
                ),
                "omitted_feature_family": (
                    omitted_family
                ),
                "original_features": len(
                    reference_feature_columns
                ),
                "omitted_features": (
                    omitted_features
                ),
                "retained_features": len(
                    retained_columns
                ),
                "reduced_components_for_80_percent_variance": (
                    reduced_reference[
                        "components_for_80_percent"
                    ]
                ),
            }
        )

        for match in matches:
            reference_index = int(
                match[
                    "reference_component_index"
                ]
            )

            candidate_index = int(
                match[
                    "candidate_component_index"
                ]
            )

            sign_multiplier = float(
                match[
                    "sign_multiplier"
                ]
            )

            score_correlation = safe_correlation(
                reference_scores[
                    :,
                    reference_index,
                ],
                sign_multiplier
                * candidate_scores[
                    :,
                    candidate_index,
                ],
            )

            reference_variance = float(
                reference_pca.explained_variance_ratio_[
                    reference_index
                ]
            )

            candidate_variance = float(
                candidate_pca.explained_variance_ratio_[
                    candidate_index
                ]
            )

            stability_rows.append(
                {
                    "matrix_representation": (
                        representation
                    ),
                    "omitted_feature_family": (
                        omitted_family
                    ),
                    "reference_principal_component": (
                        f"PC{reference_index + 1}"
                    ),
                    "reference_component_number": (
                        reference_index
                        + 1
                    ),
                    "matched_reduced_principal_component": (
                        f"PC{candidate_index + 1}"
                    ),
                    "matched_reduced_component_number": (
                        candidate_index
                        + 1
                    ),
                    "retained_features": len(
                        retained_columns
                    ),
                    "absolute_loading_correlation": (
                        match[
                            "absolute_loading_correlation"
                        ]
                    ),
                    "sign_aligned_score_correlation": (
                        score_correlation
                    ),
                    "reference_explained_variance_ratio": (
                        reference_variance
                    ),
                    "reduced_explained_variance_ratio": (
                        candidate_variance
                    ),
                    "absolute_variance_ratio_difference": abs(
                        candidate_variance
                        - reference_variance
                    ),
                }
            )

    return (
        pd.DataFrame(
            stability_rows
        ),
        pd.DataFrame(
            manifest_rows
        ),
    )


def bootstrap_pca_stability(
    representation: str,
    numeric: pd.DataFrame,
    reference: dict[str, object],
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(
        random_seed
    )

    reference_pca = reference[
        "pca"
    ]

    reference_loadings = (
        reference_pca.components_[
            :TOP_STABILITY_PCS,
            :
        ]
    )

    reference_variance = (
        reference_pca.explained_variance_ratio_[
            :TOP_STABILITY_PCS
        ]
    )

    numeric_array = numeric.to_numpy(
        dtype=float
    )

    participant_count = numeric_array.shape[0]

    replicate_rows = []

    for bootstrap_index in range(
        1,
        BOOTSTRAP_REPLICATES + 1,
    ):
        sampled_indices = rng.integers(
            low=0,
            high=participant_count,
            size=participant_count,
        )

        bootstrap_matrix = numeric_array[
            sampled_indices,
            :
        ]

        scaler = StandardScaler(
            with_mean=True,
            with_std=True,
        )

        standardized = scaler.fit_transform(
            bootstrap_matrix
        )

        pca = PCA(
            n_components=TOP_STABILITY_PCS,
            svd_solver="full",
        )

        pca.fit(
            standardized
        )

        matches = match_components(
            reference_loadings,
            pca.components_,
        )

        unique_participants = int(
            len(
                np.unique(
                    sampled_indices
                )
            )
        )

        for match in matches:
            reference_index = int(
                match[
                    "reference_component_index"
                ]
            )

            candidate_index = int(
                match[
                    "candidate_component_index"
                ]
            )

            candidate_variance = float(
                pca.explained_variance_ratio_[
                    candidate_index
                ]
            )

            replicate_rows.append(
                {
                    "matrix_representation": (
                        representation
                    ),
                    "bootstrap_replicate": (
                        bootstrap_index
                    ),
                    "sampled_rows": (
                        participant_count
                    ),
                    "unique_sampled_participants": (
                        unique_participants
                    ),
                    "reference_principal_component": (
                        f"PC{reference_index + 1}"
                    ),
                    "reference_component_number": (
                        reference_index
                        + 1
                    ),
                    "matched_bootstrap_principal_component": (
                        f"PC{candidate_index + 1}"
                    ),
                    "matched_bootstrap_component_number": (
                        candidate_index
                        + 1
                    ),
                    "absolute_loading_correlation": (
                        match[
                            "absolute_loading_correlation"
                        ]
                    ),
                    "reference_explained_variance_ratio": float(
                        reference_variance[
                            reference_index
                        ]
                    ),
                    "bootstrap_explained_variance_ratio": (
                        candidate_variance
                    ),
                    "absolute_variance_ratio_difference": abs(
                        candidate_variance
                        - float(
                            reference_variance[
                                reference_index
                            ]
                        )
                    ),
                }
            )

    replicates = pd.DataFrame(
        replicate_rows
    )

    summary_rows = []

    for (
        matrix_representation,
        component_number,
    ), group in replicates.groupby(
        [
            "matrix_representation",
            "reference_component_number",
        ],
        observed=True,
    ):
        loading_correlations = pd.to_numeric(
            group[
                "absolute_loading_correlation"
            ],
            errors="coerce",
        )

        variance_differences = pd.to_numeric(
            group[
                "absolute_variance_ratio_difference"
            ],
            errors="coerce",
        )

        matched_components = pd.to_numeric(
            group[
                "matched_bootstrap_component_number"
            ],
            errors="coerce",
        ).astype(int)

        mode_values = matched_components.mode()

        matched_mode = (
            int(
                mode_values.iloc[0]
            )
            if len(
                mode_values
            )
            else np.nan
        )

        summary_rows.append(
            {
                "matrix_representation": (
                    matrix_representation
                ),
                "reference_principal_component": (
                    f"PC{component_number}"
                ),
                "reference_component_number": (
                    component_number
                ),
                "bootstrap_replicates": len(
                    group
                ),
                "median_absolute_loading_correlation": float(
                    loading_correlations.median()
                ),
                "loading_correlation_2_5_percentile": float(
                    loading_correlations.quantile(
                        0.025
                    )
                ),
                "loading_correlation_97_5_percentile": float(
                    loading_correlations.quantile(
                        0.975
                    )
                ),
                "proportion_loading_correlation_ge_0_80": float(
                    (
                        loading_correlations
                        >= 0.80
                    ).mean()
                ),
                "proportion_loading_correlation_ge_0_90": float(
                    (
                        loading_correlations
                        >= 0.90
                    ).mean()
                ),
                "median_absolute_variance_ratio_difference": float(
                    variance_differences.median()
                ),
                "matched_bootstrap_component_mode": (
                    matched_mode
                ),
                "unique_matched_bootstrap_components": int(
                    matched_components.nunique()
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    return (
        replicates,
        summary,
    )


def clustering_analysis(
    representation: str,
    numeric: pd.DataFrame,
    metadata: pd.DataFrame,
    annotation: pd.DataFrame,
    reference: dict[str, object],
    random_seed: int,
) -> dict[str, pd.DataFrame]:
    standardized = np.asarray(
        reference[
            "standardized"
        ],
        dtype=float,
    )

    full_scores = np.asarray(
        reference[
            "scores"
        ],
        dtype=float,
    )

    components_for_80 = int(
        reference[
            "components_for_80_percent"
        ]
    )

    clustering_space = full_scores[
        :,
        :components_for_80
    ]

    participant_count = clustering_space.shape[0]

    subsample_size = int(
        round(
            participant_count
            * CLUSTER_SUBSAMPLE_FRACTION
        )
    )

    rng = np.random.default_rng(
        random_seed
    )

    metric_rows = []
    assignment_rows = []
    profile_rows = []
    marker_rows = []
    stability_rows = []

    context_labels = (
        pd.to_numeric(
            metadata[
                "previous_4vHPV_doses"
            ],
            errors="raise",
        )
        > 0
    ).astype(int).to_numpy()

    dose_labels = pd.to_numeric(
        metadata[
            "previous_4vHPV_doses"
        ],
        errors="raise",
    ).astype(int).to_numpy()

    feature_columns = list(
        numeric.columns
    )

    annotation_lookup = annotation.set_index(
        "feature_column"
    ).to_dict(
        orient="index"
    )

    full_label_registry: dict[
        int,
        np.ndarray,
    ] = {}

    for cluster_count in CLUSTER_K_VALUES:
        full_model = KMeans(
            n_clusters=cluster_count,
            n_init=100,
            random_state=(
                random_seed
                + cluster_count
            ),
            algorithm="lloyd",
        )

        full_labels = full_model.fit_predict(
            clustering_space
        )

        full_label_registry[
            cluster_count
        ] = full_labels

        observed_clusters = int(
            len(
                np.unique(
                    full_labels
                )
            )
        )

        if observed_clusters != cluster_count:
            raise RuntimeError(
                f"{representation}, k={cluster_count}: "
                f"observed only {observed_clusters} clusters."
            )

        cluster_sizes = np.bincount(
            full_labels,
            minlength=cluster_count,
        )

        silhouette = float(
            silhouette_score(
                clustering_space,
                full_labels,
                metric="euclidean",
            )
        )

        calinski_harabasz = float(
            calinski_harabasz_score(
                clustering_space,
                full_labels,
            )
        )

        davies_bouldin = float(
            davies_bouldin_score(
                clustering_space,
                full_labels,
            )
        )

        context_ari = float(
            adjusted_rand_score(
                context_labels,
                full_labels,
            )
        )

        context_nmi = float(
            normalized_mutual_info_score(
                context_labels,
                full_labels,
            )
        )

        dose_ari = float(
            adjusted_rand_score(
                dose_labels,
                full_labels,
            )
        )

        dose_nmi = float(
            normalized_mutual_info_score(
                dose_labels,
                full_labels,
            )
        )

        replicate_ari_values = []

        for replicate in range(
            1,
            CLUSTER_SUBSAMPLE_REPLICATES + 1,
        ):
            sampled_indices = rng.choice(
                participant_count,
                size=subsample_size,
                replace=False,
            )

            subsample_model = KMeans(
                n_clusters=cluster_count,
                n_init=50,
                random_state=(
                    random_seed
                    + 1000
                    * cluster_count
                    + replicate
                ),
                algorithm="lloyd",
            )

            subsample_model.fit(
                clustering_space[
                    sampled_indices,
                    :
                ]
            )

            assigned_labels, distances = (
                pairwise_distances_argmin_min(
                    clustering_space,
                    subsample_model.cluster_centers_,
                    metric="euclidean",
                )
            )

            replicate_ari = float(
                adjusted_rand_score(
                    full_labels,
                    assigned_labels,
                )
            )

            replicate_ari_values.append(
                replicate_ari
            )

            stability_rows.append(
                {
                    "matrix_representation": (
                        representation
                    ),
                    "cluster_count": (
                        cluster_count
                    ),
                    "subsample_replicate": (
                        replicate
                    ),
                    "subsample_participants": (
                        subsample_size
                    ),
                    "adjusted_rand_vs_full_solution": (
                        replicate_ari
                    ),
                    "mean_assignment_distance": float(
                        np.mean(
                            distances
                        )
                    ),
                    "maximum_assignment_distance": float(
                        np.max(
                            distances
                        )
                    ),
                }
            )

        replicate_ari_array = np.asarray(
            replicate_ari_values,
            dtype=float,
        )

        mean_stability_ari = float(
            np.mean(
                replicate_ari_array
            )
        )

        median_stability_ari = float(
            np.median(
                replicate_ari_array
            )
        )

        stability_ari_2_5 = float(
            np.quantile(
                replicate_ari_array,
                0.025,
            )
        )

        stability_ari_97_5 = float(
            np.quantile(
                replicate_ari_array,
                0.975,
            )
        )

        minimum_cluster_size = int(
            np.min(
                cluster_sizes
            )
        )

        stable_candidate = bool(
            mean_stability_ari >= 0.75
            and silhouette >= 0.20
            and minimum_cluster_size >= 8
        )

        composite_score = float(
            0.50
            * silhouette
            + 0.50
            * max(
                mean_stability_ari,
                0.0,
            )
        )

        metric_rows.append(
            {
                "matrix_representation": (
                    representation
                ),
                "cluster_count": (
                    cluster_count
                ),
                "pca_components_used": (
                    components_for_80
                ),
                "variance_threshold": 0.80,
                "minimum_cluster_size": (
                    minimum_cluster_size
                ),
                "maximum_cluster_size": int(
                    np.max(
                        cluster_sizes
                    )
                ),
                "silhouette_score": (
                    silhouette
                ),
                "calinski_harabasz_score": (
                    calinski_harabasz
                ),
                "davies_bouldin_score": (
                    davies_bouldin
                ),
                "mean_subsample_adjusted_rand": (
                    mean_stability_ari
                ),
                "median_subsample_adjusted_rand": (
                    median_stability_ari
                ),
                "subsample_adjusted_rand_2_5_percentile": (
                    stability_ari_2_5
                ),
                "subsample_adjusted_rand_97_5_percentile": (
                    stability_ari_97_5
                ),
                "context_adjusted_rand": (
                    context_ari
                ),
                "context_normalized_mutual_information": (
                    context_nmi
                ),
                "dose_group_adjusted_rand": (
                    dose_ari
                ),
                "dose_group_normalized_mutual_information": (
                    dose_nmi
                ),
                "stable_cluster_candidate": (
                    stable_candidate
                ),
                "clustering_composite_score": (
                    composite_score
                ),
            }
        )

        for participant_index, participant_row in metadata.reset_index(
            drop=True
        ).iterrows():
            assignment_rows.append(
                {
                    "matrix_representation": (
                        representation
                    ),
                    "cluster_count": (
                        cluster_count
                    ),
                    "participant_id": str(
                        participant_row[
                            "participant_id"
                        ]
                    ),
                    "previous_4vHPV_doses": int(
                        participant_row[
                            "previous_4vHPV_doses"
                        ]
                    ),
                    "analysis_context": str(
                        participant_row[
                            "analysis_context"
                        ]
                    ),
                    "cluster_label": int(
                        full_labels[
                            participant_index
                        ]
                        + 1
                    ),
                    "PC1_score": float(
                        full_scores[
                            participant_index,
                            0,
                        ]
                    ),
                    "PC2_score": float(
                        full_scores[
                            participant_index,
                            1,
                        ]
                    ),
                }
            )

        for cluster_index in range(
            cluster_count
        ):
            mask = (
                full_labels
                == cluster_index
            )

            cluster_metadata = metadata.loc[
                mask
            ]

            cluster_standardized = standardized[
                mask,
                :
            ]

            cluster_feature_means = np.mean(
                cluster_standardized,
                axis=0,
            )

            primary_count = int(
                (
                    pd.to_numeric(
                        cluster_metadata[
                            "previous_4vHPV_doses"
                        ],
                        errors="coerce",
                    )
                    == 0
                ).sum()
            )

            recall_count = int(
                (
                    pd.to_numeric(
                        cluster_metadata[
                            "previous_4vHPV_doses"
                        ],
                        errors="coerce",
                    )
                    > 0
                ).sum()
            )

            profile_rows.append(
                {
                    "matrix_representation": (
                        representation
                    ),
                    "cluster_count": (
                        cluster_count
                    ),
                    "cluster_label": (
                        cluster_index
                        + 1
                    ),
                    "participants": int(
                        np.sum(
                            mask
                        )
                    ),
                    "primary_participants": (
                        primary_count
                    ),
                    "recall_participants": (
                        recall_count
                    ),
                    "recall_fraction": float(
                        recall_count
                        / np.sum(
                            mask
                        )
                    ),
                    "dose0_participants": int(
                        (
                            pd.to_numeric(
                                cluster_metadata[
                                    "previous_4vHPV_doses"
                                ],
                                errors="coerce",
                            )
                            == 0
                        ).sum()
                    ),
                    "dose1_participants": int(
                        (
                            pd.to_numeric(
                                cluster_metadata[
                                    "previous_4vHPV_doses"
                                ],
                                errors="coerce",
                            )
                            == 1
                        ).sum()
                    ),
                    "dose2_participants": int(
                        (
                            pd.to_numeric(
                                cluster_metadata[
                                    "previous_4vHPV_doses"
                                ],
                                errors="coerce",
                            )
                            == 2
                        ).sum()
                    ),
                    "dose3_participants": int(
                        (
                            pd.to_numeric(
                                cluster_metadata[
                                    "previous_4vHPV_doses"
                                ],
                                errors="coerce",
                            )
                            == 3
                        ).sum()
                    ),
                    "mean_PC1_score": float(
                        np.mean(
                            full_scores[
                                mask,
                                0,
                            ]
                        )
                    ),
                    "mean_PC2_score": float(
                        np.mean(
                            full_scores[
                                mask,
                                1,
                            ]
                        )
                    ),
                }
            )

            highest_indices = np.argsort(
                cluster_feature_means
            )[
                ::-1
            ][
                :5
            ]

            lowest_indices = np.argsort(
                cluster_feature_means
            )[
                :5
            ]

            for direction, indices in [
                (
                    "highest_cluster_z_score",
                    highest_indices,
                ),
                (
                    "lowest_cluster_z_score",
                    lowest_indices,
                ),
            ]:
                for rank, feature_index in enumerate(
                    indices,
                    start=1,
                ):
                    feature_column = feature_columns[
                        feature_index
                    ]

                    feature_metadata = annotation_lookup[
                        feature_column
                    ]

                    marker_rows.append(
                        {
                            "matrix_representation": (
                                representation
                            ),
                            "cluster_count": (
                                cluster_count
                            ),
                            "cluster_label": (
                                cluster_index
                                + 1
                            ),
                            "marker_direction": (
                                direction
                            ),
                            "marker_rank": rank,
                            "feature_column": (
                                feature_column
                            ),
                            "antigen_target": (
                                feature_metadata[
                                    "antigen_target"
                                ]
                            ),
                            "assay_feature": (
                                feature_metadata[
                                    "assay_feature"
                                ]
                            ),
                            "feature_family": (
                                feature_metadata[
                                    "feature_family"
                                ]
                            ),
                            "cluster_mean_standardized_value": float(
                                cluster_feature_means[
                                    feature_index
                                ]
                            ),
                        }
                    )

    metrics = pd.DataFrame(
        metric_rows
    )

    metrics[
        "candidate_rank_within_representation"
    ] = (
        metrics[
            "clustering_composite_score"
        ]
        .rank(
            method="dense",
            ascending=False,
        )
        .astype(int)
    )

    stable_candidates = metrics[
        metrics[
            "stable_cluster_candidate"
        ]
    ].copy()

    if stable_candidates.empty:
        selected_cluster_count = np.nan
    else:
        selected_cluster_count = int(
            stable_candidates.sort_values(
                [
                    "clustering_composite_score",
                    "cluster_count",
                ],
                ascending=[
                    False,
                    True,
                ],
            )[
                "cluster_count"
            ].iloc[0]
        )

    metrics[
        "selected_cluster_candidate"
    ] = (
        metrics[
            "cluster_count"
        ]
        == selected_cluster_count
    )

    return {
        "metrics": metrics,
        "assignments": pd.DataFrame(
            assignment_rows
        ),
        "profiles": pd.DataFrame(
            profile_rows
        ),
        "markers": pd.DataFrame(
            marker_rows
        ),
        "stability_replicates": pd.DataFrame(
            stability_rows
        ),
    }


def main() -> None:
    for path in [
        RAW_INPUT,
        BPV_INPUT,
        METADATA_INPUT,
        C4A_DECISION_INPUT,
    ]:
        require_file(
            path
        )

    decision = pd.read_csv(
        C4A_DECISION_INPUT,
        sep="\t",
    )

    require_columns(
        decision,
        {
            "decision",
        },
        "Phase 2C4A decision table",
    )

    observed_decision = str(
        decision.loc[
            0,
            "decision",
        ]
    )

    if observed_decision != EXPECTED_C4A_DECISION:
        sys.exit(
            "ERROR: Phase 2C4A decision is "
            f"{observed_decision}; expected "
            f"{EXPECTED_C4A_DECISION}."
        )

    metadata = pd.read_csv(
        METADATA_INPUT,
        sep="\t",
        dtype={
            "participant_id": "string",
        },
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

    metadata[
        "previous_4vHPV_doses"
    ] = pd.to_numeric(
        metadata[
            "previous_4vHPV_doses"
        ],
        errors="raise",
    ).astype(int)

    if len(
        metadata
    ) != 80:
        sys.exit(
            "ERROR: Expected 80 participant metadata rows."
        )

    if metadata[
        "participant_id"
    ].duplicated().any():
        sys.exit(
            "ERROR: Duplicate participant IDs in metadata."
        )

    leave_one_family_outputs = []
    family_manifest_outputs = []
    bootstrap_replicate_outputs = []
    bootstrap_summary_outputs = []
    clustering_metric_outputs = []
    clustering_assignment_outputs = []
    clustering_profile_outputs = []
    clustering_marker_outputs = []
    clustering_stability_outputs = []

    failures: list[str] = []

    representation_summaries: dict[
        str,
        dict[str, object],
    ] = {}

    for representation_index, (
        representation,
        input_path,
    ) in enumerate(
        REPRESENTATIONS.items()
    ):
        matrix = pd.read_csv(
            input_path,
            sep="\t",
            dtype={
                "participant_id": "string",
            },
        )

        require_columns(
            matrix,
            {
                "participant_id",
            },
            f"{representation} matrix",
        )

        if len(
            matrix
        ) != 80:
            failures.append(
                f"{representation}: expected 80 rows, "
                f"observed {len(matrix)}."
            )

        if matrix[
            "participant_id"
        ].duplicated().any():
            failures.append(
                f"{representation}: duplicate participant IDs."
            )

        matrix_ids = matrix[
            "participant_id"
        ].astype(str).tolist()

        metadata_ids = metadata[
            "participant_id"
        ].astype(str).tolist()

        if matrix_ids != metadata_ids:
            failures.append(
                f"{representation}: participant order does not "
                "match metadata."
            )

        feature_columns = [
            column
            for column in matrix.columns
            if column != "participant_id"
        ]

        numeric = matrix[
            feature_columns
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )

        missing_values = int(
            numeric.isna()
            .sum()
            .sum()
        )

        if missing_values:
            failures.append(
                f"{representation}: {missing_values} missing values."
            )

        annotation = annotate_features(
            feature_columns
        )

        unresolved_features = annotation[
            (
                annotation[
                    "antigen_target"
                ]
                == "unresolved"
            )
            | (
                annotation[
                    "feature_family"
                ]
                == "unresolved"
            )
        ]

        if len(
            unresolved_features
        ):
            failures.append(
                f"{representation}: {len(unresolved_features)} "
                "unresolved feature annotations."
            )

        reference = fit_reference_pca(
            numeric
        )

        try:
            (
                leave_one_family,
                family_manifest,
            ) = leave_one_family_out_stability(
                representation,
                numeric,
                annotation,
                reference,
            )
        except Exception as error:
            failures.append(
                f"{representation}: leave-one-family-out error: "
                f"{error}"
            )

            continue

        (
            bootstrap_replicates,
            bootstrap_summary,
        ) = bootstrap_pca_stability(
            representation,
            numeric,
            reference,
            random_seed=(
                RANDOM_SEED
                + representation_index
                * 10000
            ),
        )

        clustering = clustering_analysis(
            representation,
            numeric,
            metadata,
            annotation,
            reference,
            random_seed=(
                RANDOM_SEED
                + representation_index
                * 100000
            ),
        )

        leave_one_family_outputs.append(
            leave_one_family
        )

        family_manifest_outputs.append(
            family_manifest
        )

        bootstrap_replicate_outputs.append(
            bootstrap_replicates
        )

        bootstrap_summary_outputs.append(
            bootstrap_summary
        )

        clustering_metric_outputs.append(
            clustering[
                "metrics"
            ]
        )

        clustering_assignment_outputs.append(
            clustering[
                "assignments"
            ]
        )

        clustering_profile_outputs.append(
            clustering[
                "profiles"
            ]
        )

        clustering_marker_outputs.append(
            clustering[
                "markers"
            ]
        )

        clustering_stability_outputs.append(
            clustering[
                "stability_replicates"
            ]
        )

        selected_rows = clustering[
            "metrics"
        ][
            clustering[
                "metrics"
            ][
                "selected_cluster_candidate"
            ]
        ]

        selected_k = (
            int(
                selected_rows[
                    "cluster_count"
                ].iloc[0]
            )
            if len(
                selected_rows
            )
            else np.nan
        )

        representation_summaries[
            representation
        ] = {
            "features": len(
                feature_columns
            ),
            "feature_families": int(
                annotation[
                    "feature_family"
                ].nunique()
            ),
            "components_for_80_percent_variance": (
                reference[
                    "components_for_80_percent"
                ]
            ),
            "selected_cluster_count": (
                selected_k
            ),
            "stable_cluster_candidates": int(
                clustering[
                    "metrics"
                ][
                    "stable_cluster_candidate"
                ].sum()
            ),
        }

    if failures:
        decision_value = (
            "PHASE2C4B_REPAIR_REQUIRED"
        )
    else:
        decision_value = (
            "READY_FOR_PHASE2C4C_IMMUNE_STATE_SYNTHESIS"
        )

    if leave_one_family_outputs:
        leave_one_family = pd.concat(
            leave_one_family_outputs,
            ignore_index=True,
        )

        family_manifest = pd.concat(
            family_manifest_outputs,
            ignore_index=True,
        )

        bootstrap_replicates = pd.concat(
            bootstrap_replicate_outputs,
            ignore_index=True,
        )

        bootstrap_summary = pd.concat(
            bootstrap_summary_outputs,
            ignore_index=True,
        )

        clustering_metrics = pd.concat(
            clustering_metric_outputs,
            ignore_index=True,
        )

        cluster_assignments = pd.concat(
            clustering_assignment_outputs,
            ignore_index=True,
        )

        cluster_profiles = pd.concat(
            clustering_profile_outputs,
            ignore_index=True,
        )

        cluster_markers = pd.concat(
            clustering_marker_outputs,
            ignore_index=True,
        )

        cluster_stability = pd.concat(
            clustering_stability_outputs,
            ignore_index=True,
        )
    else:
        leave_one_family = pd.DataFrame()
        family_manifest = pd.DataFrame()
        bootstrap_replicates = pd.DataFrame()
        bootstrap_summary = pd.DataFrame()
        clustering_metrics = pd.DataFrame()
        cluster_assignments = pd.DataFrame()
        cluster_profiles = pd.DataFrame()
        cluster_markers = pd.DataFrame()
        cluster_stability = pd.DataFrame()

    expected_counts = {
        "leave_one_family_rows": (
            len(
                leave_one_family
            ),
            40,
        ),
        "family_manifest_rows": (
            len(
                family_manifest
            ),
            8,
        ),
        "bootstrap_replicate_rows": (
            len(
                bootstrap_replicates
            ),
            3000,
        ),
        "bootstrap_summary_rows": (
            len(
                bootstrap_summary
            ),
            10,
        ),
        "clustering_metric_rows": (
            len(
                clustering_metrics
            ),
            10,
        ),
        "cluster_assignment_rows": (
            len(
                cluster_assignments
            ),
            800,
        ),
        "cluster_profile_rows": (
            len(
                cluster_profiles
            ),
            40,
        ),
        "cluster_marker_rows": (
            len(
                cluster_markers
            ),
            400,
        ),
        "cluster_stability_rows": (
            len(
                cluster_stability
            ),
            1000,
        ),
    }

    for label, (
        observed,
        expected,
    ) in expected_counts.items():
        if observed != expected:
            failures.append(
                f"{label}: expected {expected}, "
                f"observed {observed}."
            )

    if len(
        leave_one_family
    ):
        if leave_one_family[
            [
                "absolute_loading_correlation",
                "sign_aligned_score_correlation",
            ]
        ].isna().any().any():
            failures.append(
                "Missing leave-one-family-out stability correlations."
            )

    if len(
        bootstrap_summary
    ):
        if bootstrap_summary[
            "median_absolute_loading_correlation"
        ].isna().any():
            failures.append(
                "Missing bootstrap loading-correlation summaries."
            )

    if len(
        clustering_metrics
    ):
        required_metric_columns = [
            "silhouette_score",
            "calinski_harabasz_score",
            "davies_bouldin_score",
            "mean_subsample_adjusted_rand",
        ]

        if clustering_metrics[
            required_metric_columns
        ].isna().any().any():
            failures.append(
                "Missing clustering quality or stability metrics."
            )

    decision_value = (
        "READY_FOR_PHASE2C4C_IMMUNE_STATE_SYNTHESIS"
        if not failures
        else "PHASE2C4B_REPAIR_REQUIRED"
    )

    family_manifest_output = (
        TABLES
        / "phase2C4B_fiji_feature_family_omission_manifest.tsv"
    )

    leave_one_family_output = (
        TABLES
        / "phase2C4B_fiji_leave_one_family_out_pca_stability.tsv"
    )

    bootstrap_replicate_output = (
        PROCESSED
        / "phase2C4B_fiji_bootstrap_pca_replicates.tsv"
    )

    bootstrap_summary_output = (
        TABLES
        / "phase2C4B_fiji_bootstrap_pca_stability_summary.tsv"
    )

    clustering_metrics_output = (
        TABLES
        / "phase2C4B_fiji_clustering_metrics.tsv"
    )

    clustering_assignments_output = (
        TABLES
        / "phase2C4B_fiji_cluster_assignments.tsv"
    )

    cluster_profiles_output = (
        TABLES
        / "phase2C4B_fiji_cluster_profiles.tsv"
    )

    cluster_markers_output = (
        TABLES
        / "phase2C4B_fiji_cluster_feature_markers.tsv"
    )

    cluster_stability_output = (
        PROCESSED
        / "phase2C4B_fiji_cluster_stability_replicates.tsv"
    )

    decision_output = (
        TABLES
        / "phase2C4B_fiji_pca_stability_clustering_decision.tsv"
    )

    write_tsv(
        family_manifest,
        family_manifest_output,
    )

    write_tsv(
        leave_one_family,
        leave_one_family_output,
    )

    write_tsv(
        bootstrap_replicates,
        bootstrap_replicate_output,
    )

    write_tsv(
        bootstrap_summary,
        bootstrap_summary_output,
    )

    write_tsv(
        clustering_metrics,
        clustering_metrics_output,
    )

    write_tsv(
        cluster_assignments,
        clustering_assignments_output,
    )

    write_tsv(
        cluster_profiles,
        cluster_profiles_output,
    )

    write_tsv(
        cluster_markers,
        cluster_markers_output,
    )

    write_tsv(
        cluster_stability,
        cluster_stability_output,
    )

    raw_summary = representation_summaries.get(
        "raw_log2_change",
        {},
    )

    bpv_summary = representation_summaries.get(
        "bpv_calibrated_log2_change",
        {},
    )

    decision_frame = pd.DataFrame(
        [
            {
                "decision": decision_value,
                "leave_one_family_rows": len(
                    leave_one_family
                ),
                "family_manifest_rows": len(
                    family_manifest
                ),
                "bootstrap_replicate_rows": len(
                    bootstrap_replicates
                ),
                "bootstrap_summary_rows": len(
                    bootstrap_summary
                ),
                "clustering_metric_rows": len(
                    clustering_metrics
                ),
                "cluster_assignment_rows": len(
                    cluster_assignments
                ),
                "cluster_profile_rows": len(
                    cluster_profiles
                ),
                "cluster_marker_rows": len(
                    cluster_markers
                ),
                "cluster_stability_rows": len(
                    cluster_stability
                ),
                "raw_components_for_80_percent_variance": (
                    raw_summary.get(
                        "components_for_80_percent_variance",
                        np.nan,
                    )
                ),
                "bpv_components_for_80_percent_variance": (
                    bpv_summary.get(
                        "components_for_80_percent_variance",
                        np.nan,
                    )
                ),
                "raw_stable_cluster_candidates": (
                    raw_summary.get(
                        "stable_cluster_candidates",
                        np.nan,
                    )
                ),
                "bpv_stable_cluster_candidates": (
                    bpv_summary.get(
                        "stable_cluster_candidates",
                        np.nan,
                    )
                ),
                "raw_selected_cluster_count": (
                    raw_summary.get(
                        "selected_cluster_count",
                        np.nan,
                    )
                ),
                "bpv_selected_cluster_count": (
                    bpv_summary.get(
                        "selected_cluster_count",
                        np.nan,
                    )
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
        / "phase2C4B_fiji_pca_stability_and_clustering_report.md"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as report:
        report.write(
            "# Phase 2C4B Fiji PCA stability and clustering\n\n"
        )

        report.write(
            "## Decision\n\n"
        )

        report.write(
            f"**{decision_value}**\n\n"
        )

        report.write(
            "## PCA stability design\n\n"
        )

        report.write(
            f"- Leave-one-feature-family-out analyses: "
            f"{len(family_manifest)}\n"
        )

        report.write(
            f"- Stability rows across PC1–PC5: "
            f"{len(leave_one_family)}\n"
        )

        report.write(
            f"- Participant-bootstrap replicates per representation: "
            f"{BOOTSTRAP_REPLICATES}\n"
        )

        report.write(
            f"- Bootstrap component-level rows: "
            f"{len(bootstrap_replicates)}\n\n"
        )

        report.write(
            "Components were matched to the full-data reference PCA "
            "using maximum absolute loading correlation and their signs "
            "were aligned before score comparison. This avoids treating "
            "arbitrary PCA sign changes or component reordering as "
            "biological instability.\n\n"
        )

        report.write(
            "## Clustering design\n\n"
        )

        report.write(
            "- Candidate cluster counts: k=2 through k=6\n"
        )

        report.write(
            "- Clustering space: representation-specific PCA scores "
            "required to explain at least 80% of variance\n"
        )

        report.write(
            f"- Participant-subsampling stability replicates per k: "
            f"{CLUSTER_SUBSAMPLE_REPLICATES}\n"
        )

        report.write(
            "- A stable candidate requires mean subsample adjusted Rand "
            "index of at least 0.75, silhouette score of at least 0.20 "
            "and a minimum cluster size of at least eight participants.\n\n"
        )

        report.write(
            "A selected k is reported only when these prespecified "
            "criteria are met. Failure to identify a stable partition "
            "should be interpreted as evidence for continuous immune-state "
            "structure rather than as an analytical failure.\n\n"
        )

        report.write(
            "## Phase 2C4A biological reference\n\n"
        )

        report.write(
            "PC1 represents a cross-reactive recall-breadth axis dominated "
            "by HPV31/45/52/58 IgG, IgG-subclass and Fc-receptor features "
            "opposed to vaccine-type HPV16 features. PC2 represents a "
            "vaccine-type HPV16/18 abundance and effector axis. The raw "
            "PC2 previous-dose association was not retained after BPV "
            "calibration and therefore remains a qualified secondary "
            "schedule observation.\n"
        )

    print(
        "===== PHASE 2C4B COMPLETE ====="
    )

    print(
        f"Decision: {decision_value}"
    )

    print(
        "Leave-one-family-out rows:",
        len(
            leave_one_family
        ),
    )

    print(
        "Feature-family manifest rows:",
        len(
            family_manifest
        ),
    )

    print(
        "Bootstrap replicate rows:",
        len(
            bootstrap_replicates
        ),
    )

    print(
        "Bootstrap summary rows:",
        len(
            bootstrap_summary
        ),
    )

    print(
        "Clustering metric rows:",
        len(
            clustering_metrics
        ),
    )

    print(
        "Cluster assignment rows:",
        len(
            cluster_assignments
        ),
    )

    print(
        "Cluster profile rows:",
        len(
            cluster_profiles
        ),
    )

    print(
        "Cluster marker rows:",
        len(
            cluster_markers
        ),
    )

    print(
        "Cluster stability rows:",
        len(
            cluster_stability
        ),
    )

    print(
        "Raw selected cluster count:",
        raw_summary.get(
            "selected_cluster_count",
            np.nan,
        ),
    )

    print(
        "BPV-calibrated selected cluster count:",
        bpv_summary.get(
            "selected_cluster_count",
            np.nan,
        ),
    )

    print(
        f"Report: {report_path}"
    )

    if failures:
        print(
            "\nValidation failures:"
        )

        for failure in failures:
            print(
                f"- {failure}"
            )

        sys.exit(
            1
        )


if __name__ == "__main__":
    main()
