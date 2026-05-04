from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOGGER = logging.getLogger(__name__)
EXCLUDED_HUMAN_PANEL_SUFFIXES = {"homo", "homo_sapiens"}

BASELINE_NUMERIC = [
    "Shannon_diversity",
    "number_of_detected_species",
    "sequencing_depth_proxy",
]
BASELINE_CATEGORICAL = ["age_group"]


@dataclass(frozen=True)
class ClassifierSpec:
    name: str
    estimator: Any
    param_grid: dict[str, list[Any]]


def _default_classifier_specs(random_state: int) -> list[ClassifierSpec]:
    return [
        ClassifierSpec(
            name="logistic_regression",
            estimator=LogisticRegression(max_iter=2000, solver="liblinear", random_state=random_state),
            param_grid={
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
        ClassifierSpec(
            name="random_forest",
            estimator=RandomForestClassifier(random_state=random_state),
            param_grid={
                "model__n_estimators": [200, 500],
                "model__max_depth": [None, 5, 10],
                "model__min_samples_leaf": [1, 2, 5],
                "model__class_weight": [None, "balanced"],
            },
        ),
    ]


def _rank_top_taxa(train_df: pd.DataFrame, taxa_cols: list[str], top_n: int) -> list[str]:
    if not taxa_cols or top_n <= 0:
        return []

    taxa_train = train_df[taxa_cols].copy()
    prevalence = (taxa_train > 0).mean(axis=0)
    mean_abundance = taxa_train.mean(axis=0)
    score = mean_abundance * prevalence

    ordered = (
        pd.DataFrame({"taxon": score.index, "score": score.values})
        .sort_values(["score", "taxon"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return ordered.head(min(top_n, len(ordered)))["taxon"].tolist()


def _panel_taxa_columns(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if c.startswith("panel_")
        and not c.endswith("_log10")
        and not c.endswith("_present")
        and c.removeprefix("panel_").lower() not in EXCLUDED_HUMAN_PANEL_SUFFIXES
    ]


def _build_preprocessor(
    feature_columns: list[str], selected_taxa: list[str], age_group_categories: list[str] | None = None
) -> ColumnTransformer:
    numeric_features = [c for c in BASELINE_NUMERIC if c in feature_columns] + [
        c for c in selected_taxa if c in feature_columns
    ]
    categorical_features = [c for c in BASELINE_CATEGORICAL if c in feature_columns]

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_features,
            )
        )
    if categorical_features:
        encoder_kwargs: dict[str, Any] = {"handle_unknown": "ignore"}
        if age_group_categories is not None and "age_group" in categorical_features:
            encoder_kwargs["categories"] = [age_group_categories]
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("onehot", OneHotEncoder(**encoder_kwargs)),
                    ]
                ),
                categorical_features,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _feature_sets(selected_taxa: list[str]) -> dict[str, list[str]]:
    baseline = BASELINE_NUMERIC + BASELINE_CATEGORICAL
    return {
        "baseline_only": baseline,
        "taxa_only": selected_taxa,
        "combined": baseline + selected_taxa,
    }


def _safe_metric(func: Any, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(func(y_true, y_pred))
    except ValueError:
        return float("nan")


def run_nested_cv_modeling(
    feature_matrix_path: Path | str = Path("data/features/targeted_feature_matrix.csv"),
    cv_results_path: Path | str = Path("reports/model_cv_results.csv"),
    selected_features_path: Path | str = Path("reports/model_selected_features_by_fold.csv"),
    *,
    top_taxa_n: int = 20,
    outer_splits: int = 5,
    inner_splits: int = 4,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run nested CV for healthy-vs-cancer classification.

    For each outer fold, top taxa are recomputed using only that fold's training samples.
    Three feature-set variants are evaluated: baseline covariates only, taxa only,
    and the combination.
    """

    feature_matrix_path = Path(feature_matrix_path)
    cv_results_path = Path(cv_results_path)
    selected_features_path = Path(selected_features_path)

    df = pd.read_csv(feature_matrix_path)
    if df.empty:
        raise ValueError(f"Feature matrix is empty: {feature_matrix_path}")

    if "label" not in df.columns:
        raise ValueError("Expected 'label' column in feature matrix.")

    label_map = {"healthy": 0, "cancer": 1}
    df = df[df["label"].isin(label_map.keys())].copy()
    if df.empty:
        raise ValueError("No rows with supported labels ('healthy', 'cancer').")

    y = df["label"].map(label_map).to_numpy()
    metadata_cols = {"sample_id", "run_accession", "label", "library_name"}
    taxa_cols = _panel_taxa_columns(df)
    if not taxa_cols:
        raise ValueError("No taxa panel columns found (expected columns beginning with 'panel_').")

    candidate_feature_cols = [c for c in df.columns if c not in metadata_cols]
    feature_df = df[candidate_feature_cols].copy()
    if "age_group" in feature_df.columns:
        feature_df["age_group"] = feature_df["age_group"].astype(str).str.strip().str.lower()
        unknown_mask = feature_df["age_group"].isin({"", "unknown", "na", "n/a", "none", "nan"})
        feature_df.loc[unknown_mask, "age_group"] = "__unknown__"
        known_age_groups = sorted(
            x
            for x in feature_df["age_group"].dropna().unique().tolist()
            if x != "__unknown__"
        )
    else:
        known_age_groups = []

    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    clf_specs = _default_classifier_specs(random_state=random_state)

    fold_results: list[dict[str, Any]] = []
    fold_features: list[dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(feature_df, y), start=1):
        train_df = feature_df.iloc[train_idx].copy()
        val_df = feature_df.iloc[val_idx].copy()
        y_train, y_val = y[train_idx], y[val_idx]

        selected_taxa = _rank_top_taxa(train_df=train_df, taxa_cols=taxa_cols, top_n=top_taxa_n)
        if not selected_taxa:
            raise ValueError(f"Fold {fold_idx}: no taxa selected. Check panel taxa columns.")

        fold_feature_sets = _feature_sets(selected_taxa)

        for feature_set_name, chosen_cols in fold_feature_sets.items():
            usable_cols = [c for c in chosen_cols if c in feature_df.columns]
            if not usable_cols:
                continue

            preprocessor = _build_preprocessor(
                feature_columns=usable_cols,
                selected_taxa=selected_taxa,
                age_group_categories=known_age_groups if known_age_groups else None,
            )

            x_train = train_df[usable_cols]
            x_val = val_df[usable_cols]

            for clf in clf_specs:
                model = Pipeline([
                    ("preprocessor", preprocessor),
                    ("model", clone(clf.estimator)),
                ])

                inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
                search = GridSearchCV(
                    estimator=model,
                    param_grid=clf.param_grid,
                    scoring="roc_auc",
                    cv=inner_cv,
                    n_jobs=-1,
                    refit=True,
                )
                search.fit(x_train, y_train)

                best_model: Pipeline = search.best_estimator_
                val_proba = best_model.predict_proba(x_val)[:, 1]
                val_pred = (val_proba >= 0.5).astype(int)

                fold_results.append(
                    {
                        "fold": fold_idx,
                        "feature_set": feature_set_name,
                        "classifier": clf.name,
                        "n_train": len(train_idx),
                        "n_val": len(val_idx),
                        "inner_best_score_roc_auc": float(search.best_score_),
                        "val_roc_auc": _safe_metric(roc_auc_score, y_val, val_proba),
                        "val_accuracy": _safe_metric(accuracy_score, y_val, val_pred),
                        "val_f1": _safe_metric(f1_score, y_val, val_pred),
                        "val_precision": _safe_metric(precision_score, y_val, val_pred),
                        "val_recall": _safe_metric(recall_score, y_val, val_pred),
                        "best_params": str(search.best_params_),
                    }
                )

                fold_features.append(
                    {
                        "fold": fold_idx,
                        "feature_set": feature_set_name,
                        "classifier": clf.name,
                        "selected_taxa": "|".join(selected_taxa),
                        "used_features": "|".join(usable_cols),
                        "n_selected_taxa": len(selected_taxa),
                        "n_used_features": len(usable_cols),
                    }
                )

    results_df = pd.DataFrame(fold_results)
    features_df = pd.DataFrame(fold_features)

    cv_results_path.parent.mkdir(parents=True, exist_ok=True)
    selected_features_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(cv_results_path, index=False)
    features_df.to_csv(selected_features_path, index=False)

    LOGGER.info("Nested CV results written to %s", cv_results_path)
    LOGGER.info("Selected features by fold written to %s", selected_features_path)

    return results_df, features_df


if __name__ == "__main__":
    run_nested_cv_modeling()
