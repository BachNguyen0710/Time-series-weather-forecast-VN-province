# -*- coding: utf-8 -*-

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from tabulate import tabulate

SRC_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_ROOT))

from Preprocess import create_features, split_and_scale


DATA_PATH = Path("data/processed/data.csv")
REPORTS_DIR = Path("reports") / "decision_tree"
MODELS_DIR = Path("models")


# ------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------

def ensure_path(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dirs():
    ensure_path(REPORTS_DIR)
    ensure_path(MODELS_DIR)


# ------------------------------------------------------------------
# Data preparation
# ------------------------------------------------------------------

def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)
    df_processed = create_features(df)
    feature_names = [
        "temperature",
        "humidity",
        "wind",
        "pressure",
        "hour",
        "dayofweek",
        "month",
        "rain_lag1",
        "rain_lag3",
        "rain_lag24",
        "rain_roll3",
        "rain_roll24",
    ] + [col for col in df_processed.columns if col.startswith("city_")]

    (X_train, y_train), (X_val, y_val), (X_test, y_test), _ = split_and_scale(df_processed)
    return (
        X_train,
        y_train.values,
        X_val,
        y_val.values,
        X_test,
        y_test.values,
        feature_names,
    )


def build_decision_tree(max_depth=6, min_samples_leaf=10):
    model = DecisionTreeRegressor(
        criterion="squared_error",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    return model


def fit_model(model, X, y):
    model.fit(X, y)
    return model


# ------------------------------------------------------------------
# Grid search helpers
# ------------------------------------------------------------------

def _log_grid_search_results(grid: GridSearchCV, outdir: Path):
    cv_df = pd.DataFrame(grid.cv_results_)
    cv_df["mean_test_rmse"] = (-cv_df["mean_test_score"]).round(4)
    cv_df.to_csv(outdir / "grid_search_results.csv", index=False)

    summary = (
        cv_df
        .sort_values("rank_test_score")
        .loc[
            :,
            [
                "param_max_depth",
                "param_min_samples_leaf",
                "param_min_samples_split",
                "mean_test_rmse",
                "rank_test_score",
            ],
        ]
        .rename(
            columns={
                "param_max_depth": "max_depth",
                "param_min_samples_leaf": "min_samples_leaf",
                "param_min_samples_split": "min_samples_split",
                "rank_test_score": "rank",
            }
        )
        .head(5)
        .round({"mean_test_rmse": 4})
    )
    summary.to_csv(outdir / "grid_search_top5.csv", index=False)

    print("\nGrid search top 5 runs (sorted by CV RMSE):")
    print(tabulate(summary, headers="keys", tablefmt="github", showindex=False))


def train_with_grid(X_train, y_train):
    grid_dir = ensure_path(REPORTS_DIR / "grid_search")
    param_grid = {
        "max_depth": [4, 6, 8, 10, 12, 14, 16],
        "min_samples_leaf": [5, 10, 15, 20, 30],
        "min_samples_split": [5, 10, 20, 30, 40],
    }
    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(
        DecisionTreeRegressor(random_state=42),
        param_grid,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
        verbose=1,
        return_train_score=False,
    )
    print("Running GridSearchCV (TimeSeriesSplit + RMSE)...")
    grid.fit(X_train, y_train)
    print("Best params:", grid.best_params_)
    print(f"Best CV RMSE: {-grid.best_score_:.4f}")
    _log_grid_search_results(grid, grid_dir)
    return grid.best_estimator_


# ------------------------------------------------------------------
# Evaluation & visualization
# ------------------------------------------------------------------

def evaluate_model(model, X, y, feature_names, prefix="validation"):
    outdir = ensure_path(REPORTS_DIR / prefix)
    prediction = model.predict(X)

    rmse = np.sqrt(mean_squared_error(y, prediction))
    mae = mean_absolute_error(y, prediction)
    r2 = r2_score(y, prediction)

    metrics_df = pd.DataFrame(
        {
            "metric": ["rmse", "mae", "r2"],
            "value": [rmse, mae, r2],
        }
    )
    metrics_df["value"] = metrics_df["value"].round(4)
    metrics_df.to_csv(outdir / "metrics.csv", index=False)

    print(f"\n[{prefix.upper()}] Evaluation")
    print(tabulate(metrics_df, headers="keys", tablefmt="github", showindex=False))

    plt.figure(figsize=(6, 6))
    plt.scatter(y, prediction, alpha=0.4, s=10)
    min_val = min(y.min(), prediction.min())
    max_val = max(y.max(), prediction.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
    plt.title(f"Actual vs Predicted ({prefix})")
    plt.xlabel("Actual rain")
    plt.ylabel("Predicted rain")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / "actual_vs_predicted.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    plot_tree(
        model,
        feature_names=feature_names,
        filled=True,
        max_depth=3,
        fontsize=8,
    )
    plt.title(f"Decision Tree Structure ({prefix})")
    plt.tight_layout()
    plt.savefig(outdir / "tree_structure.png")
    plt.close()


def plot_feature_importances(model, feature_names, top_n=20):
    outdir = ensure_path(REPORTS_DIR / "feature_importances")
    if not hasattr(model, "feature_importances_"):
        print("Feature importances are unavailable for this estimator.")
        return

    importance_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
    )
    importance_df.to_csv(outdir / "feature_importances.csv", index=False)

    plot_df = importance_df.head(top_n).iloc[::-1]
    plt.figure(figsize=(10, max(2.5, plot_df.shape[0] * 0.4)))
    plt.barh(plot_df["feature"], plot_df["importance"], color="#1f77b4")
    plt.title("Top Decision Tree Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(outdir / "feature_importances.png")
    plt.close()


# ------------------------------------------------------------------
# Model persistence
# ------------------------------------------------------------------

def save_model(model, path: Path):
    dump(model, path)
    print(f"Model saved to {path}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main(use_grid=True):
    ensure_dirs()
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        feature_names,
    ) = load_and_prepare_data()

    if use_grid:
        model = train_with_grid(X_train, y_train)
        model_name = MODELS_DIR / "decision_tree_weather_grid.joblib"
    else:
        model = build_decision_tree()
        model = fit_model(model, X_train, y_train)
        model_name = MODELS_DIR / "decision_tree_weather_baseline.joblib"

    plot_feature_importances(model, feature_names)
    evaluate_model(model, X_val, y_val, feature_names, prefix="validation")
    evaluate_model(model, X_test, y_test, feature_names, prefix="test")

    save_model(model, model_name)


if __name__ == "__main__":
    main()
