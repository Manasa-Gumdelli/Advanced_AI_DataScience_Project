import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    confusion_matrix
)

from config import OUTPUT_DIR


def save_results_table(results_df: pd.DataFrame):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "model_metrics.csv"
    results_df.to_csv(path, index=False)
    return path


def plot_class_distribution(df: pd.DataFrame):
    counts = df["readmission_label"].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")
    plt.title("Readmission Class Distribution")
    plt.xlabel("Readmission Category")
    plt.ylabel("Count")
    plt.tight_layout()

    path = OUTPUT_DIR / "class_distribution.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_missing_values(df: pd.DataFrame):
    missing = (df.isna().mean() * 100).sort_values(ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    missing.sort_values().plot(kind="barh")
    plt.title("Top Missing Value Percentages")
    plt.xlabel("Missing Percentage")
    plt.ylabel("Feature")
    plt.tight_layout()

    path = OUTPUT_DIR / "missing_values.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_model_comparison(results_df: pd.DataFrame):
    metrics = results_df.set_index("model")[[
        "accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"
    ]]

    ax = metrics.plot(kind="bar", figsize=(11, 6))
    ax.set_title("Model Comparison Across Evaluation Metrics")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    plt.tight_layout()

    path = OUTPUT_DIR / "model_comparison.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_roc_curves(trained_models):
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    for model_name, artifact in trained_models.items():
        model = artifact["model"]
        X_test = artifact["X_test"]
        y_test = artifact["y_test"]

        if artifact["type"] == "pipeline":
            RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=model_name)
        else:
            y_prob = artifact["y_prob"]
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc(fpr, tpr):.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    plt.tight_layout()

    path = OUTPUT_DIR / "roc_curves.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_pr_curves(trained_models):
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    for model_name, artifact in trained_models.items():
        y_test = artifact["y_test"]
        y_prob = artifact["y_prob"]
        PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=ax, name=model_name)

    ax.set_title("Precision-Recall Curves")
    plt.tight_layout()

    path = OUTPUT_DIR / "pr_curves.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_threshold_vs_f1(trained_models):
    plt.figure(figsize=(9, 6))

    for model_name, artifact in trained_models.items():
        y_test = artifact["y_test"]
        y_prob = artifact["y_prob"]

        thresholds = np.arange(0.10, 0.81, 0.02)
        f1_scores = []

        from sklearn.metrics import f1_score
        for t in thresholds:
            preds = (y_prob >= t).astype(int)
            f1_scores.append(f1_score(y_test, preds, zero_division=0))

        plt.plot(thresholds, f1_scores, label=model_name)

    plt.title("Threshold vs F1 Score")
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.tight_layout()

    path = OUTPUT_DIR / "threshold_vs_f1.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_probability_distribution(trained_models):
    for model_name, artifact in trained_models.items():
        y_prob = artifact["y_prob"]

        plt.figure(figsize=(8, 5))
        plt.hist(y_prob, bins=30)
        plt.title(f"Predicted Probability Distribution - {model_name}")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Frequency")
        plt.tight_layout()

        path = OUTPUT_DIR / f"probability_distribution_{model_name}.png"
        plt.savefig(path, dpi=200)
        plt.close()


def plot_confusion_matrices(results_df: pd.DataFrame):
    for _, row in results_df.iterrows():
        model_name = row["model"]
        tn = row["true_negative"]
        fp = row["false_positive"]
        fn = row["false_negative"]
        tp = row["true_positive"]

        cm = np.array([[tn, fp], [fn, tp]])

        plt.figure(figsize=(5, 4))
        plt.imshow(cm, interpolation="nearest")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.colorbar()
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["Actual 0", "Actual 1"])

        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha="center", va="center")

        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()

        path = OUTPUT_DIR / f"confusion_matrix_{model_name}.png"
        plt.savefig(path, dpi=200)
        plt.close()


def plot_feature_importance(best_model_name, trained_models):
    artifact = trained_models[best_model_name]
    model = artifact["model"]

    if artifact["type"] == "pipeline":
        preprocessor = model.named_steps["preprocessor"]
        estimator = model.named_steps["model"]
        feature_names = preprocessor.get_feature_names_out()

        if hasattr(estimator, "feature_importances_"):
            importance = estimator.feature_importances_
        else:
            importance = abs(estimator.coef_[0])

    else:
        X_test = artifact["X_test"]
        feature_names = X_test.columns

        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        else:
            return None

    importance_df = (
        pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        })
        .sort_values("importance", ascending=False)
        .head(20)
    )

    plt.figure(figsize=(10, 7))
    plt.barh(importance_df["feature"][::-1], importance_df["importance"][::-1])
    plt.title(f"Top 20 Important Features - {best_model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()

    png_path = OUTPUT_DIR / "feature_importance.png"
    csv_path = OUTPUT_DIR / "feature_importance.csv"

    plt.savefig(png_path, dpi=200)
    plt.close()
    importance_df.to_csv(csv_path, index=False)

    return png_path


def save_best_model(best_model_name, trained_models):
    artifact = trained_models[best_model_name]
    path = OUTPUT_DIR / f"{best_model_name}_model.joblib"
    joblib.dump(artifact["model"], path)
    return path