import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config import RANDOM_STATE, TEST_SIZE
from data_preprocessing import build_preprocessor

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

BALANCED_RF_AVAILABLE = True


def get_sklearn_models():
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=180,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=8,
            max_iter=250,
            random_state=RANDOM_STATE
        )
    }


def choose_best_threshold(y_true, y_prob):
    thresholds = np.arange(0.10, 0.81, 0.02)
    best_threshold = 0.50
    best_f1 = -1

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return best_threshold, best_f1


def evaluate_predictions(model_name, y_test, y_prob):
    best_threshold, tuned_f1 = choose_best_threshold(y_test, y_prob)
    y_pred = (y_prob >= best_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "model": model_name,
        "threshold": round(best_threshold, 2),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": tuned_f1,
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp)
    }


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    trained_models = {}
    rows = []

    preprocessor = build_preprocessor(X_train)
    sklearn_models = get_sklearn_models()

    for model_name, model in sklearn_models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe.named_steps["model"], "predict_proba") else pipe.decision_function(X_test)

        if y_prob.ndim > 1:
            y_prob = y_prob[:, 1]

        rows.append(evaluate_predictions(model_name, y_test, y_prob))
        trained_models[model_name] = {
            "type": "pipeline",
            "model": pipe,
            "X_test": X_test,
            "y_test": y_test,
            "y_prob": y_prob
        }

    if BALANCED_RF_AVAILABLE:
        brf_model = BalancedRandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        brf_pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", brf_model)
        ])

        brf_pipe.fit(X_train, y_train)
        y_prob = brf_pipe.predict_proba(X_test)[:, 1]

        rows.append(evaluate_predictions("balanced_random_forest", y_test, y_prob))
        trained_models["balanced_random_forest"] = {
            "type": "pipeline",
            "model": brf_pipe,
            "X_test": X_test,
            "y_test": y_test,
            "y_prob": y_prob
        }

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "trained_models": trained_models,
        "results": results_df
    }