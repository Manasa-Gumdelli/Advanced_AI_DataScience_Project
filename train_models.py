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

LIGHTGBM_AVAILABLE = True
CATBOOST_AVAILABLE = True
XGBOOST_AVAILABLE = True
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

    if XGBOOST_AVAILABLE:
        xgb_pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", XGBClassifier(
                n_estimators=250,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ])

        xgb_pipe.fit(X_train, y_train)
        y_prob = xgb_pipe.predict_proba(X_test)[:, 1]

        rows.append(evaluate_predictions("xgboost", y_test, y_prob))
        trained_models["xgboost"] = {
            "type": "pipeline",
            "model": xgb_pipe,
            "X_test": X_test,
            "y_test": y_test,
            "y_prob": y_prob
        }

    if LIGHTGBM_AVAILABLE:
        X_train_lgb = X_train.copy()
        X_test_lgb = X_test.copy()

        cat_cols = X_train_lgb.select_dtypes(include=["object"]).columns.tolist()
        for col in cat_cols:
            X_train_lgb[col] = X_train_lgb[col].astype("category")
            X_test_lgb[col] = X_test_lgb[col].astype("category")

        lgb_model = LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            verbosity=-1
        )

        lgb_model.fit(X_train_lgb, y_train, categorical_feature=cat_cols)
        y_prob = lgb_model.predict_proba(X_test_lgb)[:, 1]

        rows.append(evaluate_predictions("lightgbm", y_test, y_prob))
        trained_models["lightgbm"] = {
            "type": "native",
            "model": lgb_model,
            "X_test": X_test_lgb,
            "y_test": y_test,
            "y_prob": y_prob
        }

    if CATBOOST_AVAILABLE:
        X_train_cb = X_train.copy().fillna("Missing")
        X_test_cb = X_test.copy().fillna("Missing")

        cat_cols = X_train_cb.select_dtypes(include=["object"]).columns.tolist()
        cat_idx = [X_train_cb.columns.get_loc(c) for c in cat_cols]

        cb_model = CatBoostClassifier(
            iterations=250,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=False,
            random_seed=RANDOM_STATE,
            auto_class_weights="Balanced"
        )

        cb_model.fit(X_train_cb, y_train, cat_features=cat_idx)
        y_prob = cb_model.predict_proba(X_test_cb)[:, 1]

        rows.append(evaluate_predictions("catboost", y_test, y_prob))
        trained_models["catboost"] = {
            "type": "native",
            "model": cb_model,
            "X_test": X_test_cb,
            "y_test": y_test,
            "y_prob": y_prob
        }

    results_df = (
        pd.DataFrame(rows)
        .sort_values(["f1_score", "pr_auc", "roc_auc"], ascending=False)
        .reset_index(drop=True)
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "trained_models": trained_models,
        "results": results_df
    }