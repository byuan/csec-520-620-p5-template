"""YOUR CODE GOES HERE: preprocessing and the three SVM tasks of Project 5.

Fill in the four functions marked TODO. Each returns a plain dictionary; src/p5.py
collects them into results/metrics.json (format: grading/P5_METRICS.md) and draws the
figures. Read the handout for the full task description.

Rules that the grader and rubric check:
  * Fit every preprocessing step on TRAINING data only: put the preprocessor and the
    SVM in one sklearn Pipeline, e.g. make_pipeline(make_preprocessor(), LinearSVC(...)).
  * Never tune on the testing set. Choose C, gamma and nu with cross-validation (or
    other training-only evidence); use the testing set only to report final numbers.
  * Compute every metric with the provided helpers in src/metrics.py.
  * Record training time with time.perf_counter() around the final .fit(...).

Useful imports (already available):
  from sklearn.pipeline import make_pipeline
  from sklearn.compose import ColumnTransformer, make_column_selector
  from sklearn.preprocessing import OneHotEncoder, StandardScaler
  from sklearn.svm import LinearSVC, SVC, OneClassSVM
  from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
"""
from __future__ import annotations

import time  # noqa: F401  (for time.perf_counter)

import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401

from .data import CATEGORICAL  # noqa: F401  ["proto", "service", "state"]
from .metrics import binary_metrics, per_category_detection  # noqa: F401


def build_preprocessor(cfg: dict):
    """Return an UNFITTED transformer that turns the raw UNSW-NB15 columns into features.

    It must:
      * one-hot encode the CATEGORICAL columns, grouping values seen fewer than
        cfg["data"]["min_category_count"] times in the training data into one "other"
        column (hint: OneHotEncoder's min_frequency and handle_unknown options);
      * standardize all numeric columns (and may standardize the one-hot columns too);
      * keep feature names available, so Task 1 can report which features carry weight
        (a fitted ColumnTransformer has get_feature_names_out()).
    """
    raise NotImplementedError("TODO preprocessing: build_preprocessor in src/svm_tasks.py")


def task1_linear(X_train, y_train, X_test, y_test, cfg: dict, make_preprocessor) -> dict:
    """Task 1: linear SVM on the FULL training set.

    For every C in cfg["linear"]["C"], train make_pipeline(make_preprocessor(), LinearSVC(...))
    and record training and test metrics. Choose the best C using training-only evidence
    (for example 5-fold cross-validation F1), refit it, and report its weights.

    Return:
      {"C": chosen C,
       "precision", "recall", "f1", "fpr": test metrics at the chosen C,
       "train_seconds": seconds for the final fit at the chosen C,
       "sweep": [{"C": C, "train": binary_metrics(...), "test": binary_metrics(...),
                  "cv_f1": optional}, ...one per C...],
       "top_features": [{"feature": name, "weight": w}, ...the 10 largest |w|, largest first]}
    """
    raise NotImplementedError("TODO Task 1: task1_linear in src/svm_tasks.py")


def task2_rbf(X_sub, y_sub, X_test, y_test, cfg: dict, make_preprocessor) -> dict:
    """Task 2: RBF-kernel SVM on the 20,000-row training subsample.

    Grid-search SVC(kernel="rbf") over cfg["rbf"]["C"] x cfg["rbf"]["gamma"] with
    cfg["rbf"]["cv_folds"]-fold stratified CV, scoring "f1", with return_train_score=True
    and n_jobs=cfg["n_jobs"]. Refit the best setting on the whole subsample (time it) and
    evaluate on the testing set.

    Return:
      {"C": best C, "gamma": best gamma,
       "precision", "recall", "f1", "fpr": test metrics of the refit model,
       "n_support_vectors": total support vectors of the refit model,
       "train_seconds": seconds for the refit,
       "grid": [{"C": C, "gamma": g, "train_f1": mean train F1, "cv_f1": mean CV F1},
                ...all 16 settings...]}
    """
    raise NotImplementedError("TODO Task 2: task2_rbf in src/svm_tasks.py")


def task3_one_class(X_sub, y_sub, X_test, y_test, cat_test, cfg: dict, make_preprocessor) -> dict:
    """Task 3: one-class SVM trained on the NORMAL rows of the subsample only.

    For every nu in cfg["one_class"]["nu"], fit make_pipeline(make_preprocessor(),
    OneClassSVM(kernel="rbf", nu=nu)) on the rows with y_sub == 0. OneClassSVM.predict
    returns +1 for "inside" (normal) and -1 for "outside": convert -1 to 1 (attack) and
    +1 to 0 (normal) before computing metrics.

    Return:
      {"nu": cfg["one_class"]["chosen_nu"],
       "precision", "recall", "f1", "fpr": test metrics of the chosen run,
       "n_support_vectors": support vectors of the chosen run,
       "runs": [{"nu": nu, "detection_rate": recall on the testing set, "fpr": ...,
                 "per_category": per_category_detection(pred, cat_test)}, ...one per nu...]}
    """
    raise NotImplementedError("TODO Task 3: task3_one_class in src/svm_tasks.py")
