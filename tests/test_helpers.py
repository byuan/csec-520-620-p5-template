"""Unit tests for the provided helpers. They pass before you write any task code."""
import json

import numpy as np
import pytest

from src import p5
from src.data import ATTACK_CATEGORIES, CATEGORICAL, CATEGORY, LABEL, stratified_subsample, synthetic, _split_frame
from src.metrics import binary_metrics, per_category_detection


def test_binary_metrics_hand_example():
    y = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    p = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]          # TP 3, FN 1, FP 1, TN 5
    m = binary_metrics(y, p)
    assert m["precision"] == pytest.approx(0.75)
    assert m["recall"] == pytest.approx(0.75)
    assert m["f1"] == pytest.approx(0.75)
    assert m["fpr"] == pytest.approx(1 / 6)


def test_binary_metrics_no_positives_is_zero_not_error():
    m = binary_metrics([0, 0, 1], [0, 0, 0])
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0


def test_per_category_excludes_normal():
    cats = np.array(["Normal", "DoS", "DoS", "Worms"])
    out = per_category_detection([1, 1, 0, 1], cats)
    assert out == {"DoS": 0.5, "Worms": 1.0}


def test_synthetic_has_unsw_columns():
    train, test = synthetic(n_train=500, n_test=200)
    for df in (train, test):
        for c in CATEGORICAL + [LABEL, CATEGORY, "id"]:
            assert c in df.columns
        assert set(df[CATEGORY]) <= set(ATTACK_CATEGORIES) | {"Normal"}
    X, y, cat = _split_frame(train)
    assert "id" not in X.columns and LABEL not in X.columns and CATEGORY not in X.columns
    assert ((y == 0) == (cat == "Normal")).all()


def test_stratified_subsample_keeps_ratio_and_is_seeded():
    train, _ = synthetic(n_train=4000, n_test=10)
    X, y, cat = _split_frame(train)
    Xs, ys, cs = stratified_subsample(X, y, cat, 1000, seed=7)
    assert len(ys) == 1000 and len(Xs) == 1000 and len(cs) == 1000
    assert abs(ys.mean() - y.mean()) < 0.01
    Xs2, ys2, _ = stratified_subsample(X, y, cat, 1000, seed=7)
    assert (Xs.values == Xs2.values).all()


def test_runner_reports_unfinished_tasks(tmp_path, monkeypatch):
    """With the TODOs still in place, the runner writes partial results and exits 2."""
    from src import svm_tasks
    try:
        svm_tasks.build_preprocessor({"data": {"min_category_count": 50}})
    except NotImplementedError:
        pass
    else:
        pytest.skip("build_preprocessor is implemented; this test only covers the starter state")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"""project: p5
seed: 0
data: {{source: synthetic, subsample_rows: 500, min_category_count: 50}}
linear: {{C: [1], max_iter: 1000}}
rbf: {{C: [1], gamma: [0.1], cv_folds: 2}}
one_class: {{nu: [0.05], chosen_nu: 0.05}}
n_jobs: 1
output: {{dir: {tmp_path / 'results'}}}
""")
    assert p5.main(["--config", str(cfg)]) == p5.EXIT_TODO
    m = json.loads((tmp_path / "results" / "metrics.json").read_text())
    assert m["project"] == "p5" and m["data"]["subsample_rows"] == 500
