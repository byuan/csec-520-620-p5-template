"""Project 5 entry point: python -m src.p5 --config config.yaml   (what `make reproduce` runs)

PROVIDED: you do not need to change this file. It loads the data, draws the seeded
subsample, calls your four functions in src/svm_tasks.py, and writes
results/metrics.json plus the three figures. Tasks you have not written yet are
skipped and listed at the end; the run then exits with an error, so `make reproduce`
only succeeds once every task is done.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

from . import plots, svm_tasks
from .data import load, stratified_subsample


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Project 5: SVMs for intrusion detection")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--synthetic", action="store_true",
                    help="dry run on small UNSW-shaped fake data (overrides data.source)")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.synthetic:
        cfg["data"]["source"] = "synthetic"
    out = Path(cfg["output"]["dir"]); out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    d = load(cfg)
    X_sub, y_sub, _ = stratified_subsample(d["X_train"], d["y_train"], d["cat_train"],
                                           cfg["data"]["subsample_rows"], cfg["seed"])
    print(f"train {len(d['y_train'])} rows, test {len(d['y_test'])} rows, "
          f"subsample {len(y_sub)} rows (attack share {y_sub.mean():.2f})")

    def make_preprocessor():
        return svm_tasks.build_preprocessor(cfg)

    results = {"project": "p5",
               "data": {"source": cfg["data"].get("source", "unsw"),
                        "train_rows": int(len(d["y_train"])), "test_rows": int(len(d["y_test"])),
                        "subsample_rows": int(len(y_sub)), "seed": int(cfg["seed"])}}
    todo = []
    try:
        pre = make_preprocessor().fit(d["X_train"])
        results["data"]["n_features"] = int(pre.transform(d["X_train"].head(5)).shape[1])
    except NotImplementedError as e:
        todo.append(str(e))
        print(f"SKIPPED: {e}")
        return finish(results, out, todo)

    tasks = [
        ("linear", lambda: svm_tasks.task1_linear(d["X_train"], d["y_train"], d["X_test"], d["y_test"],
                                                  cfg, make_preprocessor), plots.plot_c_sweep, "linear_c_sweep.png"),
        ("rbf", lambda: svm_tasks.task2_rbf(X_sub, y_sub, d["X_test"], d["y_test"], cfg, make_preprocessor),
         plots.plot_rbf_heatmap, "rbf_heatmap.png"),
        ("one_class", lambda: svm_tasks.task3_one_class(X_sub, y_sub, d["X_test"], d["y_test"], d["cat_test"],
                                                        cfg, make_preprocessor),
         plots.plot_per_category, "one_class_per_category.png"),
    ]
    for key, fn, plot, fig in tasks:
        t = time.perf_counter()
        try:
            results[key] = fn()
        except NotImplementedError as e:
            todo.append(str(e)); print(f"SKIPPED: {e}"); continue
        plot(results[key], out / fig)
        print(f"{key:9s} done in {time.perf_counter() - t:6.1f}s  test f1 = {results[key].get('f1', float('nan')):.3f}"
              f"  fpr = {results[key].get('fpr', float('nan')):.3f}")
    print(f"total {time.perf_counter() - t0:.1f}s")
    return finish(results, out, todo)


def finish(results: dict, out: Path, todo: list) -> int:
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {out / 'metrics.json'}")
    if todo:
        print("\nNot done yet:\n  " + "\n  ".join(todo))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
