"""Load UNSW-NB15 (or a small synthetic stand-in) and draw the training subsample.

PROVIDED: you do not need to change this file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CATEGORICAL = ["proto", "service", "state"]
LABEL = "label"          # 0 = normal, 1 = attack  (the prediction target)
CATEGORY = "attack_cat"  # attack family; used only for the per-category analysis
DROP = ["id"]
ATTACK_CATEGORIES = ["Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers", "Generic",
                     "Reconnaissance", "Shellcode", "Worms"]

DOWNLOAD_HELP = (
    "UNSW-NB15 CSV not found: {path}\n"
    "Download the course dataset ZIP from Google Drive (the same download as Project 2):\n"
    "  https://drive.google.com/file/d/1jDsXYALnEsLzYYGCEtxgkAYygS-vKIt7/view?usp=sharing\n"
    "Extract UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv into data/.\n"
    "For a dry run without the data, set data.source: synthetic in config.yaml."
)


def _split_frame(df: pd.DataFrame):
    df = df.drop(columns=[c for c in DROP if c in df.columns])
    y = df[LABEL].astype(int).to_numpy()
    cat = df[CATEGORY].fillna("Normal").astype(str).str.strip().to_numpy()
    X = df.drop(columns=[LABEL, CATEGORY])
    for c in CATEGORICAL:
        X[c] = X[c].astype(str).str.strip().str.lower()
    return X, y, cat


def load(cfg: dict) -> dict:
    """Return the official train/test partition as DataFrames and label arrays."""
    d = cfg["data"]
    if d.get("source", "unsw") == "synthetic":
        train, test = synthetic(seed=cfg["seed"])
    else:
        frames = []
        for key in ("train_csv", "test_csv"):
            path = Path(d[key])
            if not path.exists():
                raise SystemExit(DOWNLOAD_HELP.format(path=path))
            frames.append(pd.read_csv(path, low_memory=False))
        train, test = frames
    X_train, y_train, cat_train = _split_frame(train)
    X_test, y_test, cat_test = _split_frame(test)
    return {"X_train": X_train, "y_train": y_train, "cat_train": cat_train,
            "X_test": X_test, "y_test": y_test, "cat_test": cat_test}


def stratified_subsample(X: pd.DataFrame, y: np.ndarray, cat: np.ndarray, n: int, seed: int):
    """A seeded random sample of n rows with the same normal/attack ratio as the full data."""
    if n >= len(y):
        return X.reset_index(drop=True), y, cat
    idx, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=seed)
    idx = np.sort(idx)
    return X.iloc[idx].reset_index(drop=True), y[idx], cat[idx]


def synthetic(n_train: int = 6000, n_test: int = 3000, seed: int = 0):
    """Small UNSW-shaped fake data so the pipeline can run before you download anything.

    The numbers mean nothing; use it only to check that your code runs end to end.
    """
    rng = np.random.default_rng(seed)

    def make(n):
        is_attack = rng.random(n) < 0.6
        cat = np.where(is_attack, rng.choice(ATTACK_CATEGORIES, n,
                                             p=[.02, .02, .1, .3, .2, .3, .04, .01, .01]), "Normal")
        shift = np.array([ATTACK_CATEGORIES.index(c) + 1 if c != "Normal" else 0 for c in cat])
        num = {
            "dur": rng.exponential(1.0, n) * (1 + 0.3 * shift),
            "spkts": rng.poisson(10 + 3 * shift),
            "dpkts": rng.poisson(12 + 1 * shift),
            "sbytes": rng.lognormal(7 + 0.15 * shift, 1.0),
            "dbytes": rng.lognormal(7.5 - 0.1 * shift, 1.0),
            "rate": rng.exponential(1000, n) * (1 + 0.5 * is_attack),
            "sttl": np.where(is_attack, rng.choice([62, 254], n, p=[.3, .7]), rng.choice([31, 62], n)),
            "dttl": np.where(is_attack, rng.choice([0, 252], n), rng.choice([29, 60], n)),
            "ct_srv_src": rng.poisson(3 + shift),
            "ct_state_ttl": rng.integers(0, 3, n) + 2 * is_attack,
        }
        df = pd.DataFrame(num)
        df.insert(0, "id", np.arange(1, n + 1))
        df["proto"] = rng.choice(["tcp", "udp", "unas", "arp", "ospf", "sctp"], n, p=[.5, .3, .1, .05, .04, .01])
        df["service"] = rng.choice(["-", "http", "dns", "ftp", "smtp", "irc"], n, p=[.5, .2, .15, .08, .06, .01])
        df["state"] = np.where(is_attack, rng.choice(["INT", "FIN", "CON"], n, p=[.6, .3, .1]),
                               rng.choice(["FIN", "CON", "REQ"], n, p=[.6, .3, .1]))
        df[CATEGORY] = cat
        df[LABEL] = is_attack.astype(int)
        return df

    return make(n_train), make(n_test)
