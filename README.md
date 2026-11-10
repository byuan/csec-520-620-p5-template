# CSEC 520/620 — Project 5 (Bonus): SVMs for Intrusion Detection

Starter repository for the optional Project 5 bonus project. You compare a linear SVM,
a tuned RBF-kernel SVM and a one-class SVM on UNSW-NB15, then recommend one for
deployment. The full task description and rubric are in the handout on myCourses.

It follows the course **reproducibility standard**:

> A grader can recreate your results from your repository in **one command**: `make reproduce`.

Requires Python 3.10+ and `make`. No GPU, no PyTorch.

---

## 1. Get your copy

1. Click **"Use this template" → Create a new repository**.
2. Name it `csec-520-620-p5-<yourid>` and set it to **Private**.
3. Add the instructor (`@byuan`): **Settings → Collaborators → Add people**.

## 2. Set up

```bash
git clone git@github.com:<you>/csec-520-620-p5-<yourid>.git
cd csec-520-620-p5-<yourid>
make setup          # creates .venv and installs requirements
make test           # the provided helpers' tests should pass
```

Then download the data (see `data/README.md`) into `data/`.

## 3. What is provided and what you write

| File | Status | What it does |
|---|---|---|
| `src/data.py` | provided | loads the official train/test CSVs, drops `id`, draws the seeded 20,000-row subsample; also makes synthetic data for a dry run |
| `src/metrics.py` | provided | precision, recall, F1, FPR, and detection rate per attack category |
| `src/plots.py` | provided | the C-sweep plot, the C × γ heatmap and the per-category bar chart |
| `src/p5.py` | provided | the entry point: calls your code, writes `results/metrics.json` and the figures |
| **`src/svm_tasks.py`** | **you write** | `build_preprocessor`, `task1_linear`, `task2_rbf`, `task3_one_class` (four TODOs) |
| `config.yaml` | you may edit | data paths, subsample size, the C / γ / ν grids, your chosen ν |
| `report/` | you write | your 2–3 page report |
| `SUBMISSION.md` | you fill in | manifest the grader reads first |

Each TODO's docstring says exactly what to return. Do not change the grids in
`config.yaml`; the grader checks for the values in the handout.

## 4. Work loop

```bash
make smoke          # dry run on synthetic data: checks your code runs end to end (seconds)
make reproduce      # full run on UNSW-NB15 (the RBF grid search takes about 5-20 minutes)
make check          # P5 grading checks on the results you have, without rerunning
```

`make reproduce` writes results for every task you have finished and lists the ones still
to do; it only succeeds once all four are written. On GitHub, CI runs the tests and a
synthetic dry run on every push: unfinished TODOs show as a warning, and a crash in your
code fails the run. `make smoke` results are marked
`"source": "synthetic"` and are not accepted for grading, so finish with `make reproduce`.

## 5. Submit

```bash
python grading/grade.py            # full check: reruns make reproduce, then validates
git add -A && git commit -m "p5 final" && git push
git tag p5-final && git push origin p5-final
```

Submit your repository URL, the `p5-final` tag and your report to the myCourses dropbox
by **Mon Nov 30, 11:59 PM**. No late submissions for the bonus.

## Grading

Graded by an AI agent plus instructor review against `grading/rubric_p5.yaml`:

| Component | Points |
|---|---|
| Reproducibility | 15 |
| Preprocessing & pipeline | 10 |
| Task 1: linear SVM | 15 |
| Task 2: RBF SVM & tuning | 25 |
| Task 3: one-class SVM | 20 |
| Task 4: analysis | 15 |

Your score out of 100 converts to up to **+3 percentage points** on the final course grade.
The `results/metrics.json` format and the automatic checks are described in
`grading/P5_METRICS.md`; the grader's protocol is in `grading/AGENT_GRADING.md`.

## Rules

- Never commit data or model files; `data/` and `results/` are covered by `.gitignore`.
- Fit preprocessing on training data only (inside a Pipeline). Never tune on the testing set.
- Per the syllabus AI-use policy, acknowledge substantive AI use in `SUBMISSION.md`.
