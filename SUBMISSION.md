# Submission Manifest — Project 5 (bonus): SVMs for intrusion detection

Fill this in before submitting. The grading agent reads it first.

## Student
- Name (RIT ID):

## Course level
- [ ] 520 (undergraduate)
- [ ] 620 (graduate)

## Dataset
- UNSW-NB15, official partition from the course Drive ZIP
  (`UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv`)
- Training subsample size and seed (from `config.yaml`):

## How to reproduce
```bash
make setup
# put the two CSVs in data/ (see data/README.md)
make reproduce
```
Runtime on your machine (the RBF grid search is the slow part):
Anything non-default the grader must know:

## Choices
- Chosen C for the linear SVM, and how you chose it (training-only evidence):
- Best (C, gamma) for the RBF SVM (cross-validation F1):
- Chosen nu for the one-class SVM, and why:

## Claimed results (must match `results/metrics.json`)
| Model | Precision | Recall | F1 | FPR |
|---|---|---|---|---|
| Linear SVM | | | | |
| RBF SVM | | | | |
| One-class SVM | | | | |

## Deployment recommendation (one sentence)

## AI-use acknowledgment
Per the syllabus policy, briefly note any substantive use of AI assistants.
