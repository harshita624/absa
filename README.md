# Adaptive Depth for Aspect-Level Sentiment

PyTorch research scaffold for aspect-level valence-arousal regression with a shared sentence encoding, layer-specific Gaussian readouts, and a learned halting controller. This repository is designed for Windows, macOS, and Linux; paths use pathlib and scripts are invoked with Python.

## Data

Download the English **Track A, Subtask 1 (DimASR)** data from the official [DimABSA task repository](https://github.com/DimABSA/DimABSA2026/tree/main/task-dataset/track_a/subtask_1/eng). For both English domains, put these four files under data/:

~~~text
eng_laptop_train_alltasks.jsonl
eng_laptop_dev_task1.jsonl
eng_restaurant_train_alltasks.jsonl
eng_restaurant_dev_task1.jsonl
~~~

The task repository also has test files, but this project does not need them: use the labeled dev files for evaluation, and reserve 10% of training sentences for validation. Do not use test files for model selection. The parser supports the official dev field Aspect_VA and extracts Aspect/VA pairs from the training Triplet or Quadruplet annotations. Run:

~~~text
python scripts/inspect_data.py
~~~

The parser accepts common aliases for sentence text, IDs, aspects, valence/arousal pairs, and VA strings. It ignores records without parseable labels, preserves NULL aspects (their query falls back to [CLS]), and falls back to [CLS] when an aspect is absent from the text. The collator uses the first case-insensitive match and tokenizer offsets.

The trainer uses an official development file when a filename contains dev or valid. If no such file exists, it takes a deterministic 10% sentence-level holdout from the training records using the run seed. The split is recorded in each run JSON. Hidden test labels are never assumed. The current run script will fail clearly if no training JSON/JSONL exists.

Synthetic smoke data:

~~~text
python scripts/make_synthetic_data.py
python scripts/inspect_data.py
~~~

Synthetic runs write to results_smoke/ only. They are deliberately small and must not be cited as research results.

## Setup and smoke run

Create a virtual environment, then install requirements.txt. For example:

~~~text
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/run_experiments.py --quick
python -m pytest
~~~

The quick run makes synthetic data if needed and uses a single short training epoch. A pretrained model must be available from the Hugging Face cache or downloadable when the run starts. Device selection is CUDA, then Apple MPS, then CPU. --amp enables CUDA AMP.

## Training and evaluation

Default run (three seeds):

~~~text
python scripts/run_experiments.py
~~~

Single run:

~~~text
python -m absa_depth.train --root . --seed 42 --epochs 3 --alpha 0.3 --beta 0.003 --tau 0.5
~~~

CPU pilot on a random 512-sentence training subset (one seed, one epoch; diagnostic only, not a main result):

~~~text
python -m absa_depth.train --root . --seed 42 --epochs 1 --max_train_sentences 512
~~~

The trainer prints device, dataset sizes, and batch progress. Set max_train_sentences to 0 (the default) to use all training sentences.

Two-stage fallback:

~~~text
python -m absa_depth.train --root . --seed 42 --two_stage
~~~

The model trains with normalized targets (y-5)/4; reported RMSE and Pearson metrics are calculated after converting back to the original 1-9 scale. A real-data checkpoint and adjacent JSON record are written under checkpoints/; quick synthetic checkpoints and run CSVs are written under results_smoke/ so they cannot replace real-data artifacts. Regular run CSVs are written under results/. The JSON includes configuration, available Git commit, epoch validation metrics, split choice and final metrics.

## Experiment commands

Quick synthetic smoke run (never mixed into real results):

~~~text
python scripts/run_experiments.py --quick
python -m pytest -q
~~~

Core real-data suite for one seed:

~~~text
python scripts/run_experiments.py --suite core --seeds 42
~~~

Reduced-data three-seed CPU pilot (512 training sentences, 2 epochs for OURS, 1 epoch per core baseline):

~~~text
python scripts/run_fast_pilot.py
~~~

This is a faster preliminary comparison, not the full-data experiment. It copies the official data into fast_pilot/ and keeps its checkpoints, CSVs, and figures separate from results/.

Full suite (three seeds by default; includes pair-input, B2 last-layer, deep supervision, learned gate, no-variance and no-deep-supervision ablations, and beta retraining sweep):

~~~text
python scripts/run_experiments.py --suite full
~~~

Each trained checkpoint is followed by batched evaluation sweeps for learned-gate tau, fixed depth and variance threshold. For the learned gate, the evaluator also measures true batch-one BERT latency with ten warmup sentences excluded. RQ3 subgroup tables, analytic cost estimates, aggregate mean/std CSVs and figures are written only from completed run files. Pair-input B1 measures batch-one latency separately.

The orchestrator defaults to CPU-compatible settings, but all full runs are expensive on CPU. For a small real-data diagnostic only:

~~~text
python -m absa_depth.train --root . --seed 42 --epochs 1 --max_train_sentences 512
~~~

Do not treat subset runs as primary results. Add --two_stage when the joint gate collapses. The two-stage phase first trains encoder/readouts/heads with deep supervision, then freezes them and trains only the gate.

The trainer prints device, dataset sizes and every tenth training batch. It chooses checkpoints on held-out training RMSE_VA at the last layer. Official dev is used for final evaluation, never for checkpoint selection. Synthetic data is excluded from real runs and real-data aggregation.

## Cost and metrics

Targets are normalized as (y-5)/4 for training and metrics are computed after converting predictions back to the 1-9 scale. RMSE_VA is sqrt(mean((Vhat-V)^2+(Ahat-A)^2)).

The analytic estimate is approximately D*(12*S*d^2+2*S^2*d) + A*E*(16*d^2+2*S*d) multiply-adds per sentence: D is sentence encoder depth, A is aspects per sentence, E is average aspect depth, S is token count and d is hidden width. It excludes embedding lookup, normalization, nonlinearities, optimizer work and memory movement. Measured latency is reported separately.

RQ3 uses aspect-level masks. Negation words are not, no, never, neither, nor, without, hardly and n't. Contrast markers are but, however, although, though, yet and while. Multi-aspect means at least two annotated aspects in the sentence; extreme valence means absolute gold valence minus 5 is at least 3.

The supplied literature deck is background only. Its novelty statement remains provisional pending a complete literature review.

## Execution record

On 2026-11-17, eight unit tests passed in the supplied Anaconda environment. Synthetic quick runs exercised Ours, B1 pair-input, B2 last-layer, deep-supervised, and no-variance code paths; these runs are under results_smoke/ and are not research results.

A real-data two-stage pilot was run on seed 42 using a random 512-sentence training subset for four epochs. It evaluated on the official English laptop and restaurant dev files (400 sentences). Its saved dev row reports RMSE_V=1.141, RMSE_A=1.039, RMSE_VA=1.543, average aspect depth 11.852/12 and average sentence depth 11.925/12. Batch-one true-early-exit latency on CPU was median 126.3 ms and p95 205.0 ms (400 dev sentences, ten warmups excluded). It is a one-seed subset pilot, not the final full-data result.

The pilot sweep and RQ3 CSVs and two figures are present under results/ and figures/. On 2026-11-18, the primary two-stage model was trained for three epochs on all 5,724 training sentences (seed 42), selected on the 636-sentence training holdout, and evaluated on the 400 official dev sentences. Its official-dev last-layer RMSE_VA was 1.1501. At the preselected tau=0.5, RMSE_VA was 1.1366 with 9.16 average aspect layers (9.23 average sentence layers); batch-one CPU latency was 79.69 ms median and 116.55 ms p95 (10 warmup sentences excluded). The validation last-layer RMSE_VA was 1.2789. The RQ3 subgroup table and exit-policy sweep for this full-data single-seed run are in results/rq3_ours_seed42.csv and results/sweeps_ours_seed42.csv. These are preliminary single-seed findings; threshold selection must not be tuned on the dev set.

The required three-seed mean±std, full-data baseline comparisons (including B1/B2 and the deep-supervised checkpoint for B3), full beta retraining sweep, and final paper conclusions have not been run. The single-seed two-stage run does not establish statistical significance or prove that the method beats baselines. The B3 fixed-layer curve is available only for checkpoints trained with deep supervision; do not present OURS's fixed-layer predictions as that baseline. Do not cite synthetic numbers or treat the subset pilot as evidence for full-data research questions.

Checkpoint note (2026-11-18): after the full-data results were evaluated, the first `--quick` smoke run wrote its synthetic checkpoint to `checkpoints/ours_seed42.pt`, replacing the full-data weights. The synthetic checkpoint was moved to `results_smoke/checkpoints/`, and the trainer was fixed so future quick runs save there. The completed full-data metrics, sweeps, subgroup table, and latency JSON remain available, but the full-data checkpoint itself must be retrained if needed for inference or reproduction.
