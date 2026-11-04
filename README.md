# IHND

Invariant Hypergraph Neural Diffusion for cross-site major depressive disorder identification from resting-state fMRI.

This repository implements the MTD hypergraph construction and IHND equations in *Cross-Site Major Depressive Disorder Identification from Resting-State fMRI via Invariant Dynamic Hypergraph Neural Diffusion*. It contains IHND, its training objective, and data preparation, training, and prediction commands. Participant data and trained checkpoints are not distributed.

## Installation

Use Python 3.10 or later and install a PyTorch build compatible with your CUDA installation. From the repository directory:

```bash
python -m pip install -e .
python -m ihnd --help
```

The native MTD backend uses the CUDA driver and NVRTC supplied with a CUDA-enabled PyTorch installation or CUDA toolkit. It compiles the included CUDA kernel for the selected GPU. The torch backend supports CPU and CUDA without this compiler. The automatic backend reports a warning before falling back to torch.

## Input data

Start from preprocessed ROI time series with shape `timepoints x 90`. MAT files use the `Data` variable by default. NPY files contain the matrix directly; NPZ files use the `timeseries` key. ROI ordering must be identical across participants. The repository starts after image preprocessing and ROI extraction.

Create a CSV manifest containing these columns:

| Column | Meaning |
| --- | --- |
| `subject_id` | Unique participant identifier |
| `site` | Acquisition-site identifier |
| `label` | `0` for NC, `1` for MDD |
| `timeseries_path` | File path, absolute or relative to the manifest |

The input manifest is supplied by the dataset owner. No diagnosis is inferred from filenames.

## Construct MTD hypergraphs

```bash
ihnd prepare --manifest data/participants.csv --output prepared --config configs/loso.json --device cuda
```

Construction follows Eqs. (1)--(4): normalized temporal derivatives, standardized products, sign-consistent interactions, and mean absolute interaction strength. All 4,005 two-ROI and 117,480 three-ROI combinations are evaluated. Each ROI sequence is independently shuffled in every temporal permutation, and product normalization is recomputed after shuffling. The same shuffled participant is used across candidate hyperedges within an iteration. Defaults are 10,000 permutations and uncorrected `p < 0.05`.

The upper-tail empirical probability is `(1 + exceedances) / (permutations + 1)`, where ties count as exceedances. Temporal standard deviations use `ddof=0`. Each subject's retained hyperedges, strengths, node features, probabilities, and exceedance counts are stored in an NPZ file. Graphs are constructed without diagnosis or site information. Zero-variance or nonfinite ROI signals raise an error instead of receiving replacement connections.

To select a backend explicitly:

```bash
ihnd prepare --manifest data/participants.csv --output prepared --device cuda --backend native
ihnd prepare --manifest data/participants.csv --output prepared_cpu --device cpu --backend torch
```

Construction records include configuration, input hashes, retained cardinalities, and elapsed computation time. Existing files are reused only if their input hash and construction configuration match. Timing excludes image preprocessing, file serialization, and one-time backend initialization.

## Train and evaluate

For leave-one-site-out evaluation:

```bash
ihnd train --manifest prepared/manifest.csv --output runs/loso --config configs/loso.json --device cuda
```

Each site becomes a held-out test domain. A stratified 12.5% validation partition is taken from the remaining participants. Test participants never enter training or checkpoint selection. Use `--fold` with a site identifier to run one fold.

For stratified five-fold evaluation:

```bash
ihnd train --manifest prepared/manifest.csv --output runs/fivefold --config configs/loso.json --protocol fivefold --device cuda
```

Five-fold evaluation splits participants by diagnosis and allows the same acquisition sites across partitions. It is distinct from LOSO evaluation. In both protocols, the validation partition is taken only from the source training pool.

The manuscript defaults are:

| Setting | Value |
| --- | --- |
| Hidden dimension | 64 |
| Diffusion layers | 2 |
| Diffusion step size | 1 |
| Shortcut interventions | 4 |
| Shortcut loss coefficient | 1 |
| Sparsity coefficient | 0.05 |
| Risk variance coefficient | 1 |
| Optimizer | Adam |
| Learning rate | 0.01 |
| Weight decay | 0.001 |
| Batch size | 32 |
| Maximum epochs | 150 |
| Checkpoint selection | Highest validation ACC |

All 150 epochs are run by default. Validation loss breaks ties in validation ACC. Predictions use only the invariant component. Binary F1, SEN, SPE, and AUC treat MDD as the positive class. Fold metrics are stored as fractions. Summary means and standard deviations are reported as percentages with five decimal places; folds have equal weight and standard deviation uses `ddof=1`. Undefined metrics remain empty and their fold identifiers are reported.

Each run produces:

```text
runs/loso/
    run_config.json
    environment.json
    fold_metrics.csv
    predictions.csv
    summary.json
    SITE_ID/
        split.csv
        feature_alignment.json
        history.csv
        best.pt
        test_predictions.csv
        metrics.json
```

Use a new run directory for each experiment. Existing checkpoints are not overwritten.

## Predict

```bash
ihnd predict --checkpoint runs/loso/SITE_ID/best.pt --manifest prepared/manifest.csv --output runs/predictions.csv --device cuda
```

Supply a manifest for the intended evaluation participants. Predicting all participants with a single site-specific checkpoint does not produce LOSO predictions. The training command exports the correctly held-out predictions for each fold.

## Implementation details

The sparse diffusion operator implements `Z - tau * G.T * A(Z) * G * Z`. Node degrees are sums of incident hyperedge weights. Gradients use degree-normalized node representations, and attention is normalized over the hyperedges incident to each node. The adjoint subtracts the mean flux within each hyperedge. No additional hyperedge-weight factor is inserted into Eq. (8). Zero-degree nodes have a zero gradient contribution.

The same encoder is used for the full, invariant, and shortcut components. Hyperedge membership probabilities depend on the mean encoded ROI representation and the MTD strength. Training uses continuous Gumbel--Sigmoid masks. Inference uses deterministic probabilities. Cross-site donors are drawn from a different site when one is available in the current batch; otherwise another participant is used, or the same participant for a one-person batch. The intervention gate detaches the shortcut logits. Shortcut classification detaches the shortcut representation. The sparsity term averages probabilities within each subject before averaging subjects.

They are explicit here: two-linear-layer MLPs, LeakyReLU slope 0.2 for attention, Gumbel temperature 1, continuous inference masks, seed 42, and population variance across intervention risks.

## Verification

Tests use an existing graph constructed from a real participant. Set `IHND_TEST_GRAPH` to its NPZ path, and optionally set `IHND_TEST_SUBJECT`, `IHND_TEST_SITE`, and `IHND_TEST_LABEL` to the corresponding metadata. Then run:

```bash
python -m unittest discover -s tests -v
```

The checks cover the explicit diffusion operator, gradient boundaries, invariant-only prediction, and CPU/GPU interaction strengths. Tests requiring participant data are skipped when no graph path is provided.

## Repository contents

```text
ihnd/
    data.py
    model.py
    objective.py
    mtd.py
    native.py
    strength.cu
    training.py
    cli.py
configs/
tests/
    test_ihnd.py
```
