# Evaluating Wavelet Pooling over Conventional Pooling in CNNs for Audio Deepfake Detection

João Vitor Ramos Assalim  
Department of Computing, São Paulo State University (UNESP), Bauru, Brazil  
joao.assalim@unesp.br

A 1D CNN on raw audio, with one pooling layer that is either max, average, or a single node of a wavelet packet. The backbone does not change. Models are trained on ASVspoof 2019 LA, selected by development EER, scored on the 2019 evaluation set during training, and then scored on ASVspoof 2021 LA.

The reported runs use the seeds **42, 52, 62, 72, and 82**.

## Setup

Python 3.13. The requirements pin PyTorch for CUDA 13.0 (RTX 5060 Ti).

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Put the corpora one directory above this repository:

- ASVspoof 2019 LA at `../Data/LA/LA/` ([DS_10283_3336](https://datashare.ed.ac.uk/handle/10283/3336)).
- ASVspoof 2021 LA eval at `../Data/ASVspoof2021_LA_eval/ASVspoof2021_LA_eval/` (`flac/` from [Zenodo 4837263](https://zenodo.org/records/4837263)). Copy `keys/LA/CM/trial_metadata.txt` from `LA-keys-full.tar.gz` into that same folder. The labels are not in the Zenodo archive.
- For the t-DCF on 2021, copy two more files from the same keys archive into that folder: `keys/LA/ASV/trial_metadata.txt` saved as `trial_metadata_tdcf.txt`, and `keys/LA/ASV/ASVTorch_Kaldi/score.txt` as `score.txt`. The 2019 ASV scores already ship in `../Data/LA/LA/ASVspoof2019_LA_asv_scores/`.

`soundfile` is pinned to 0.12.1 because newer releases reject part of the 2021 FLACs. `load_dataset.py` clips the few decoded spikes into `int16`.

Build the waveform cache once. Audio is resampled to 16 kHz, fixed at 4 s (64,000 samples), and stored as `int16` memmaps in `datasets/waveforms/`. Splits are read with `mmap_mode="r"`.

```bash
python load_dataset.py
```

## Reproduce

Defaults come from `deepfake/constants.py`: poolings `average`, `max`, and `wavelet`; families `db4`, `db8`, `db16`, `db24`, `db38`; level `WAVELET_LEVEL` (currently 3, so 8 nodes). That is 42 models per seed.

Run the four scripts in this order, with the same seeds. Training already writes the 2019 scores, so the second script only fills in what is missing (normally ASVspoof 2021). An interrupted training run skips combinations already listed in `results.csv`.

```bash
python train_models.py --seeds 42 52 62 72 82
python evaluate_and_save_scores.py --seeds 42 52 62 72 82
python model_performance_analysis.py --seeds 42 52 62 72 82
python compute_tdcf.py
```

| Script | Role | Output |
| --- | --- | --- |
| `train_models.py` | Train on 2019, keep the lowest dev EER, score 2019 eval | `model.pt`, `history.csv`, `scores_eval_2019.npy`, one `ASVspoof 2019` row |
| `evaluate_and_save_scores.py` | Reload `model.pt` and score any split whose `.npy` is missing | `scores_eval_2021.npy`, one `ASVspoof 2021` row |
| `model_performance_analysis.py` | EER per attack, and per codec on 2021 | `attacks_score_2019.csv`, `attacks_score_2021.csv`, `codecs_score_2021.csv` |
| `compute_tdcf.py` | min t-DCF of every row, from the saved scores and the official ASV scores | `min_tdcf` column in `results.csv` |

The first three scripts share `--seeds`, `--poolings`, `--wavelets`, `--nodes`, and `--results-dir` (default `results/train_all`). `compute_tdcf.py` takes only `--results-dir` and `--results-file` (default `results.csv`), and it recomputes every row of that file without loading audio (a few seconds). The level is not a CLI flag; change `WAVELET_LEVEL` in `deepfake/constants.py`.

The min t-DCF follows the official evaluation packages:

- **Cost model:** `Pspoof=0.05`, `Cmiss=1`, `Cfa=10`, `Cfa_spoof=10`.
- **ASV threshold:** fixed at the ASV's own EER point.
- **2019:** the legacy t-DCF (Kinnunen et al., 2018).
- **2021:** the revised t-DCF (Kinnunen et al., 2020). It uses only the trials in the `eval` phase: 14,816 bonafide and 133,360 spoof. The `eer` column on 2021 covers all 181,566 trials, so the two 2021 metrics are measured on different trial sets.

`compute_tdcf.py` rewrites `results.csv`. Do not run it while a training script is still appending to the same file.

```bash
python train_models.py --seeds 42 --poolings wavelet --wavelets db4 --nodes aaa
```

Results land in `results/train_all/results.csv` and in one folder per run:

```text
results/train_all/runs/average/seed_<seed>/
results/train_all/runs/max/seed_<seed>/
results/train_all/runs/wavelet/<family>/<node>/seed_<seed>/
```

`results.csv` columns: `seed`, `pooling`, `wavelet`, `node`, `eval_split`, `eer`, `roc_auc`, `best_epoch`, `best_dev_eer`, `train_time_s`, and `min_tdcf` after `compute_tdcf.py`. Max and average leave `wavelet` and `node` empty. Because of that, `train_models.py` may retrain a finished max or average run: empty fields are read as `NaN` and no longer match the run key. The evaluation script does not have this problem.

## Layout

```text
load_dataset.py                  # FLAC -> datasets/waveforms/
train_models.py                  # train and score ASVspoof 2019
evaluate_and_save_scores.py      # score saved checkpoints on 2021
model_performance_analysis.py    # EER per attack and codec
compute_tdcf.py                  # min t-DCF column in results.csv
train_entropy_based_model.py     # test: per-sample node chosen by min or max entropy
wavelet_entropy_analysis.ipynb   # test: which nodes the entropy pooling picks
deepfake/
  constants.py                   # families, level, poolings, epochs, batch size
  data.py                        # splits, node order, seeds
  features.py                    # wavelet-packet nodes, subband entropy, dataset
  cnn.py                         # CustomCNN, the three poolings, entropy pooling
  training.py                    # loops, early stopping, class weights
  metrics.py                     # EER and LLR score
paper_plots.ipynb                # figures, after the four scripts
statistical_analysis.ipynb       # comparison across seeds
```

`comparing_results_single_seed.ipynb` and `tsne_analysis.ipynb` are extra notebooks. They also read `results/train_all` and the waveform cache.

## Model

Input is the raw waveform in `[-1, 1)`, with no hand-crafted features. `CustomCNN` applies two linear convolutions (Conv1d + BatchNorm, no ReLU), one pooling with factor \(2^L\), three convolutional blocks with ReLU, mean and standard deviation over time, and a two-class head.

- **Max / average:** `MaxPool1d` or `AvgPool1d` with `kernel_size=2**L` and `ceil_mode=True`.
- **Wavelet:** one node of a level-\(L\) packet (`get_node` in `deepfake/features.py`). A path such as `ada` is low-pass, high-pass, low-pass, each with stride 2. Nodes follow frequency order from `frequency_ordered_nodes`.

Training uses class-weighted cross-entropy, AdamW (`lr=1e-4`, `weight_decay=1e-4`), `ReduceLROnPlateau` on dev EER, batch size 32, at most 30 epochs, and early stopping with patience 5. The score is `logit[bonafide] − logit[spoof]`.

## Test: entropy-based node selection

This is an exploratory test, not part of the main comparison. It checks how a pooling that picks its wavelet node per sample behaves, and measures it against the fixed-node models.

`WaveletEntropyBasedCNN` (`deepfake/cnn.py`) uses the same backbone as `CustomCNN`. Only the pooling changes: `WaveletEntropyBasedPooling`.

- **Selection:** for each sample, it computes all \(2^L\) level-\(L\) nodes and the Shannon entropy of each one (`compute_subband_entropy` in `deepfake/features.py`). The entropy is taken over the normalized energy of that node's coefficients across channels and time. `--selection min` (default) keeps the node with the lowest entropy, and `--selection max` keeps the one with the highest.
- **Gradient:** the choice is made under `no_grad`. The gradient flows only through the chosen subband.
- **Counts:** the `subband_selection` buffer counts the chosen nodes during training only, in `frequency_ordered_nodes` order. It is saved in `model.pt` and holds the counts up to the best epoch.

```bash
python train_entropy_based_model.py --seeds 42 --wavelets db4 db8 db16 db24 db38 --selection min
python compute_tdcf.py --results-dir results/train_entropy_based

python train_entropy_based_model.py --seeds 42 --wavelets db4 db8 db16 db24 db38 --selection max
python compute_tdcf.py --results-dir results/train_entropy_based --results-file results_max_entropy.csv
```

The script trains one model per seed and family, then scores ASVspoof 2019 and 2021. Each mode has its own results file and run folders, so the two can be compared side by side:

| `--selection` | Results | Runs |
| --- | --- | --- |
| `min` | `results.csv` | `runs/<family>/seed_<seed>/` |
| `max` | `results_max_entropy.csv` | `runs_max_entropy/<family>/seed_<seed>/` |

A run with both splits already in its results file is skipped. If only the 2019 row exists, the model is trained again before scoring 2021, so the two splits then come from different models. Delete the 2019 row first in that case.

```text
results/train_entropy_based/results.csv          # seed, wavelet, eval_split, eer, roc_auc, best_epoch, best_dev_eer, train_time_s (+ min_tdcf)
results/train_entropy_based/runs/<family>/seed_<seed>/
  model.pt, history.csv
  scores_eval_2019.npy, scores_eval_2021.npy     # one score per file, in the meta order
  selection_eval_2019.npy, selection_eval_2021.npy  # chosen node index per file (uint8)
```

`results_max_entropy.csv` and `runs_max_entropy/` follow the same layout. `wavelet_entropy_analysis.ipynb` reads these files; point its `run_dir` at `runs/` or `runs_max_entropy/`. It shows:

- which nodes were chosen during training (`features.2.subband_selection` in `model.pt`);
- the share of each node on eval;
- crosstabs of node by class and node by attack.

Map an index to its node with `frequency_ordered_nodes(WAVELET_LEVEL)[index]`.

**Cost.** Every step computes the whole level-\(L\) packet. With db4 the step is about 40% slower than a fixed node, and the batch size does not change throughput. With seed 42, a run took about 1–4 h on an RTX 5060 Ti, training plus both evaluations, depending on the family and the number of epochs.

**What was seen.** On raw audio, the lowest entropy is mostly in `daa` (7–8 kHz, the emptiest band), for both classes and all attacks. The choice reflects how sparse a band is, not what separates bonafide from spoof. Compare the model with the fixed `daa` node and the best fixed node on the same family and seeds. If it matches the fixed `daa`, the per-sample selection adds nothing. With the maximum entropy, the choice is almost always `aaa` (0–1 kHz, which holds most of the energy), so that mode should behave close to the fixed `aaa` node.

## Citation

João Assalim, JoaoAssalim/wavelet-pooling-audio-deepfake: v1.0.0. (Nov. 04, 2026). Zenodo. [doi: 10.5281/zenodo.23950081](https://doi.org/10.5281/zenodo.23950081).

```bibtex
@software{assalim2026wavelet_pooling,
  author    = {Assalim, João},
  title     = {JoaoAssalim/wavelet-pooling-audio-deepfake: v1.0.0},
  year      = {2026},
  month     = nov,
  publisher = {Zenodo},
  version   = {v1.0.0},
  doi       = {10.5281/zenodo.23950081},
  url       = {https://doi.org/10.5281/zenodo.23950081}
}
```
