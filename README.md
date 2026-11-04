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

`soundfile` is pinned to 0.12.1 because newer releases reject part of the 2021 FLACs. `load_dataset.py` clips the few decoded spikes into `int16`.

Build the waveform cache once. Audio is resampled to 16 kHz, fixed at 4 s (64,000 samples), and stored as `int16` memmaps in `datasets/waveforms/`. Splits are read with `mmap_mode="r"`.

```bash
python load_dataset.py
```

## Reproduce

Defaults come from `deepfake/constants.py`: poolings `average`, `max`, and `wavelet`; families `db4`, `db8`, `db16`, `db24`, `db38`; level `WAVELET_LEVEL` (currently 3, so 8 nodes). That is 42 models per seed.

Run the three scripts in this order, with the same seeds. Training already writes the 2019 scores, so the second script only fills in what is missing (normally ASVspoof 2021). An interrupted training run skips combinations already listed in `results.csv`.

```bash
python train_models.py --seeds 42 52 62 72 82
python evaluate_and_save_scores.py --seeds 42 52 62 72 82
python model_performance_analysis.py --seeds 42 52 62 72 82
```

| Script | Role | Output |
| --- | --- | --- |
| `train_models.py` | Train on 2019, keep the lowest dev EER, score 2019 eval | `model.pt`, `history.csv`, `scores_eval_2019.npy`, one `ASVspoof 2019` row |
| `evaluate_and_save_scores.py` | Reload `model.pt` and score any split whose `.npy` is missing | `scores_eval_2021.npy`, one `ASVspoof 2021` row |
| `model_performance_analysis.py` | EER per attack, and per codec on 2021 | `attacks_score_2019.csv`, `attacks_score_2021.csv`, `codecs_score_2021.csv` |

The three scripts share `--seeds`, `--poolings`, `--wavelets`, `--nodes`, and `--results-dir` (default `results/train_all`). The level is not a CLI flag; change `WAVELET_LEVEL` in `deepfake/constants.py`.

```bash
python train_models.py --seeds 42 --poolings wavelet --wavelets db4 --nodes aaa
```

Results land in `results/train_all/results.csv` and in one folder per run:

```text
results/train_all/runs/average/seed_<seed>/
results/train_all/runs/max/seed_<seed>/
results/train_all/runs/wavelet/<family>/<node>/seed_<seed>/
```

`results.csv` columns: `seed`, `pooling`, `wavelet`, `node`, `eval_split`, `eer`, `roc_auc`, `best_epoch`, `best_dev_eer`, `train_time_s`. Max and average leave `wavelet` and `node` empty. Because of that, `train_models.py` may retrain a finished max or average run: empty fields are read as `NaN` and no longer match the run key. The evaluation script does not have this problem.

## Layout

```text
load_dataset.py                  # FLAC -> datasets/waveforms/
train_models.py                  # train and score ASVspoof 2019
evaluate_and_save_scores.py      # score saved checkpoints on 2021
model_performance_analysis.py    # EER per attack and codec
deepfake/
  constants.py                   # families, level, poolings, epochs, batch size
  data.py                        # splits, node order, seeds
  features.py                    # wavelet-packet node and dataset
  cnn.py                         # CustomCNN and the three poolings
  training.py                    # loops, early stopping, class weights
  metrics.py                     # EER and LLR score
paper_plots.ipynb                # figures, after the three scripts
statistical_analysis.ipynb       # comparison across seeds
```

`comparing_results_single_seed.ipynb` and `tsne_analysis.ipynb` are extra notebooks. They also read `results/train_all` and the waveform cache.

## Model

Input is the raw waveform in `[-1, 1)`, with no hand-crafted features. `CustomCNN` applies two linear convolutions (Conv1d + BatchNorm, no ReLU), one pooling with factor \(2^L\), three convolutional blocks with ReLU, mean and standard deviation over time, and a two-class head.

- **Max / average:** `MaxPool1d` or `AvgPool1d` with `kernel_size=2**L` and `ceil_mode=True`.
- **Wavelet:** one node of a level-\(L\) packet (`get_node` in `deepfake/features.py`). A path such as `ada` is low-pass, high-pass, low-pass, each with stride 2. Nodes follow frequency order from `frequency_ordered_nodes`.

Training uses class-weighted cross-entropy, AdamW (`lr=1e-4`, `weight_decay=1e-4`), `ReduceLROnPlateau` on dev EER, batch size 32, at most 30 epochs, and early stopping with patience 5. The score is `logit[bonafide] − logit[spoof]`.
