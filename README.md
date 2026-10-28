# Evaluating Wavelet Pooling over Conventional Pooling in CNNs for Audio Deepfake Detection

João Vitor Ramos Assalim  
Department of Computing, São Paulo State University (UNESP), Bauru, Brazil  
joao.assalim@unesp.br

This repository compares wavelet-packet pooling with max and average pooling in a 1D CNN that works directly on raw audio. Models are trained on the ASVspoof 2019 Logical Access (LA) partition and evaluated on the ASVspoof 2019 LA evaluation set. The saved checkpoints are then evaluated on ASVspoof 2021 LA in a separate pass (`evaluate_out_datasets.py`).

The backbone is identical for every run; only the pooling layer changes. Wavelet pooling keeps a single node of a wavelet packet decomposition, so sweeping all nodes of one level shows how much each frequency subband contributes to detection.

## Method

### Input

Audio is resampled to 16 kHz and fixed at 4 s (64,000 samples). Shorter recordings are zero-padded on both sides (`librosa.util.pad_center`), and longer ones are truncated. Waveforms are stored as `int16`, which is lossless because the source FLACs are 16-bit, and scaled back to `[-1, 1)` (`/ 32768`) when a batch is loaded. No hand-crafted features are extracted.

### Architecture

`deepfake/cnn.py` — `CustomCNN(wavelet_family, level, node, pooling)`:

| Stage | Layers | Channels | Length |
| --- | --- | --- | --- |
| Pre-pooling | 2 × (Conv1d k=3 + BatchNorm), **no ReLU** | 1 → 16 → 16 | 64,000 |
| Pooling | max / average / wavelet, factor \(2^L\) | 16 | ≈ 64,000 / \(2^L\) |
| Block 1 | 2 × (Conv1d k=3 + BatchNorm + ReLU) | 16 → 64 → 64 | |
| Block 2 | 2 × (Conv1d k=3 + BatchNorm + ReLU) | 64 → 128 → 128 | |
| Block 3 | 2 × (Conv1d k=3 + BatchNorm + ReLU) + `Dropout1d(0.1)` | 128 → 256 → 256 | |
| Statistics | mean and standard deviation over time | 512 | |
| Classifier | Linear 128 + ReLU + Dropout 0.4 + Linear 2 | | |

The convolutions before the pooling are linear, so the pooled signal is a true subband of a linear filtering of the input. ReLU is applied only after the pooling.

### Pooling layers

There is a single pooling, and its downsampling factor \(2^L\) is the same for all three methods:

- **Max:** `MaxPool1d(kernel_size=2**L, ceil_mode=True)`.
- **Average:** `AvgPool1d(kernel_size=2**L, ceil_mode=True)`.
- **Wavelet:** `WaveletPooling` returns one node of a level-\(L\) wavelet packet decomposition (`deepfake/features.py: get_node`). For a node path such as `"adad"`, each character is one analysis step: reflect padding, then a depthwise `conv1d` with the low-pass (`a`) or high-pass (`d`) filter and stride 2. Only the filters along the path are computed, rather than the full tree. The output equals `ptwt.WaveletPacket(...)[node]` (relative error ~1e-16), and the original `ptwt` version is kept commented in the same file.

Nodes come from `frequency_ordered_nodes(L)` (`deepfake/data.py`), which returns `ptwt`'s `get_level(L)` in frequency (Gray-code) order. A level-\(L\) decomposition has \(2^L\) nodes. The longest meaningful level for 64,000 samples is 15, and the study targets the middle level, \(L = 8\) (256 nodes, ~250–324 samples per node depending on the filter length).

Daubechies families evaluated by default: `db4`, `db8`, `db16`, `db24`, `db38`. The longer list (`db4`–`db38`, 12 families) is kept commented in `train_models.py`.

### Training

| Setting | Value |
| --- | --- |
| Loss | Cross-entropy with class weights (`total / (2 · n_class)`) |
| Optimizer | AdamW, lr \(10^{-4}\), weight decay \(10^{-4}\) |
| Scheduler | `ReduceLROnPlateau` on dev EER (factor 0.1, patience 3, min lr \(10^{-6}\)) |
| Batch size | 32 |
| Epochs | up to 30 |
| Early stopping | patience 5 on dev EER |
| Score | log-likelihood ratio, `logit[bonafide] − logit[spoof]` |

The checkpoint with the lowest dev EER is evaluated on ASVspoof 2019 LA eval. Results report EER and ROC AUC.

## Repository

```text
.
├── load_dataset.py          # FLAC -> int16 memmaps in datasets/waveforms/
├── train_models.py          # training loop over poolings x wavelets x nodes x seeds
├── deepfake/
│   ├── data.py              # load_data_split, frequency_ordered_nodes
│   ├── features.py          # get_node (wavelet packet node), SubbandDataset
│   ├── cnn.py               # CustomCNN, WaveletPooling, make_pooling
│   ├── training.py          # train/evaluate loops, early stopping, class weights
│   └── metrics.py           # EER, LLR score
├── datasets/waveforms/      # generated, not versioned
├── results/train_all/       # results.csv + runs/
└── requirements.txt
```

### Data files

`load_dataset.py` writes one memmap `.npy` file of shape `(N, 64000)` with dtype `int16` per split, plus a `<split>_meta.csv` holding `filename`, `speaker`, `attack` and `is_bonafide`:

| Split | Utterances | Size |
| --- | --- | --- |
| `train_2019` | 25,380 | 3.2 GB |
| `dev_2019` | 24,844 | 3.2 GB |
| `eval_2019` | 71,237 | 9.1 GB |
| `eval_2021` | 181,566 | 23.2 GB |

`load_data_split` opens these files with `mmap_mode="r"`, so a split is never loaded into RAM all at once. Batches are read from disk through the page cache.

### Outputs

Each finished run appends one row to `results/train_all/results.csv`:

```text
seed, pooling, wavelet, node, eval_split, eer, roc_auc, best_epoch, best_dev_eer, train_time_s
```

The run also writes `history.csv` (loss, EER and lr per epoch) and `model.pt` (best `state_dict`) to `results/train_all/runs/<pooling>/<wavelet>/<node>/seed_<seed>/`. Max and average runs have no wavelet or node, so they go to `runs/<pooling>/seed_<seed>/`.

Runs already listed in `results.csv` are skipped, so an interrupted sweep resumes where it stopped.

## Reproduction

Python 3.13; the requirements pin CUDA 13.0 builds of PyTorch (RTX 5060 Ti, Blackwell).

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Place the corpora one directory above the repository:

- ASVspoof 2019 LA at `../Data/LA/LA/` ([Edinburgh DataShare DS_10283_3336](https://datashare.ed.ac.uk/handle/10283/3336)).
- ASVspoof 2021 LA eval at `../Data/ASVspoof2021_LA_eval/ASVspoof2021_LA_eval/` (`flac/` from [Zenodo 4837263](https://zenodo.org/records/4837263), MD5 `2abee34d8b0b91159555fc4f016e4562`). The labels in `trial_metadata.txt` are not part of that archive; they come from the ASVspoof 2021 keys package (`LA-keys-full.tar.gz`, `keys/LA/CM/trial_metadata.txt`) and must be copied into the same folder.

Build the waveform cache (once):

```bash
python load_dataset.py
```

Train:

```bash
python train_models.py --seeds 42
```

Evaluate the saved checkpoints on ASVspoof 2021 LA (same arguments; appends `ASVspoof 2021` rows to `results.csv`):

```bash
python evaluate_out_datasets.py --seeds 42
```

| Argument | Default | Description |
| --- | --- | --- |
| `--seeds` | `42` | One or more random seeds |
| `--poolings` | `average max wavelet` | Pooling methods to run |
| `--wavelets` | `WAVELET_FAMILIES` | Daubechies families for wavelet pooling |
| `--nodes` | all nodes of `WAVELET_LEVEL` | Subset of wavelet packet nodes |
| `--results-dir` | `results/train_all` | Output directory |

The decomposition level is the `WAVELET_LEVEL` constant in `train_models.py` (currently `3`; the study targets `8`). `--keep-cache` and `--num-workers` are parsed but not used yet.

Each epoch prints one line:

```text
Epoch 01/30 | LR: 0.000100 | Train Loss: 0.4123 | Train EER: 12.3456% | Dev Loss: 0.3890 | Dev EER: 9.8765% | ES Counter: 0/5 | Time: 127s
```

### Compute cost

Measured on an RTX 5060 Ti with batch size 32. The figures are minutes per epoch (train + dev):

| Level | Wavelet pooling (`db4`–`db38`) | Max / average |
| --- | --- | --- |
| 3 | 2.0–2.5 (mean 2.2) | 1.8 |
| 8 | 0.5–1.0 (mean 0.8) | 0.3 |

At level 8 one seed means \(256 \times |\text{wavelets}| + 2\) runs. With 5 families and ~15 epochs per run, that is about 11 days.

## Known issues

- **ASVspoof 2021 FLACs and decoder versions.** The official archive (MD5 verified against Zenodo) contains 181,566 FLACs encoded with libFLAC 1.3.3. Newer decoders (`flac` 1.4.3, and `soundfile` 0.14, which bundles libsndfile 1.2.2) reject 79,645 of them (43.9%) with `LOST_SYNC`. The files are not corrupted. Each FLAC stores the MD5 of the original PCM; comparing it with the decoded audio from `soundfile` 0.12.1 (libsndfile 1.2.0) and from `ffmpeg` gives:

  | Group | Files | Decoded audio |
  | --- | --- | --- |
  | passes `flac -t` | 101,921 | identical to the original |
  | fails `flac -t`, MD5 matches | 59,806 | identical to the original |
  | fails `flac -t`, MD5 differs | 19,839 (10.9%) | isolated spikes outside the 16-bit range (median 14 samples per file, ~0.04% of the audio) |

  `soundfile` 0.12.1 and `ffmpeg`, two independent decoders, return the same samples for all 181,566 files, so any pipeline that reads the corpus sees the same audio. `load_dataset.py` clips the spikes to the `int16` range. This is why `soundfile` is pinned to 0.12.1: newer versions raise an error on files that decode correctly. Do not use `flac -t` to validate this corpus; check the archive MD5 instead.
- **Resume for max/average.** `done_runs` in `train_models.py` and `save_eval` in `evaluate_out_datasets.py` read `results.csv` without `keep_default_na=False`. The empty `wavelet` and `node` fields then become `NaN`. As a result, finished max/average runs are trained again, and `save_eval` finds no 2019 row for them (`IndexError`).

## Previous version

An earlier version of this study used MFCC + LFCC features with a 2D CNN and single-level DWT pooling. Its code and results are archived on Zenodo:

```bibtex
@software{assalim2026wavelet_pooling,
  author       = {Assalim, João Vitor Ramos},
  title        = {Evaluating Wavelet Pooling over Conventional
                  Pooling in CNNs for Audio Deepfake Detection},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.23339527},
  url          = {https://doi.org/10.5281/zenodo.23339526}
}
```
