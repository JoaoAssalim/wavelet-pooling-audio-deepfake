# Evaluating Wavelet Pooling over Conventional Pooling in CNNs for Audio Deepfake Detection

João Vitor Ramos Assalim  
Department of Computing, São Paulo State University (UNESP), Bauru, Brazil  
joao.assalim@unesp.br

**Code:** [10.5281/zenodo.23339527](https://doi.org/10.5281/zenodo.23339527)

This repository contains the implementation and results of a comparison between wavelet pooling and conventional pooling in convolutional neural networks for audio deepfake detection. Models are trained on the ASVspoof 2019 Logical Access (LA) partition and evaluated on the ASVspoof 2019 LA and ASVspoof 2021 LA evaluation sets.

## Summary

Audio is resampled to 16 kHz and fixed at 4 s (64,000 samples). Short recordings are repeated; long recordings are truncated. Each utterance is represented by 20 Mel-frequency cepstral coefficients (MFCC) and 20 linear-frequency cepstral coefficients (LFCC), extracted with a 20 ms window and a 10 ms hop, then concatenated into a \(T \times 40\) feature matrix. Features are standardized with the mean and standard deviation of the training set.

The classifier is a CNN with four convolutional blocks. Each block has two \(3 \times 3\) convolutions, batch normalization, and ReLU. Channel width grows from 32 to 256. The first three blocks downsample with one of three pooling operators. The fourth block applies `Dropout2d` (\(p = 0.1\)). Feature maps are reduced by the mean over frequency, then summarized by the mean and standard deviation over time into a 512-dimensional vector. A fully connected classifier (128 units, ReLU, dropout \(p = 0.4\), 2-class output) predicts bonafide or spoof.

Three pooling strategies are compared inside the same backbone:

- Max pooling
- Average pooling
- Wavelet pooling: a single-level Daubechies discrete wavelet transform (`wavedec`, `ptwt`), keeping the low-frequency approximation coefficients, followed by average pooling on the frequency axis

Twelve filter supports are evaluated: `db4`, `db6`, `db8`, `db14`, `db16`, `db18`, `db24`, `db26`, `db28`, `db34`, `db36`, and `db38`.

Each of the 14 configurations is trained for up to 100 epochs with Adam (learning rate \(10^{-3}\), weight decay \(10^{-4}\)), binary cross-entropy, batch size 32, and early stopping with patience 10 on the development equal error rate (EER). The checkpoint with the lowest development EER is evaluated on both test sets. Ten random seeds are used. Differences across pooling methods are tested with Autorank (Friedman test and Nemenyi post-hoc).

## Results

EER is reported as median \(\pm\) median absolute deviation over ten seeds.

### ASVspoof 2019 LA

| Architecture | EER |
| --- | --- |
| CNN + Wavelet db8 | 0.0628 \(\pm\) 0.0017 |
| CNN + Wavelet db6 | 0.0660 \(\pm\) 0.0036 |
| CNN + Wavelet db18 | 0.0663 \(\pm\) 0.0047 |
| CNN + Wavelet db24 | 0.0668 \(\pm\) 0.0023 |
| CNN + Wavelet db4 | 0.0679 \(\pm\) 0.0035 |
| CNN + Max | 0.0684 \(\pm\) 0.0036 |
| CNN + Wavelet db34 | 0.0690 \(\pm\) 0.0013 |
| CNN + Wavelet db26 | 0.0694 \(\pm\) 0.0060 |
| CNN + Average | 0.0694 \(\pm\) 0.0036 |
| CNN + Wavelet db28 | 0.0699 \(\pm\) 0.0045 |
| CNN + Wavelet db38 | 0.0703 \(\pm\) 0.0055 |
| CNN + Wavelet db14 | 0.0724 \(\pm\) 0.0071 |
| CNN + Wavelet db16 | 0.0734 \(\pm\) 0.0048 |
| CNN + Wavelet db36 | 0.0758 \(\pm\) 0.0047 |
| GMM + LFCC | 0.0809 |
| GMM + CQCC | 0.0957 |

On ASVspoof 2019, the proposed models outperform both official baselines. The lowest EER is CNN + wavelet `db8`.

### ASVspoof 2021 LA

| Architecture | EER |
| --- | --- |
| B03 – GMM + CQCC | 0.0926 |
| B04 – GMM + LFCC | 0.0950 |
| CNN + Average | 0.1439 \(\pm\) 0.0027 |
| CNN + Wavelet db24 | 0.1453 \(\pm\) 0.0037 |
| CNN + Wavelet db26 | 0.1456 \(\pm\) 0.0034 |
| CNN + Wavelet db18 | 0.1472 \(\pm\) 0.0038 |
| CNN + Wavelet db6 | 0.1480 \(\pm\) 0.0038 |
| CNN + Wavelet db4 | 0.1481 \(\pm\) 0.0045 |
| CNN + Wavelet db8 | 0.1482 \(\pm\) 0.0056 |
| CNN + Max | 0.1494 \(\pm\) 0.0014 |
| CNN + Wavelet db16 | 0.1508 \(\pm\) 0.0057 |
| CNN + Wavelet db14 | 0.1511 \(\pm\) 0.0026 |
| CNN + Wavelet db34 | 0.1522 \(\pm\) 0.0064 |
| CNN + Wavelet db36 | 0.1532 \(\pm\) 0.0047 |
| CNN + Wavelet db28 | 0.1549 \(\pm\) 0.0049 |
| CNN + Wavelet db38 | 0.1553 \(\pm\) 0.0061 |
| B01 – LFCC-LCNN | 0.1562 |
| B02 – LFCC-GMM | 0.1930 |

On ASVspoof 2021, B03 and B04 remain below the proposed models. Every proposed model remains below B01 and B02. The lowest EER among the proposed models is CNN + average pooling.

The Friedman test does not reject equal central tendency across pooling methods (\(p = 0.071\)). The Nemenyi post-hoc test (critical distance 6.274) finds a difference only between `db8` and `db36` on ASVspoof 2019. Shorter Daubechies supports rank above longer supports. No pairwise difference is found on ASVspoof 2021.

## Repository

```text
.
├── notebooks/
│   ├── normalize_data.ipynb
│   ├── max_pooling_seeds/
│   ├── average_pooling_seeds/
│   ├── wavelet_pooling_seeds/
│   ├── analyze_results.ipynb
│   └── statistical_analysis.ipynb
├── results/v2.5/
└── requirements.txt
```

Seeds: `32`, `42`, `49`, `51`, `69`, `73`, `81`, `91`, `99`, `101`.

| Seed | Output directory |
| --- | --- |
| 42 | `v1.1` |
| 51 | `v1.2` |
| 73 | `v1.3` |
| 69 | `v1.4` |
| 49 | `v1.5` |
| 99 | `v1.6` |
| 101 | `v1.7` |
| 32 | `v1.8` |
| 81 | `v1.9` |
| 91 | `v1.10` |

`Data/` is not versioned. Feature arrays are stored as `float16` with shape `(N, 2, 20, 401)`.

## Reproduction

Python 3.13. The experiments reported in the paper were run on an AMD Ryzen 5 8400F, 32 GB RAM, and an NVIDIA RTX 2060 (6 GB).

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Place ASVspoof 2019 LA at `Data/DS_10283_3336/LA/` ([Edinburgh DataShare DS_10283_3336](https://datashare.ed.ac.uk/handle/10283/3336)) and ASVspoof 2021 LA eval at `Data/ASVspoof2021_LA_eval/` (`flac/` and `trial_metadata.txt`).

Run the notebooks from their own directories, in this order:

1. `notebooks/normalize_data.ipynb` — duration normalization, MFCC/LFCC extraction, and metadata.
2. One notebook per seed in `notebooks/max_pooling_seeds/`, `notebooks/average_pooling_seeds/`, and `notebooks/wavelet_pooling_seeds/`.
3. `notebooks/analyze_results.ipynb` — mean and standard deviation of EER across the ten seeds (`results/v2.5/final_results_ten_seeds_2019.csv`, `results/v2.5/final_results_ten_seeds_2021.csv`).
4. `notebooks/statistical_analysis.ipynb` — Friedman and Nemenyi tests.

Notebooks under `*_pooling_seeds/` read `../../Data/` and write `../../results/v2.5/`.

## Citation

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
