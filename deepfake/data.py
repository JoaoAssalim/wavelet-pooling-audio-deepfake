import os
import ptwt

import numpy as np
import pandas as pd
import torch

DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
WAVEFORMS_DIR = os.path.join(DATASETS_DIR, "waveforms")


def frequency_ordered_nodes(level=8):
    return ptwt.WaveletPacket(data=torch.zeros(1, 64000), wavelet="db1", maxlevel=level).get_level(level)


def load_data_split(dataset):
    X = np.load(f"{WAVEFORMS_DIR}/{dataset}.npy", mmap_mode="r")
    meta = pd.read_csv(f"{WAVEFORMS_DIR}/{dataset}_meta.csv")
    y = meta["is_bonafide"].to_numpy().astype(np.int64)
    return X.reshape(X.shape[0], -1, X.shape[-1]), y, meta
