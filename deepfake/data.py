import os
import itertools
import ptwt
import random
import torch

import numpy as np
import pandas as pd

from deepfake.constants import WAVELET_LEVEL

DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
WAVEFORMS_DIR = os.path.join(DATASETS_DIR, "waveforms")


def frequency_ordered_nodes(level=8):
    return ptwt.WaveletPacket(data=torch.zeros(1, 64000), wavelet="db1", maxlevel=level).get_level(level)


def load_data_split(dataset):
    X = np.load(f"{WAVEFORMS_DIR}/{dataset}.npy", mmap_mode="r")
    meta = pd.read_csv(f"{WAVEFORMS_DIR}/{dataset}_meta.csv")
    y = meta["is_bonafide"].to_numpy().astype(np.int64)
    return X.reshape(X.shape[0], -1, X.shape[-1]), y, meta


def combinations(args):
    combos = []

    if "wavelet" in args.poolings:
        combos += list(
            itertools.product(args.wavelets, args.nodes, ["wavelet"])
        )
    if "max" in args.poolings:
        combos += [("", "", "max")] 
    if "average" in args.poolings:
            combos += [("", "", "average")] 
    for wavelet, node, pooling in combos:
        yield pooling, wavelet, WAVELET_LEVEL, node


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
