import argparse
import os
import time
import torch

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    roc_curve,
)
import torch.nn as nn
from torch.utils.data import DataLoader

from deepfake.constants import (
    BATCH_SIZE,
    POOLING_LAYERS,
    RUN_KEY,
    WAVELET_FAMILIES,
    WAVELET_LEVEL
)
from deepfake.cnn import CustomCNN
from deepfake.data import combinations, frequency_ordered_nodes, load_data_split , seed_everything
from deepfake.features import SubbandDataset
from deepfake.metrics import compute_eer
from deepfake.training import class_weights, evaluate_out_models


WAVELET_NODES = frequency_ordered_nodes(WAVELET_LEVEL)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--wavelets", nargs="+", default=WAVELET_FAMILIES)
    parser.add_argument("--poolings", nargs="+", default=POOLING_LAYERS)
    parser.add_argument("--nodes", nargs="+", default=WAVELET_NODES, choices=WAVELET_NODES)
    parser.add_argument("--results-dir", default="results/train_all")
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_transmission_type(score_path, eval_split):
    if eval_split == "2019":
        return "attack"
    elif eval_split == "2021" and "attack" in score_path:
        return "attack"
    elif eval_split == "2021 DF" and "attack" in score_path:
        return "attack"
    return "codec"


def stack_results(score_path, results_path, eval_split, pooling, wavelet, node, seed):
    df = pd.read_csv(score_path)
    transmission_type = get_transmission_type(score_path, eval_split)
    rows = []

    for idx, row in df.iterrows():
        rows.append({
            "eval_split": eval_split,
            "eer": row["eer"],
            "type": transmission_type,
            "transmission": row[transmission_type],
            "pooling": pooling,
            "wavelet": wavelet,
            "node": node,
            "seed": seed
        })

    pd.DataFrame(rows).to_csv(results_path, mode="a", header=not os.path.exists(results_path), index=False)
    return rows


def evaluate_models():
    args = parse_args()
    results_path = os.path.join(args.results_dir, "results_transmissions.csv")

    for seed in args.seeds:
        for pooling, wavelet, level, node in combinations(args): 
            seed_everything(seed)
            run_dir = os.path.join(args.results_dir, "runs", pooling, wavelet, node, f"seed_{seed}")
            # ASV 2019
            stack_results(os.path.join(run_dir, "attacks_score_2019.csv"), results_path, "2019", pooling, wavelet, node, seed)
            # ASV 2021 Attack
            stack_results(os.path.join(run_dir, "attacks_score_2021.csv"), results_path, "2021 LA", pooling, wavelet, node, seed)
            # ASV 2021 Codec
            stack_results(os.path.join(run_dir, "codecs_score_2021.csv"), results_path, "2021 LA", pooling, wavelet, node, seed)
            # ASV 2021 Attack
            stack_results(os.path.join(run_dir, "attacks_score_2021_df.csv"), results_path, "2021 DF", pooling, wavelet, node, seed)
            # ASV 2021 Codec
            stack_results(os.path.join(run_dir, "codecs_score_2021_df.csv"), results_path, "2021 DF", pooling, wavelet, node, seed)



evaluate_models()