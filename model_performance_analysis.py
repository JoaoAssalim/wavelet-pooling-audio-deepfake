import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    roc_curve,
)

from deepfake.constants import (
    POOLING_LAYERS,
    WAVELET_FAMILIES,
    WAVELET_LEVEL
)
from deepfake.data import combinations, frequency_ordered_nodes
from deepfake.metrics import compute_eer


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


def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


def eer_per_attack(meta, scores):
    meta = meta.assign(score=scores)
    bonafide = meta.loc[meta["is_bonafide"], "score"].to_numpy()
    rows = []
    for attack, spoof in meta[~meta["is_bonafide"]].groupby("attack"):
        labels = np.r_[np.ones(len(bonafide)), np.zeros(len(spoof))]
        rows.append({"attack": attack, "eer": compute_eer(labels, np.r_[bonafide, spoof["score"]])})
    rows.append({"attack": "pooled", "eer": compute_eer(meta["is_bonafide"].astype(int), meta["score"])})
    return pd.DataFrame(rows)


def eer_per_codec(meta, scores):
    meta = meta.assign(score=scores)
    bonafide = meta.loc[meta["is_bonafide"], "score"].to_numpy()
    rows = []
    for codec, spoof in meta[~meta["is_bonafide"]].groupby("codec"):
        labels = np.r_[np.ones(len(bonafide)), np.zeros(len(spoof))]
        rows.append({"codec": codec, "eer": compute_eer(labels, np.r_[bonafide, spoof["score"]])})
    rows.append({"codec": "pooled", "eer": compute_eer(meta["is_bonafide"].astype(int), meta["score"])})
    return pd.DataFrame(rows)


def analysis_models_attacks_and_codecs():
    args = parse_args()

    for seed in args.seeds:
        for pooling, wavelet, level, node in combinations(args): 
            run_dir = os.path.join(args.results_dir, "runs", pooling, wavelet, node, f"seed_{seed}")

            meta_2019 = pd.read_csv("./datasets/waveforms/eval_2019_meta.csv")

            scores_2019 = np.load(os.path.join(run_dir, "scores_eval_2019.npy"))
            
            attack_scores_2019 = eer_per_attack(meta_2019, scores_2019)
            attack_scores_2019.to_csv(os.path.join(run_dir, "attacks_score_2019.csv"), index=False)
            attack_scores_2019["eer"] = attack_scores_2019["eer"] * 100


            meta_2021 = pd.read_csv("./datasets/waveforms/eval_2021_meta.csv")
            scores_2021 = np.load(os.path.join(run_dir, "scores_eval_2021.npy"))

            attack_scores_2021 = eer_per_attack(meta_2021, scores_2021)
            attack_scores_2021.to_csv(os.path.join(run_dir, "attacks_score_2021.csv"), index=False)
            attack_scores_2021["eer"] = attack_scores_2021["eer"] * 100

            codec_scores_2021 = eer_per_codec(meta_2021, scores_2021)
            codec_scores_2021.to_csv(os.path.join(run_dir, "codecs_score_2021.csv"), index=False)
            codec_scores_2021["eer"] = codec_scores_2021["eer"] * 100



analysis_models_attacks_and_codecs()


