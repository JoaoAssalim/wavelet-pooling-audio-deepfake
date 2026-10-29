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


def done_runs(results_path, dataset):
    if not os.path.exists(results_path):
        return set()
    results = pd.read_csv(results_path, keep_default_na=False)
    results = results[results["eval_split"] == dataset]
    return set(results[RUN_KEY].astype(str).itertuples(index=False, name=None))


def run_key(pooling, seed, wavelet, node):
    return tuple(str(v) for v in (pooling, seed, wavelet, node))


def save_eval(out, run_dir, results_path, key_values, y_eval, eval_split, train_time, save_eval_data):
    rows = []
    scores = out["eval_score"]
    scores_name = f"scores_eval_{"2019" if eval_split == "ASVspoof 2019" else "2021"}.npy"
    np.save(os.path.join(run_dir, scores_name), out["eval_score"])

    fpr, tpr, _ = roc_curve(y_eval, scores, pos_label=1)

    train_df = pd.read_csv(results_path)
    filtered_df = train_df.loc[
        (train_df["seed"] == key_values["seed"])
        & (train_df["pooling"] == key_values["pooling"])
        & (train_df["eval_split"] == "ASVspoof 2019")
    ]

    if key_values["pooling"] == "wavelet":
        filtered_df = filtered_df.loc[
            (train_df["wavelet"] == key_values["wavelet"])
            & (train_df["node"] == key_values["node"])
        ].iloc[0]
    else:
        filtered_df = filtered_df.loc[
            (train_df["wavelet"].isna())
            & (train_df["node"].isna())
        ].iloc[0]

    
    rows.append({
        **key_values,
        "eval_split": eval_split,
        "eer": compute_eer(y_eval, scores),
        "roc_auc": auc(fpr, tpr),
        "best_epoch": filtered_df["best_epoch"],
        "best_dev_eer": filtered_df["best_dev_eer"],
        "train_time_s": round(train_time, 1),
    })

    if save_eval_data:
        pd.DataFrame(rows).to_csv(results_path, mode="a", header=not os.path.exists(results_path), index=False)
    return rows


def evaluate_and_save(
    model_path, y_train, seed, 
    eval_loader, y_eval, wavelet, 
    level, node, pooling, device, 
    run_dir, results_path, dataset, save_eval_data
):
    start = time.time()

    model = CustomCNN(wavelet, level, node, pooling).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    criterion = nn.CrossEntropyLoss(
        weight=class_weights(y_train, device)
    )

    key_values = {
        "seed": seed, "pooling": pooling, "wavelet": wavelet, "node": node,
    }

    eval_output = evaluate_out_models(model, eval_loader, criterion, device)

    row, = save_eval(eval_output, run_dir, results_path, key_values, y_eval, dataset, time.time() - start, save_eval_data)
    log(f"{dataset}: EER {row['eer']:.4%} | AUC {row['roc_auc']:.4f}")


def evaluate_models():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_path = os.path.join(args.results_dir, "results.csv")

    done_2019 = done_runs(results_path, "ASVspoof 2019")
    done_2021 = done_runs(results_path, "ASVspoof 2021")

    # just load for class_weights
    X_train, y_train, train_meta = load_data_split("train_2019")

    # eval loads
    X_eval, y_eval, eval_meta = load_data_split("eval_2019")
    eval_dataset = SubbandDataset(X_eval, y_eval)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)
    log("Eval 2019 dataset loaded")

    X_eval_2021, y_eval_2021, eval_2021_meta = load_data_split("eval_2021")
    eval_dataset_2021 = SubbandDataset(X_eval_2021, y_eval_2021)
    eval_loader_2021 = DataLoader(eval_dataset_2021, batch_size=BATCH_SIZE, shuffle=False)
    log("Eval 2021 dataset loaded")

    for seed in args.seeds:
        for pooling, wavelet, level, node in combinations(args): 
            seed_everything(seed)
            model_path = os.path.join(args.results_dir, "runs", pooling, wavelet, node, f"seed_{seed}", "model.pt")
            run_dir = os.path.join(args.results_dir, "runs", pooling, wavelet, node, f"seed_{seed}")

            if not os.path.exists(os.path.join(run_dir, "scores_eval_2019.npy")):
                log(f"--- seed {seed} | {pooling} | {wavelet or '-'} | {node or '-'}")

                save_eval = run_key(pooling, seed, wavelet, node) not in done_2019

                evaluate_and_save(
                    model_path, y_train, seed, eval_loader, y_eval, wavelet, level, 
                    node, pooling, device, run_dir, results_path, "ASVspoof 2019", save_eval
                )

            if not os.path.exists(os.path.join(run_dir, "scores_eval_2021.npy")):
                log(f"--- seed {seed} | {pooling} | {wavelet or '-'} | {node or '-'}")

                save_eval = run_key(pooling, seed, wavelet, node) not in done_2021

                evaluate_and_save(
                    model_path, y_train, seed, eval_loader_2021, y_eval_2021, wavelet, level, 
                    node, pooling, device, run_dir, results_path, "ASVspoof 2021", save_eval
                )


evaluate_models()