import random
import itertools
import argparse
import time
import os

import numpy as np
import pandas as pd
import ptwt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    auc,
    roc_curve,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from deepfake.data import frequency_ordered_nodes, load_data_split
from deepfake.metrics import compute_eer
from deepfake.cnn import CustomCNN
from deepfake.training import class_weights, run_model, evaluate_and_save
from deepfake.features import SubbandDataset


# WAVELET_FAMILIES = ["db4", "db6", "db8", "db14", "db16", "db18", "db24", "db26", "db28", "db34", "db36", "db38"]
WAVELET_FAMILIES = ["db4", "db8", "db16", "db24", "db38"]
WAVELET_LEVEL = 3
WAVELET_NODES = frequency_ordered_nodes(WAVELET_LEVEL)

POOLING_LAYERS = [
    "average",
    "max",
    "wavelet"
]

EPOCHS = 30
PATIANCE = 5 
BATCH_SIZE = 32

RUN_KEY = ["pooling", "seed", "wavelet", "node"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--wavelets", nargs="+", default=WAVELET_FAMILIES)
    parser.add_argument("--poolings", nargs="+", default=POOLING_LAYERS)
    parser.add_argument("--nodes", nargs="+", default=WAVELET_NODES, choices=WAVELET_NODES)
    parser.add_argument("--results-dir", default="results/train_all")
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


def done_runs(results_path):
    if not os.path.exists(results_path):
        return set()
    results = pd.read_csv(results_path)
    return set(results[RUN_KEY].astype(str).itertuples(index=False, name=None))


def run_key(pooling, seed, wavelet, node):
    return tuple(str(v) for v in (pooling, seed, wavelet, node))


def save_run(out, run_dir, results_path, key_values, y_eval, eval_split, train_time):
    os.makedirs(run_dir, exist_ok=True)
    pd.DataFrame(out["history"]).to_csv(os.path.join(run_dir, "history.csv"), index=False)
    torch.save(out["state_dict"], os.path.join(run_dir, "model.pt"))

    rows = []
    scores = out["eval_score"]
    fpr, tpr, _ = roc_curve(y_eval, scores, pos_label=1)
    rows.append({
        **key_values,
        "eval_split": eval_split,
        "eer": compute_eer(y_eval, scores),
        "roc_auc": auc(fpr, tpr),
        "best_epoch": out["best"]["epoch"],
        "best_dev_eer": out["best"]["best_dev_eer"],
        "train_time_s": round(train_time, 1),
    })

    pd.DataFrame(rows).to_csv(results_path, mode="a", header=not os.path.exists(results_path), index=False)
    return rows
    

def run_models():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_path = os.path.join(args.results_dir, "results.csv")

    done = done_runs(results_path)
    log("Loading datasets...")

    # load train
    X_train, y_train, train_meta = load_data_split("train_2019")
    train_dataset = SubbandDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    log("Train dataset loaded")

    #load dev
    X_dev, y_dev, dev_meta = load_data_split("dev_2019")
    dev_dataset = SubbandDataset(X_dev, y_dev)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
    log("Dev dataset loaded")

    # load eval
    X_eval, y_eval, eval_meta = load_data_split("eval_2019")
    eval_dataset = SubbandDataset(X_eval, y_eval)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)
    log("Eval 2019 dataset loaded")

    # X_eval_2021, y_eval_2021, eval_2021_meta = load_data_split("eval_2021")
    # eval_dataset_2021 = SubbandDataset(X_eval_2021, y_eval_2021)
    # eval_loader_2021 = DataLoader(eval_dataset_2021, batch_size=BATCH_SIZE, shuffle=False)
    # log("Eval 2021 loaded")

    for seed in args.seeds:
        for pooling, wavelet, level, node in combinations(args):
            seed_everything(seed)
            if run_key(pooling, seed, wavelet, node) in done: 
                continue

            run_dir = os.path.join(args.results_dir, "runs", pooling, wavelet, node, f"seed_{seed}")
            log(f"--- seed {seed} | {pooling} | {wavelet or '-'} | {node or '-'}")
            start = time.time()

            custom_cnn = CustomCNN(wavelet, level, node, pooling).to(device)

            criterion = nn.CrossEntropyLoss(
                weight=class_weights(y_train, device)
            )
        
            optimizer = torch.optim.AdamW(
                custom_cnn.parameters(),
                lr=1e-4,
                weight_decay=1e-4
            )
        
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.1,
                patience=3,
                threshold=1e-4,
                min_lr=1e-6
            )

            model, history, best_model_state, best_model_status = run_model(
                custom_cnn, train_loader, dev_loader, criterion,
                optimizer, scheduler, device, EPOCHS, PATIANCE
            )

            # Eval
            key_values = {
                "seed": seed, "pooling": pooling, "wavelet": wavelet, "node": node,
            }

            eval_output = evaluate_and_save(
                model, eval_loader, criterion, device, best_model_state,
                best_model_status, history
            )

            row, = save_run(eval_output, run_dir, results_path, key_values, y_eval, "ASVspoof 2019", time.time() - start)
            log(f"ASVspoof 2019: EER {row['eer']:.4%} | AUC {row['roc_auc']:.4f}")

            # eval_output = evaluate_and_save(
            #     model, eval_loader_2021, criterion, device, best_model_state,
            #     best_model_status, history
            # )

            # row, = save_run(eval_output, run_dir, results_path, key_values, y_eval_2021, "ASVspoof 2021", time.time() - start)
            # log(f"ASVspoof 2021: EER {row['eer']:.4%} | AUC {row['roc_auc']:.4f}")

            done.add(run_key(pooling, seed, wavelet, node))

            if device.type == "cuda":
                torch.cuda.empty_cache()


run_models()        