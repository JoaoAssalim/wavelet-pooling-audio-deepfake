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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from deepfake.constants import (
    BATCH_SIZE,
    EPOCHS,
    PATIANCE,
    POOLING_LAYERS,
    RUN_KEY,
    WAVELET_FAMILIES,
    WAVELET_LEVEL
)
from deepfake.cnn import CustomCNN
from deepfake.data import combinations, frequency_ordered_nodes, load_data_split , seed_everything
from deepfake.features import SubbandDataset
from deepfake.metrics import compute_eer
from deepfake.training import class_weights, run_model, evaluate_and_save


WAVELET_NODES = frequency_ordered_nodes(WAVELET_LEVEL)


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


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
    scores_name = f"scores_eval_{"2019" if eval_split == "ASVspoof 2019" else "2021"}.npy"
    
    np.save(os.path.join(run_dir, scores_name), out["eval_score"])

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

            done.add(run_key(pooling, seed, wavelet, node))

            if device.type == "cuda":
                torch.cuda.empty_cache()


run_models()        