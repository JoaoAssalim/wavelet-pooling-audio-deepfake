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
    WAVELET_FAMILIES,
    WAVELET_LEVEL
)
from deepfake.cnn import WaveletEntropyBasedCNN
from deepfake.data import load_data_split , seed_everything
from deepfake.features import SubbandDataset
from deepfake.metrics import compute_eer
from deepfake.training import class_weights, run_model, evaluate_and_save


RUN_KEY = ["seed", "wavelet"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--wavelets", nargs="+", default=WAVELET_FAMILIES)
    parser.add_argument("--results-dir", default="results/train_entropy_based")
    parser.add_argument("--selection", choices=["min", "max"], default="min")
    return parser.parse_args()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def done_runs(results_path, dataset):
    if not os.path.exists(results_path):
        return set()
    results = pd.read_csv(results_path, keep_default_na=False)
    results = results[results["eval_split"] == dataset]
    return set(results[RUN_KEY].astype(str).itertuples(index=False, name=None))


def run_key(seed, wavelet):
    return tuple(str(v) for v in (seed, wavelet))


def save_run(out, run_dir, results_path, key_values, y_eval, eval_split, train_time):
    os.makedirs(run_dir, exist_ok=True)
    pd.DataFrame(out["history"]).to_csv(os.path.join(run_dir, "history.csv"), index=False)
    torch.save(out["state_dict"], os.path.join(run_dir, "model.pt"))
    scores_name = f"scores_eval_{"2019" if eval_split == "ASVspoof 2019" else "2021"}.npy"
    
    np.save(os.path.join(run_dir, scores_name), out["eval_score"])
    np.save(os.path.join(run_dir, scores_name.replace("scores", "selection")), out["selection"])

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


def evaluate_with_selection(model, eval_loader, criterion, device, best_model_state, best_model_status, history):
    selections = []
    pooling = model.features[2]
    hook = pooling.register_forward_hook(
        lambda module, inputs, output: selections.append(module.last_selection.cpu())
    )
    out = evaluate_and_save(model, eval_loader, criterion, device, best_model_state, best_model_status, history)
    hook.remove()

    out["selection"] = torch.cat(selections).numpy().astype(np.uint8)
    return out


def load_saved_run(run_dir, results_path, seed, wavelet):
    best_model_state = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu", weights_only=True)
    history = pd.read_csv(os.path.join(run_dir, "history.csv")).to_dict("list")

    results = pd.read_csv(results_path)
    row_2019 = results[
        (results["seed"] == seed)
        & (results["wavelet"] == wavelet)
        & (results["eval_split"] == "ASVspoof 2019")
    ].iloc[0]
    best_model_status = {"epoch": row_2019["best_epoch"], "best_dev_eer": row_2019["best_dev_eer"]}

    return best_model_state, history, best_model_status


def run_models():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # min keeps results.csv / runs/; max goes to results_max_entropy.csv / runs_max_entropy/
    suffix = "" if args.selection == "min" else "_max_entropy"
    results_path = os.path.join(args.results_dir, f"results{suffix}.csv")

    done_2019 = done_runs(results_path, "ASVspoof 2019")
    done_2021 = done_runs(results_path, "ASVspoof 2021")
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

    X_eval_2021, y_eval_2021, eval_meta_2021 = load_data_split("eval_2021")
    eval_dataset_2021 = SubbandDataset(X_eval_2021, y_eval_2021)
    eval_loader_2021 = DataLoader(eval_dataset_2021, batch_size=BATCH_SIZE, shuffle=False)
    log("Eval 2021 dataset loaded")

    for seed in args.seeds:
        for wavelet in args.wavelets:
            seed_everything(seed)

            key = run_key(seed, wavelet)
            if key in done_2019 and key in done_2021:
                continue

            run_dir = os.path.join(args.results_dir, f"runs{suffix}", wavelet, f"seed_{seed}")
            log(f"--- seed {seed} | {wavelet or '-'} | {args.selection} entropy")
            start = time.time()

            criterion = nn.CrossEntropyLoss(
                weight=class_weights(y_train, device)
            )

            custom_cnn = WaveletEntropyBasedCNN(wavelet, WAVELET_LEVEL, args.selection).to(device)

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
                "seed": seed, "wavelet": wavelet,
            }

            if key not in done_2019:
                eval_output = evaluate_with_selection(
                    model, eval_loader, criterion, device, best_model_state,
                    best_model_status, history
                )

                row, = save_run(eval_output, run_dir, results_path, key_values, y_eval, "ASVspoof 2019", time.time() - start)
                log(f"ASVspoof 2019: EER {row['eer']:.4%} | AUC {row['roc_auc']:.4f}")

                done_2019.add(key)

            if key not in done_2021:
                eval_output_2021 = evaluate_with_selection(
                    model, eval_loader_2021, criterion, device, best_model_state,
                    best_model_status, history
                )

                row_2021, = save_run(eval_output_2021, run_dir, results_path, key_values, y_eval_2021, "ASVspoof 2021", time.time() - start)
                log(f"ASVspoof 2021: EER {row_2021['eer']:.4%} | AUC {row_2021['roc_auc']:.4f}")

                done_2021.add(key)

            if device.type == "cuda":
                torch.cuda.empty_cache()


run_models()