"""
Claude Code generated script to calculate t-DCF
"""

import argparse
import os

import numpy as np
import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data")
ASV_2019 = os.path.join(DATA_DIR, "LA/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt")
ASV_2021_DIR = os.path.join(DATA_DIR, "ASVspoof2021_LA_eval/ASVspoof2021_LA_eval")
META_DIR = "./datasets/waveforms"

P_SPOOF = 0.05
COST_MODEL = {
    "Pspoof": P_SPOOF,
    "Ptar": (1 - P_SPOOF) * 0.99,
    "Pnon": (1 - P_SPOOF) * 0.01,
    "Cmiss": 1,
    "Cfa": 10,
    "Cfa_spoof": 10,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/train_all")
    # results_max_entropy.csv reads its runs from runs_max_entropy/
    parser.add_argument("--results-file", default="results.csv")
    return parser.parse_args()


def compute_det_curve(target_scores, nontarget_scores):
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]

    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))
    return frr, far, thresholds


def asv_error_rates(tar_asv, non_asv, spoof_asv):
    # ASV threshold fixed at its EER point (target vs nontarget)
    frr, far, thresholds = compute_det_curve(tar_asv, non_asv)
    threshold = thresholds[np.argmin(np.abs(frr - far))]
    return {
        "Pfa_asv": np.mean(non_asv >= threshold),
        "Pmiss_asv": np.mean(tar_asv < threshold),
        "Pmiss_spoof_asv": np.mean(spoof_asv < threshold),
        "Pfa_spoof_asv": np.mean(spoof_asv >= threshold),
    }


def min_tdcf_2019(bonafide_cm, spoof_cm, asv):
    # legacy t-DCF (Kinnunen et al., 2018), used in ASVspoof 2019
    c = COST_MODEL
    C1 = c["Ptar"] * (c["Cmiss"] - c["Cmiss"] * asv["Pmiss_asv"]) - c["Pnon"] * c["Cfa"] * asv["Pfa_asv"]
    C2 = c["Cfa_spoof"] * c["Pspoof"] * (1 - asv["Pmiss_spoof_asv"])

    Pmiss_cm, Pfa_cm, _ = compute_det_curve(bonafide_cm, spoof_cm)
    tdcf = C1 * Pmiss_cm + C2 * Pfa_cm
    return float(np.min(tdcf / np.minimum(C1, C2)))


def min_tdcf_2021(bonafide_cm, spoof_cm, asv):
    # revised t-DCF (Kinnunen et al., 2020), used in ASVspoof 2021
    c = COST_MODEL
    C0 = c["Ptar"] * c["Cmiss"] * asv["Pmiss_asv"] + c["Pnon"] * c["Cfa"] * asv["Pfa_asv"]
    C1 = c["Ptar"] * c["Cmiss"] - C0
    C2 = c["Cfa_spoof"] * c["Pspoof"] * asv["Pfa_spoof_asv"]

    Pmiss_cm, Pfa_cm, _ = compute_det_curve(bonafide_cm, spoof_cm)
    tdcf = C0 + C1 * Pmiss_cm + C2 * Pfa_cm
    return float(np.min(tdcf / (C0 + np.minimum(C1, C2))))


def load_asv_2019():
    asv = pd.read_csv(ASV_2019, sep=" ", header=None, names=["attack", "key", "score"])
    return asv_error_rates(
        asv.loc[asv["key"] == "target", "score"].to_numpy(),
        asv.loc[asv["key"] == "nontarget", "score"].to_numpy(),
        asv.loc[asv["key"] == "spoof", "score"].to_numpy(),
    )


def load_asv_2021():
    # keys/LA/ASV/trial_metadata.txt (saved as trial_metadata_tdcf.txt) and keys/LA/ASV/ASVTorch_Kaldi/score.txt, row aligned
    key = pd.read_csv(os.path.join(ASV_2021_DIR, "trial_metadata_tdcf.txt"), sep=" ", header=None)
    score = pd.read_csv(os.path.join(ASV_2021_DIR, "score.txt"), sep=" ", header=None)
    assert (key[0] == score[0]).all() and (key[1] == score[1]).all()

    asv = pd.DataFrame({"key": key[5], "phase": key[7], "score": score[2]})
    asv = asv[asv["phase"] == "eval"]
    return asv_error_rates(
        asv.loc[asv["key"] == "target", "score"].to_numpy(),
        asv.loc[asv["key"] == "nontarget", "score"].to_numpy(),
        asv.loc[asv["key"] == "spoof", "score"].to_numpy(),
    )


def run_dir_of(results_dir, runs_name, row):
    if "pooling" in row:
        return os.path.join(results_dir, runs_name, row["pooling"], row["wavelet"], row["node"], f"seed_{row['seed']}")
    return os.path.join(results_dir, runs_name, row["wavelet"], f"seed_{row['seed']}")


def compute_tdcf():
    args = parse_args()
    results_path = os.path.join(args.results_dir, args.results_file)
    runs_name = os.path.splitext(args.results_file)[0].replace("results", "runs", 1)
    results = pd.read_csv(results_path, keep_default_na=False)

    meta_2019 = pd.read_csv(os.path.join(META_DIR, "eval_2019_meta.csv"))
    meta_2021 = pd.read_csv(os.path.join(META_DIR, "eval_2021_meta.csv"))
    # official 2021 numbers use only the "eval" phase (no progress / hidden trials)
    eval_phase_2021 = (meta_2021["phase"] == "eval").to_numpy()

    splits = {
        "ASVspoof 2019": ("scores_eval_2019.npy", meta_2019["is_bonafide"].to_numpy(),
                          np.ones(len(meta_2019), dtype=bool), load_asv_2019(), min_tdcf_2019),
        "ASVspoof 2021": ("scores_eval_2021.npy", meta_2021["is_bonafide"].to_numpy(),
                          eval_phase_2021, load_asv_2021(), min_tdcf_2021),
    }

    min_tdcf = []
    for _, row in results.iterrows():
        scores_name, is_bonafide, keep, asv, tdcf_fn = splits[row["eval_split"]]
        scores_path = os.path.join(run_dir_of(args.results_dir, runs_name, row), scores_name)
        if not os.path.exists(scores_path):
            print(f"missing {scores_path}")
            min_tdcf.append(np.nan)
            continue

        scores = np.load(scores_path)
        assert len(scores) == len(is_bonafide), scores_path
        min_tdcf.append(tdcf_fn(scores[keep & is_bonafide], scores[keep & ~is_bonafide], asv))

    results["min_tdcf"] = min_tdcf

    tmp_path = results_path + ".tmp"
    results.to_csv(tmp_path, index=False)
    os.replace(tmp_path, results_path)
    print(results.groupby(["eval_split"])["min_tdcf"].describe())


compute_tdcf()
