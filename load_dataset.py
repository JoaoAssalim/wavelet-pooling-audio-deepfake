

import os
import time
import librosa

import numpy as np
import pandas as pd

from joblib import Parallel, delayed


TARGET_DURATION = 4.0
SR = 16000
N_JOBS = -1

target_length = int(TARGET_DURATION * SR)

output_path = "./datasets/waveforms/"
os.makedirs(output_path, exist_ok=True)


def read_asv_protocol(path, dataset="asv_2019"):
    if dataset == "asv_2019":
        prot = pd.read_csv(path, sep=" ", header=None, names=["speaker", "filename", "dash", "attack", "key"])
    elif dataset == "asv_2021":
        prot = pd.read_csv(path, sep=r"\s+", header=None, names=["speaker", "filename", "codec", "transmission", "attack", "key", "trim", "phase"])
    prot["is_bonafide"] = prot["key"] == "bonafide"
    return prot.reset_index(drop=True)



def load_waveform(path):
    y, sr = librosa.load(path, sr=SR)

    if len(y) < target_length:
        y = librosa.util.pad_center(y, size=target_length)

    return y[:target_length]


def show_progress(label, done, total, t0):
    elapsed = time.time() - t0
    rate = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / rate if rate > 0 else 0
    msg = (f"{label}: {done}/{total} ({done / total:.1%}) | {rate:.1f} áudios/s | "
           f"decorrido {elapsed / 60:.1f} min | restante ~{eta / 60:.1f} min")
    print(f"\r{msg:<130}", end="", flush=True)


def cache_split(split_name, paths, prot):
    wave_file = f"{output_path}{split_name}.npy"
    X = np.lib.format.open_memmap(
        wave_file, mode="w+", dtype=np.int16,
        shape=(len(paths), target_length),
    )

    t0 = time.time()
    results = Parallel(n_jobs=N_JOBS, return_as="generator", batch_size=32)(
        delayed(load_waveform)(p) for p in paths
    )
    for i, y in enumerate(results):
        X[i] = np.clip(np.round(y * 32768), -32768, 32767)
        if (i + 1) % 100 == 0 or i + 1 == len(paths):
            show_progress(split_name, i + 1, len(paths), t0)

    X.flush()
    prot.to_csv(f"{output_path}{split_name}_meta.csv", index=False)
    print(f"\n{split_name}: {X.shape} | {os.path.getsize(wave_file)/1e9:.2f} GB | {(time.time() - t0) / 60:.1f} min | "
          f"bonafide={int(prot['is_bonafide'].sum())} spoof={int((~prot['is_bonafide']).sum())}")


def cache_asv(split_name, flac_dir, protocol_path, dataset="asv_2019"):
    prot = read_asv_protocol(protocol_path, dataset)
    paths = [os.path.join(flac_dir, f"{n}.flac") for n in prot["filename"]]
    cache_split(split_name, paths, prot)


cache_asv("train_2019", "../Data/LA/LA/ASVspoof2019_LA_train/flac",
          "../Data/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
          dataset="asv_2019")

cache_asv("dev_2019", "../Data/LA/LA/ASVspoof2019_LA_dev/flac",
          "../Data/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
          dataset="asv_2019")

cache_asv("eval_2019", "../Data/LA/LA/ASVspoof2019_LA_eval/flac",
          "../Data/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
          dataset="asv_2019")

cache_asv("eval_2021", "../Data/ASVspoof2021_LA_eval/ASVspoof2021_LA_eval/flac",
          "../Data/ASVspoof2021_LA_eval/ASVspoof2021_LA_eval/trial_metadata.txt",
          dataset="asv_2021")
