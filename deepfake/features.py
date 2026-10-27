import numpy as np
import ptwt
import pywt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

SR = 16000
N_COEFFS = 20


def reconstruct_node(waveforms, wavelet_family, level, node):
    dtwpt = ptwt.WaveletPacket(data=waveforms, wavelet=wavelet_family, maxlevel=level)

    all_nodes = dtwpt.get_level(level)

    rec_dtwpt = ptwt.WaveletPacket(data=torch.zeros_like(waveforms), wavelet=wavelet_family, maxlevel=level)
    rec_dtwpt.initialize([n for n in all_nodes if n != node])
    rec_dtwpt[node] = dtwpt[node]
    rec_dtwpt.reconstruct()

    return rec_dtwpt[""]


# def get_node(waveforms, wavelet_family, level, node):
#     dtwpt = ptwt.WaveletPacket(data=waveforms, wavelet=wavelet_family, maxlevel=level)
#     return dtwpt[node]


def get_node(waveforms, wavelet_family, level, node):
    wavelet = pywt.Wavelet(wavelet_family)
    filters = {"a": wavelet.dec_lo[::-1], "d": wavelet.dec_hi[::-1]}
    padding = wavelet.dec_len - 2
    channels = waveforms.shape[1]

    coeffs = waveforms
    for key in node:
        coeffs = F.pad(coeffs, (padding, padding + coeffs.shape[-1] % 2), mode="reflect")
        kernel = torch.tensor(filters[key], dtype=coeffs.dtype, device=coeffs.device)
        coeffs = F.conv1d(coeffs, kernel.repeat(channels, 1, 1), stride=2, groups=channels)

    return coeffs


class SubbandDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx].astype(np.float32) / 32768), self.y[idx]
