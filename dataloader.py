import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
torch.multiprocessing.set_sharing_strategy('file_system')
from tqdm import tqdm
import numpy as np
import os
from os.path import join, basename
from boltons.fileutils import iter_find_files
import soundfile as sf
import librosa
import pickle
from multiprocessing import Pool
import random
import torchaudio
import math
from torchaudio.datasets import LIBRISPEECH


def collate_fn_padd(batch):
    """collate_fn_padd
    Padds batch of variable length

    :param batch:
    """
    # get sequence lengths
    spects = [t[0] for t in batch]
    segs = [t[1] for t in batch]
    labels = [t[2] for t in batch]
    lengths = [t[3] for t in batch]
    fnames = [t[4] for t in batch]

    padded_spects = torch.nn.utils.rnn.pad_sequence(spects, batch_first=True)
    lengths = torch.LongTensor(lengths)

    return padded_spects, segs, labels, lengths, fnames


def spectral_size(wav_len):
    layers = [(10,5,0), (8,4,0), (4,2,0), (4,2,0), (4,2,0)]
    for kernel, stride, padding in layers:
        wav_len = math.floor((wav_len + 2*padding - 1*(kernel-1) - 1)/stride + 1)
    return wav_len


def get_subset(dataset, percent):
    A_split = int(len(dataset) * percent)
    B_split = len(dataset) - A_split
    dataset, _ = torch.utils.data.random_split(dataset, [A_split, B_split])
    return dataset


class WavPhnDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.data = list(iter_find_files(self.path, "*.wav"))
        super(WavPhnDataset, self).__init__()

    @staticmethod
    def get_datasets(path):
        raise NotImplementedError

    def process_file(self, wav_path):
        phn_path = wav_path.replace("wav", "phn")

        # load audio
        audio, sr = torchaudio.load(wav_path)
        audio = audio[0]
        audio_len = len(audio)
        spectral_len = spectral_size(audio_len)
        len_ratio = (audio_len / spectral_len)

        # load labels -- segmentation and phonemes
        with open(phn_path, "r") as f:
            lines = f.readlines()
            lines = list(map(lambda line: line.split(" "), lines))

            # get segment times
            times = torch.FloatTensor(list(map(lambda line: int(int(line[1]) / len_ratio), lines)))[:-1]  # don't count end time as boundary

            # get phonemes in each segment (for K times there should be K+1 phonemes)
            phonemes = list(map(lambda line: line[2].strip(), lines))

        return audio, times.tolist(), phonemes, wav_path

    def __getitem__(self, idx):
        audio, seg, phonemes, fname = self.process_file(self.data[idx])
        return audio, seg, phonemes, spectral_size(len(audio)), fname

    def __len__(self):
        return len(self.data)


class TrainTestDataset(WavPhnDataset):
    def __init__(self, path):
        super(TrainTestDataset, self).__init__(path)

    @staticmethod
    def get_datasets(path, val_ratio=0.1):
        train_dataset = TrainTestDataset(join(path, 'train'))
        test_dataset  = TrainTestDataset(join(path, 'test'))

        train_len   = len(train_dataset)
        train_split = int(train_len * (1 - val_ratio))
        val_split   = train_len - train_split
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_split, val_split])

        train_dataset.path = join(path, 'train')
        val_dataset.path = join(path, 'train')

        return train_dataset, val_dataset, test_dataset


class TrainValTestDataset(WavPhnDataset):
    def __init__(self, paths):
        super(TrainValTestDataset, self).__init__(paths)

    @staticmethod
    def get_datasets(path, percent=1.0):
        train_dataset = TrainValTestDataset(join(path, 'train'))
        if percent != 1.0:
            train_dataset = get_subset(train_dataset, percent)
            train_dataset.path = join(path, 'train')
        val_dataset   = TrainValTestDataset(join(path, 'val'))
        test_dataset  = TrainValTestDataset(join(path, 'test'))

        return train_dataset, val_dataset, test_dataset


class LibriSpeechDataset(LIBRISPEECH):
    def __init__(self, path, subset, percent):
        self.libri_dataset = LIBRISPEECH(path, url=subset, download=False)
        if percent != 1.0:
            self.libri_dataset = get_subset(self.libri_dataset, percent)
        self.path = path
    
    def __getitem__(self, idx):
        wav, sr, utt, spk_id, chp_id, utt_id = self.libri_dataset[idx]
        wav = wav[0]
        return wav, None, None, spectral_size(len(wav)), None

    def __len__(self):
        return len(self.libri_dataset)


class MixedDataset(Dataset):
    def __init__(self, ds1, ds2):
        self.ds1 = ds1
        self.ds2 = ds2
        self.path = f"{ds1.path}+{ds2.path}"
        self.ds1_len, self.ds2_len = len(ds1), len(ds2)
    
    def __len__(self):
        return self.ds1_len + self.ds2_len
    
    def __getitem__(self, idx):
        if idx < self.ds1_len:
            return self.ds1[idx]
        else:
            return self.ds2[idx - self.ds1_len]
