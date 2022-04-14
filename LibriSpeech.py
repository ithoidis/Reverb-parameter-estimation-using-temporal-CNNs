import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import librosa
import pandas as pd
from tqdm import tqdm
import itertools
from sklearn.preprocessing import LabelEncoder
from scipy.io.wavfile import read as wavread
from sklearn.model_selection import train_test_split
import torchaudio

class LibriSpeech(Dataset):
    def __init__(self, path=LIBRISPEECH_PATH, type='train', train_type='train-clean-100', channels=None, snrs=None,
                 labels=None, target=None):
        super().__init__()

        self.path = path
        self.type = type
        self.train_type = train_type
        self.sample_rate = 16000
        self.dataset = torchaudio.datasets.LIBRISPEECH(self.path, url=self.train_type if type == 'train' else 'test-clean')


    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id = self.dataset.__getitem__(idx)
        assert sample_rate == self.sample_rate
        id = {'fs': sample_rate,
              'utterance': utterance.lower(),
              'speaker_id': speaker_id,
              'chapter_id':chapter_id,
              'utterance_id': utterance_id}
        return waveform.squeeze(), id # utterance_int
