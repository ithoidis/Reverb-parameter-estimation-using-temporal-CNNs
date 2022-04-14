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
from utils import *
import random

class NoiseReal(Dataset):
        def __init__(self, root=NOISE_ROOT, type='train', datasets=None):
            super().__init__()
            if datasets == None:
                datasets = ['UrbanSound8k']

            # Music, speech babble, other classes
            self.root = root
            self.type = type
            self.sample_rate = 16000
            self.ext_audio = tuple(['.wav', '.flac', '.ogg'])
            assert self.type in ["train", "test"]

            _walker = []
            for dat_name in datasets:
                path = os.path.join(*[root, dat_name, self.type])
                walker = walk_files(path, suffix=self.ext_audio, prefix=True, remove_suffix=False)
                _walker.extend(list(walker))
            np.random.shuffle(_walker)
            self._walker = _walker

            self.classes = self.get_classes()
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.get_classes())

        def get_classes(self):
            classes = []
            for idx in range(self.__len__()):
                y = self.__getclass__(idx)
                classes.append(y)
                classes = np.unique(classes).tolist()
            return classes

        def encode(self, text_label):
            return self.label_encoder.transform(text_label)

        def decode(self, int_label):
            return self.label_encoder.inverse_transform(int_label)

        def __len__(self):
            return len(self._walker)

        def __getclass__(self, idx):
            audio_filepath = self._walker[idx]
            y = os.path.split(audio_filepath)[1].split("-")[0]
            return y

        def __getitem__(self, idx, n_samples=None):
            audio_filepath = self._walker[idx]
            y = os.path.split(audio_filepath)[1].split("-")[0]
            wav, sr = torchaudio.load(audio_filepath)
            wav = wav[0]
            assert sr == self.sample_rate
            if n_samples is not None:
                if len(wav) > n_samples:
                    start = random.randint(0, len(wav) - n_samples - 1)
                else:
                    start = 0
                    wav = torch.nn.functional.pad(wav, (0, (n_samples - len(wav))))
                wav = wav[start:start+n_samples]
            y = self.encode([y])
            return wav, y

        def get_random_sample_segment(self, segment_samples):
            idx = random.randint(0, self.__len__()-1)
            x, y = self.__getitem__(idx)
            x = x[0]
            if len(x) > segment_samples:
                start = random.randint(0, len(x) - segment_samples - 1)
            else:
                start = 0
                x = torch.nn.functional.pad(x, (0, (segment_samples - len(x))))
            return x[start:start+segment_samples], y


        def split_resample_urban(self, duration=10, fs=16000):
            path='UrbanSound8k'
            save_path = 'UrbanSound8k_%ds' % duration
            for type in ['train', 'test']:
                filepaths = get_list_of_files(path + type)
                for filepath in filepaths:
                    x, sr = torchaudio.load(filepath, normalize=True)
                    x = librosa.resample(x[0].numpy(), orig_sr=sr, target_sr=fs)
                    for index, i in enumerate(range(0, len(x)-int(duration*fs), int(fs*duration))):
                        filename = os.path.split(os.path.splitext(filepath)[0])[-1]
                        count = filename.split('_')[-1]
                        filename = filename.split('_')[0]
                        save_filepath = save_path + type + '\\' + filename + '-' + count + str(index) +'.wav'
                        torchaudio.save(save_filepath, torch.tensor(x[i:i+ int(duration*fs)][None, :]), fs, bits_per_sample=16)
                        print(save_filepath)
