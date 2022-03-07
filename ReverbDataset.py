from NoiseReal import NoiseReal
import random
from utils import rms
import matplotlib.pyplot as plt
from sklearn.utils import column_or_1d
import pathlib
import realrirs.datasets
from torchTools import SplitFrames
from scipy import signal
from torch.utils.data import Dataset
from LibriSpeech import LibriSpeech
import numpy as np
import torch
import torch.nn.functional as F
from analysis import reverberation_time
from helpers import create_folder, delete_folder
import os
from helpers import get_list_of_files
from utils import read_audio
from pedalboard import load_plugin
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from time import time
import torchaudio


class ReverbDataset(Dataset):
    def __init__(self, type='train', speech_subset_rate=1., rev_subset_rate =1.,
                 sample_window=1.0, sample_hop=0.5, noisy=False, snrs=None, cache_files=True, stimuli='speech'):
        super().__init__()
        self.type = type
        assert stimuli in ['speech', 'drums']
        if stimuli =='speech':
            self.speech_dataset = LibriSpeech(type=self.type)
        else:
            self.speech_dataset = DSD100(type=self.type, sources=[stimuli])

        if self.type == 'test':
            speech_subset_rate = 1.0

        self.fs = self.speech_dataset.sample_rate
        self.sample_window = sample_window
        self.sample_hop = sample_hop
        self.win_samples = int(self.fs * self.sample_window)
        self.hop_samples = int(self.fs * self.sample_hop)
        self.snrs = np.linspace(0., 60., num=61) if snrs == None else snrs
        self.noisy = noisy
        self.cache_files = cache_files
        self.cache_folder = 'E:\\FastDatasets\\temp_data'
        if self.noisy:
            self.noise_dataset = NoiseReal(type=self.type, datasets=['UrbanSound8k-Noise'])
            noise_ids = np.arange(self.noise_dataset.__len__())
        else:
            noise_ids = [0]
        # self.ir_dataset = realrirs.datasets.BUTDataset(
        #     pathlib.Path("D:\\PhD\\Datasets\\BUT_ReverbDB_rel_19_06_RIR-Only"))
        self.ir_dataset = AIRDataset('D:\\PhD\\Datasets\\AIR_1_4_16khz')
        self.ir_f = self.ir_dataset.list_files()
        self.ir_f = self.ir_f[:int(len(self.ir_f) * rev_subset_rate)]
        speech_ids = np.arange(self.speech_dataset.__len__())
        speech_ids = list(speech_ids) * int(np.ceil(speech_subset_rate))
        speech_ids = speech_ids[:int(len(speech_ids) * speech_subset_rate)]
        wet_blends = [np.random.uniform() for _ in range(len(speech_ids))]
        rt60s = [3 * reverberation_time(self.ir_dataset.__getitem__(ir_f)[0], self.fs, -5, -25, plot=False)
                 for ir_f in self.ir_f]
        ir_idxs = np.arange(len(self.ir_f))
        ir_idxs = [random.choice(ir_idxs) for i in range(len(speech_ids))]

        self.meta = [{'speech_id': speech_ids[i],
                      'wet_blend': wet_blends[i],
                      'ir_f': self.ir_f[ir_idxs[i]],
                      'rt60': rt60s[ir_idxs[i]],
                      'snr': random.choice(self.snrs),
                      'noise_id': random.choice(noise_ids)} for i in range(len(speech_ids))]

    def __len__(self):
        return self.meta.__len__()

    def __getitem__(self, idx):
        fpath = os.path.join(self.cache_folder, 'sample_%d.pt' % (idx))
        if self.cache_files:
            if os.path.exists(fpath):
                x, y = torch.load(fpath)
                return x, y

        x, speech_meta = self.speech_dataset.__getitem__(self.meta[idx]['speech_id'])
        rir = self.ir_dataset.__getitem__(self.meta[idx]['ir_f'])[0] # is sampled at 16kHz
        if self.noisy:
            noise, _ = self.noise_dataset.__getitem__(self.meta[idx]['noise_id'], n_samples=x.shape[-1])
            x = self.add_noise(x, noise, snr=self.meta[idx]['snr'])

        # wet blend to dB
        # is_blend_in_dB = False
        # if is_blend_in_dB:
        #     blend = 10 ** ((self.meta[idx]['wet_blend'] * 60 - 60) / 10.)
        # else:
        #     blend = self.meta[idx]['wet_blend']
        #     blend = 1. # * (blend > 0.75) + (0.75 >= blend >= 0.25) * 0.5

        rms_old = torch.sqrt(torch.mean(x ** 2))
        x = self.reverberate(x, rir, wet_blend=1.0)
        rms_new = torch.sqrt(torch.mean(x ** 2))
        y = (rms_new - rms_old) / rms_old
        # y = self.meta[idx]['rt60']

        # if y == None:
        #     import os
        #     os.remove(self.meta[idx]['ir_f'])
        #     print('Deleting', self.meta[idx]['ir_f'])
        if self.cache_files:
            if not os.path.exists(fpath):
                torch.save([x, y], fpath)

        return x, y

    def window_signal(self, x, win_samples, hop_samples):
        padding = ((x.shape[-1] // hop_samples) + int(x.shape[-1] % hop_samples > 0)) * hop_samples - x.shape[-1]
        x = torch.nn.functional.pad(x, (0, padding))
        y = [x[i:i+win_samples] for i in range(0, x.shape[-1], hop_samples)][:-1]
        return torch.stack(y)

    def add_noise(self, wav, noise, snr):
        pre_snr = torch.sqrt(torch.mean(wav ** 2)) / (torch.sqrt(torch.mean(noise ** 2)) + 1e-6)
        scale_factor = 10. ** (-1 * snr / 20.) * pre_snr
        return wav + scale_factor * noise

    def reverberate(self, wav, ir, wet_blend=0.5):
        wav = np.array(wav)
        # IR normalization
        ir = ir / np.amax(np.abs(ir))
        p_max = np.argmax(np.abs(ir))
        signal_rev = signal.fftconvolve(wav, ir, mode="full")
        # Normalization
        signal_rev = signal_rev / np.amax(np.abs(signal_rev)) * np.amax(np.abs(wav))
        # IR delay compensation. Trim reverberated signal (same length as clean sig)
        signal_rev = wet_blend * self.shift(signal_rev, -p_max)[0:wav.shape[0]] + wav * (1-wet_blend)
        return torch.tensor(signal_rev, dtype=torch.float32)

    def shift(self, xs, n):
        e = np.empty_like(xs)
        if n >= 0:
            e[:n] = 0.0
            e[n:] = xs[:-n]
        else:
            e[n:] = 0.0
            e[:n] = xs[-n:]
        return e

    def get_n_targets(self):
        return 1


class AIRDataset:
    def __init__(self, root):
        super(AIRDataset, self).__init__()
        fpaths = get_list_of_files(root)
        self.filepaths = [a for a in fpaths if '.wav' in a]

    def __len__(self):
        return len(self.filepaths)

    def list_files(self):
        return self.filepaths

    def __getitem__(self, idx):
        fs, x = read_audio(idx)
        return x, fs



class SimLoader(Dataset):
    def __init__(self, meta, cache_folder):
        super(SimLoader, self).__init__()
        self.meta = meta
        self.cache_folder = cache_folder

    def __getitem__(self, idx):
        fpath = os.path.join(self.cache_folder, 'sample_%d.pt' % (idx))
        if os.path.exists(fpath):
            x, y = torch.load(fpath)
            return x, y

    def __len__(self):
        return len(self.meta)

    def get_keys(self):
        return [*self.meta[0].keys()]


class SimReverbDataset(ReverbDataset):
    def __init__(self, params=None, param_size=10, type='train', speech_subset_rate=1., rev_subset_rate =1.,
                 sample_window=1.0, sample_hop=0.5, noisy=False, snrs=None, cache_files=True, stimuli='speech'):
        super().__init__(type=type, speech_subset_rate=speech_subset_rate, rev_subset_rate=rev_subset_rate,
                         sample_window=sample_window, sample_hop=sample_hop, noisy=noisy, snrs=snrs,
                         cache_files=cache_files, stimuli=stimuli)
        if self.noisy:
            self.noise_dataset = NoiseReal(type=self.type, datasets=['GTZAN'])
            noise_ids = np.arange(self.noise_dataset.__len__())
        else:
            noise_ids = [0]

        self.param_size = param_size
        if params is None:
            self.param_names = ['wet', 'size', 'delay', 'diffuse']
        else:
            self.param_names = params

        # self.vst = OrilRiver()
        self.vst = TALReverb4()

        # get all valid values
        param_values = [self.vst.decode_all(param_name, self.vst.vst.parameters[param_name].valid_values)
                        for param_name in self.param_names]
        # plt.hist(param_values[2])
        # plt.show()

        speech_ids = np.arange(self.speech_dataset.__len__())
        speech_ids = list(speech_ids) * int(np.ceil(speech_subset_rate))
        speech_ids = speech_ids[:int(len(speech_ids) * speech_subset_rate)]

        # select with pre-defined step
        param_values = [param_val[::len(param_val) // param_size] for param_val in param_values]

        self.vst_param_dict = {key: value for (key, value) in zip(self.param_names, param_values)}
        vst_parameter_grid = [{key: random.choice(value) for (key, value) in zip(self.param_names, param_values)} for i in range(len(speech_ids))]
        # vst_parameter_grid = list(ParameterGrid(self.vst_param_dict))
        # random.shuffle(vst_parameter_grid)
        # if len(vst_parameter_grid) < len(speech_ids):
        #     vst_parameter_grid = vst_parameter_grid * (len(speech_ids) // len(vst_parameter_grid) + 1)

        self.meta = [{'speech_id': speech_ids[i],
                      'snr': random.choice(self.snrs),
                      'noise_id': random.choice(noise_ids),
                        **vst_parameter_grid[i]
                      } for i in range(len(speech_ids))]

    def get_n_targets(self):
        return len(self.param_names)

    def generate_dataset(self):
        delete_folder(self.cache_folder)
        if self.cache_files:
            create_folder(self.cache_folder)

        for i in tqdm(range(self.__len__()), desc="Generate Dataset"):
            _ = self.__getitem__(i)

    def __getitem__(self, idx):
        fpath = os.path.join(self.cache_folder, 'sample_%d.pt' % (idx))
        if self.cache_files:
            if os.path.exists(fpath):
                x, y = torch.load(fpath)
                return x, y

        x, speech_meta = self.speech_dataset.__getitem__(self.meta[idx]['speech_id'])

        # set plugin parameters
        for attr in self.param_names:
            val = self.meta[idx][attr]
            self.vst.vst.__setattr__(attr, self.vst.encode(attr, val))

        # process signal
        x_rev = torch.tensor(self.vst(x))
        x = x_rev / torch.sqrt(torch.mean(x_rev ** 2)) * torch.sqrt(torch.mean(x ** 2))

        # reverberation is not added to the noise
        if self.noisy:
            noise, _ = self.noise_dataset.__getitem__(self.meta[idx]['noise_id'], n_samples=x.shape[-1])
            x = self.add_noise(x, noise, snr=self.meta[idx]['snr'])

        y = [self.meta[idx][attr] for attr in self.param_names]

        # energy-based SRR calculation
        # rms_old = torch.sqrt(torch.mean(x ** 2))
        # rms_new = torch.sqrt(torch.mean(x ** 2))
        # y = (rms_new - rms_old) / rms_old

        if self.cache_files:
            if not os.path.exists(fpath):
                torch.save([x, y], fpath)

        return x, y


class DSD100(Dataset):
    def __init__(self, root='E:\\FastDatasets\\DSD100\\Sources', type='train', sources=None):
        super(DSD100, self).__init__()
        assert type in ['train', 'test']
        tfolder = 'Dev' if type == 'train' else 'Test'
        self.sources = sources if sources is not None else ['drums', 'bass', 'vocals']
        fpaths = get_list_of_files(os.path.join(root, tfolder))
        self.sample_rate = 16000
        self.filepaths = [a for a in fpaths if '.wav' in a ]
        self.filepaths = [a for a in fpaths if any([s in a for s in sources])]

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.filepaths[idx])
        wav = wav[0]
        return wav, os.path.split(self.filepaths[idx])[-1][:-4]

class TALReverb4:
    def __init__(self):
        super(TALReverb4, self).__init__()
        self.vst = load_plugin("vst\TAL-Reverb-4.vst3")
        self.vst.dry = 50. # %
        self.vst.diffuse = 0.2
        self.vst.modulation_depth = 0.0
        self.vst.modulation_rate = 10. # %
        self.param_names = ['size', 'wet', 'diffuse'] #, 'delay', 'modulation_rate', 'modulation_depth']

    def get_param_names(self):
        return self.param_names

    def __call__(self, x):
        # convert to stereo
        if x.dim() == 1:
            x = np.stack([x, x])
        y = self.vst(x, 16000)
        # convert to mono
        y = y[:, 0]  if y.shape[-1] == 2 else y[0, :]
        return y

    def encode(self, key, value):
        if key == 'delay':
            param_max = 1.0 # sec
            param_min = 0.0
            value = value * (param_max - param_min) + param_min
            value = '%.4f s' % value
        else:
            param_max = self.vst.parameters[key].max_value
            param_min = self.vst.parameters[key].min_value
            if key == 'high_cut' or key == 'low_cut':
                param_max = np.log(param_max + 100.)
                param_min = np.log(param_min + 100.)
                value = value * (param_max - param_min) + param_min
                value =  np.exp(value) - 100.
            else:
                value = value * (param_max - param_min) + param_min

        return value

    def decode_all(self, key, value):
        return [self.decode(key, v) for v in value]

    def decode(self, key, value):
        if key == 'delay':
            value = float(value.split()[0]) if isinstance(value, str) else value
            param_max = 1.0 # sec
            param_min = 0.0
        else:
            param_max = self.vst.parameters[key].max_value
            param_min = self.vst.parameters[key].min_value
            if key == 'high_cut' or key == 'low_cut':
                value = np.log(value + 100.)
                param_max = np.log(param_max + 100.)
                param_min = np.log(param_min + 100.)
        value = (value - param_min) / (param_max - param_min)
        return value


class OrilRiver:
    def __init__(self):
        super().__init__()
        self.vst = load_plugin("vst\OrilRiver.vst3")
        self.vst.predelay_ms = 10.0
        self.vst.e_r_variation = 'Variation 1'
        self.vst.damp_hz = 8000.
        self.vst.dry_db = '-6.00'

        self.param_names = ['wet_db', 'reverb_db', 'early_reflections_db'] #
        # ['decay_time_sec', 'pre_delay_ms', 'room_size', 'diffusion', 'damp_intensity']

    def __call__(self, x):
        # convert to stereo
        if x.dim() == 1:
            x = np.stack([x, x])
        y = self.vst(x, 16000)
        # convert to mono
        y = y[:, 0]  if y.shape[-1] == 2 else y[0, :]
        return y

    def get_param_names(self):
        return self.param_names

    def encode(self, key, value):
        if key in ['dry_db', 'wet_db', 'reverb_db', 'early_reflections_db']:
            param_max = 0.0 # sec
            param_min = -60.0
            value = value ** (1/6)
            value = value * (param_max - param_min) + param_min
            value = '%.2f' % value
        elif key == 'decay_time_sec':
            value = value ** (2)
            param_max = self.vst.parameters[key].max_value ** (1 / 2)
            param_min = self.vst.parameters[key].min_value ** (1 / 2)
            value = value * (param_max - param_min) + param_min
        else:
            param_max = self.vst.parameters[key].max_value
            param_min = self.vst.parameters[key].min_value
            value = value * (param_max - param_min) + param_min

        return value

    def decode_all(self, key, value):
        return [self.decode(key, v) for v in value]

    def decode(self, key, value):
        if key in ['dry_db', 'wet_db', 'reverb_db', 'early_reflections_db']:
            if value == '-oo':
                value = '-60.00'
            value = float(value.split()[0]) if isinstance(value, str) else value
            param_max = 0.0 # dB
            param_min = -60 # dB
            value = (value - param_min) / (param_max - param_min)
            value = value ** 6
        elif key == 'decay_time_sec':
            value = value ** (1 / 2)
            param_max = self.vst.parameters[key].max_value ** (1 / 2)
            param_min = self.vst.parameters[key].min_value ** (1 / 2)
            value = (value - param_min) / (param_max - param_min)
        else:
            param_max = self.vst.parameters[key].max_value
            param_min = self.vst.parameters[key].min_value
            value = (value - param_min) / (param_max - param_min)
        return value