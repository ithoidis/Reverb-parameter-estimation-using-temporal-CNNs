import os
import warnings
warnings.simplefilter("ignore", UserWarning)
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torchaudio
import numpy as np
from utils import *
from speech_recognition_utils import *
from speaker_recognition_models import *
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from ReverbDataset import *
from torchTools import EarlyStopping
from tqdm import tqdm
import subprocess
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import time, shutil, sys
from helpers import create_folder, Logger
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from sklearn.manifold import TSNE
plt.rcParams['font.family'] = "serif"


class RevNet2D(nn.Module):
    def __init__(self, n_targets=1, rnn_dim=256, n_mels=64, dropout=0.3, use_attention=False, frontend='dB', fs=16000):
        super(RevNet2D, self).__init__()
        self.use_attention = use_attention
        self.fs = fs
        self.batch_size = 40
        self.n_targets = n_targets
        assert frontend in ['dB', 'persa+', 'persa', 'pcen']
        if frontend == 'dB':
            self.frontend = torchaudio.transforms.AmplitudeToDB()
        elif frontend == 'persa+':
            self.frontend = Persa_plus(snr=9, n_mels=n_mels, n_fft=800)
        elif frontend == 'pcen':
            self.frontend = PCEN(trainable=True)
        elif frontend == 'persa':
            self.frontend = Persa()
        else:
            self.frontend = None

        self.mel_specgram = torchaudio.transforms.MelSpectrogram(self.fs, n_fft=400, hop_length=200,
                                                                 n_mels=n_mels, normalized=False).cuda() # (channel, n_mels, time)

        self.conv1 = torch.nn.Conv2d(1, 256, kernel_size=(5, 5), padding=5//2)
        self.norm1 = torch.nn.BatchNorm2d(256)
        self.drop1 = torch.nn.Dropout(dropout)
        self.pool1 = torch.nn.MaxPool2d((2, 2))

        self.conv2 = torch.nn.Conv2d(256, 64, kernel_size=(3, 3), padding=3//2)
        self.norm2 = torch.nn.BatchNorm2d(64)
        self.drop2 = torch.nn.Dropout(dropout)
        self.pool2 = torch.nn.MaxPool2d((2, 2))

        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=(3, 3), padding=3//2)
        self.norm3 = torch.nn.BatchNorm2d(128) #torch.nn.GroupNorm(1, 128),
        self.drop3 = torch.nn.Dropout(dropout)
        self.pool3 = torch.nn.MaxPool2d((2, 2))

        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=(3, 3), padding=3 // 2)
        self.norm4 = torch.nn.BatchNorm2d(256)
        self.drop4 = torch.nn.Dropout(dropout)
        self.pool4 = torch.nn.MaxPool2d((2, 2))

        self.lin5 = nn.ModuleList([nn.Linear(rnn_dim, rnn_dim) for i in range(self.n_targets)])
        self.norm5 = nn.ModuleList([torch.nn.BatchNorm1d(rnn_dim) for i in range(self.n_targets)])
        self.drop5 = nn.ModuleList([nn.Dropout(dropout) for i in range(self.n_targets)])
        self.lin6 = nn.ModuleList([nn.Linear(rnn_dim, rnn_dim//2) for i in range(self.n_targets)])
        self.norm6 = nn.ModuleList([torch.nn.BatchNorm1d(rnn_dim//2) for i in range(self.n_targets)])
        self.drop6 = nn.ModuleList([nn.Dropout(dropout) for i in range(self.n_targets)])
        self.linout = nn.ModuleList([nn.Linear(rnn_dim//2, 1) for i in range(self.n_targets)])

    def forward(self, x):
        if x.dim() == 1: x = x.unsqueeze(0)
        # to reduce memory
        x = self.mel_specgram(x)
        x = self.frontend(x)
        x = x.unsqueeze(1)
        x = self.drop1(self.pool1(self.norm1(torch.relu(self.conv1(x)))))
        x = self.drop2(self.pool2(self.norm2(torch.relu(self.conv2(x)))))
        x = self.drop3(self.pool3(self.norm3(torch.relu(self.conv3(x)))))
        x = self.drop4(self.pool4(self.norm4(torch.relu(self.conv4(x)))))

        # (batch, channels, time, freq)
        x = torch.mean(x, dim=-1)
        # (batch, channels, time)
        # take mean onyl of segments that are not padded
        x = torch.mean(x, dim=-1)
        emb = x
        y = []
        for i in range(self.n_targets):
            x0 = self.drop5[i](torch.relu(self.lin5[i](x)))
            x0 = self.drop6[i](torch.relu(self.lin6[i](x0)))
            x0 = self.linout[i](x0)
            y.append(x0)
        x = torch.cat(y, dim=-1)
        return x, emb


class RevNetGLU(nn.Module):
    def __init__(self, n_targets=1, embedding_dim=256, dropout=0.3, model_capacity='medium', fs=16000):
        super().__init__()

        self.batch_size = 16
        self.fs = fs
        self.n_targets = n_targets
        self.dropout = dropout
        self.embedding_dim = embedding_dim

        capacity_multiplier = {
            'tiny': 4, 'small': 8, 'light': 12, 'medium': 16, 'large': 24, 'full': 32}[model_capacity]
        self.layers = [1, 2, 3, 4, 5, 6]
        self.channels = [n * capacity_multiplier for n in [32, 4, 4, 4, 8, 16]]
        self.channels[-1] = self.embedding_dim * self.n_targets
        self.kernel_sizes = [512, 64, 64, 64, 64, 64]
        self.strides = [4, 1, 1, 1, 1, 1]
        self.pool = 4

        self.conv1 = nn.Conv1d(1, self.channels[0] * 2, self.kernel_sizes[0], self.strides[0], padding=self.kernel_sizes[0]//2)
        self.act1 = nn.GLU(dim=-2)
        self.bn1 = nn.BatchNorm1d(self.channels[0])
        self.pool1 = nn.MaxPool1d(self.pool)
        self.drop1 = nn.Dropout(self.dropout)

        self.conv2 = nn.Conv1d(self.channels[0], self.channels[1]* 2, self.kernel_sizes[1], self.strides[1], padding=self.kernel_sizes[1]//2)
        self.act2 = nn.GLU(dim=-2)
        self.bn2 = nn.BatchNorm1d(self.channels[1])
        self.pool2 = nn.MaxPool1d(self.pool)
        self.drop2 = nn.Dropout(self.dropout)

        self.conv3 = nn.Conv1d(self.channels[1], self.channels[2]* 2, self.kernel_sizes[2], self.strides[2], padding=self.kernel_sizes[2] // 2)
        self.act3 = nn.GLU(dim=-2)
        self.bn3 = nn.BatchNorm1d(self.channels[2])
        self.pool3 = nn.MaxPool1d(self.pool)
        self.drop3 = nn.Dropout(self.dropout)

        self.conv4 = nn.Conv1d(self.channels[2], self.channels[3]* 2, self.kernel_sizes[3], self.strides[3], padding=self.kernel_sizes[3] // 2)
        self.act4 = nn.GLU(dim=-2)
        self.bn4 = nn.BatchNorm1d(self.channels[3])
        self.pool4 = nn.MaxPool1d(self.pool)
        self.drop4 = nn.Dropout(self.dropout)

        self.conv5 = nn.Conv1d(self.channels[3], self.channels[4]* 2, self.kernel_sizes[4], self.strides[4], padding=self.kernel_sizes[4] // 2)
        self.act5 = nn.GLU(dim=-2)
        self.bn5 = nn.BatchNorm1d(self.channels[4])
        self.pool5 = nn.MaxPool1d(self.pool)
        self.drop5 = nn.Dropout(self.dropout)

        self.conv6 = nn.Conv1d(self.channels[4], self.channels[5]* 2, self.kernel_sizes[5], self.strides[5], padding=self.kernel_sizes[5] // 2)
        self.act6 = nn.GLU(dim=-2)
        self.bn6 = nn.BatchNorm1d(self.channels[5])
        self.pool6 = nn.MaxPool1d(self.pool)
        self.drop6 = nn.Dropout(self.dropout)

        self.lin5o = nn.ModuleList([nn.Linear(self.channels[5], self.embedding_dim) for i in range(self.n_targets)])
        self.norm5o = nn.ModuleList([torch.nn.BatchNorm1d(self.embedding_dim) for i in range(self.n_targets)])
        self.drop5o = nn.ModuleList([nn.Dropout(dropout) for i in range(self.n_targets)])
        self.lin6o = nn.ModuleList([nn.Linear(self.embedding_dim, self.embedding_dim//2) for i in range(self.n_targets)])
        self.norm6o = nn.ModuleList([torch.nn.BatchNorm1d(self.embedding_dim//2) for i in range(self.n_targets)])
        self.drop6o = nn.ModuleList([nn.Dropout(dropout) for i in range(self.n_targets)])
        self.linout = nn.ModuleList([nn.Linear(self.embedding_dim//2, 1) for i in range(self.n_targets)])


    def forward(self, x, x_len=None):
        if x.dim() == 1: x = x.unsqueeze(0)
        if x.dim() == 2: x = x.unsqueeze(1)

        y, emb = self.nn_forward(x, x_len=x_len)

        return y, emb


    def nn_forward(self, x, x_len=None):
        if x_len is None:
            x_len = [x.shape[-1]] * x.shape[0]
        x_len_padded = x.shape[-1]

        x = self.conv1(x)
        x = self.act1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.conv3(x)
        x = self.act3(x)
        x = self.bn3(x)
        x = self.pool3(x)
        x = self.drop3(x)

        x = self.conv4(x)
        x = self.act4(x)
        x = self.bn4(x)
        x = self.pool4(x)
        x = self.drop4(x)

        x = self.conv5(x)
        x = self.act5(x)
        x = self.bn5(x)
        x = self.pool5(x)
        x = self.drop5(x)

        x = self.conv6(x)
        x = self.act6(x)
        x = self.bn6(x)
        x = self.pool6(x)
        x = self.drop6(x)

        # (batch, channels, time)
        # mask padded windows from average pooling
        x_per = [int(x.shape[-1] * (xl / x_len_padded)) for xl in x_len]
        x = torch.cat([torch.mean(x[j, :, :x_per[j]], dim=-1).unsqueeze(0) for j in range(x.shape[0])], dim=0)
        emb = x
        y = []
        for i in range(self.n_targets):
            x0 = self.drop5o[i](torch.relu(self.lin5o[i](x)))
            x0 = self.drop6o[i](torch.relu(self.lin6o[i](x0)))
            x0 = self.linout[i](x0)
            y.append(x0)
        x = torch.cat(y, dim=-1)

        return x, emb


    def frame(self, x, win_samples, hop_samples):
        padding = ((x.shape[-1] // hop_samples) +  1 * (x.shape[-1] % hop_samples > 0)) * hop_samples - x.shape[-1]
        x = F.pad(x, (0, padding))
        y = x.unfold(dimension=-1, size=win_samples, step=hop_samples)
        y = y[:, 0] # unfold introduced new dim
        return y


class RevNet(nn.Module):
    def __init__(self, n_targets=1, embedding_dim=128, dropout=0.3, fs=16000):
        super(RevNet, self).__init__()
        self.fs = fs
        self.batch_size = 12
        self.n_targets = n_targets

        self.channels = [512, 128, 128, 256, embedding_dim * self.n_targets]
        self.pool = 4
        self.embedding_dim = self.channels[-1]
        self.conv_in = nn.Conv1d(1, self.channels[0], kernel_size=81, stride=4, padding= 81 // 2)
        self.norm_in = nn.GroupNorm(1, self.channels[0]) #ChannelWiseLayerNorm(self.channels[0])

        self.conv11 = nn.Conv1d(self.channels[0], self.channels[1], 3, padding=1)
        self.norm11 = nn.GroupNorm(1, self.channels[1])
        self.conv12 = nn.Conv1d(self.channels[1], self.channels[1], 3, padding=1)
        self.norm12 = nn.GroupNorm(1, self.channels[1])
        self.conv13 = nn.Conv1d(self.channels[1], self.channels[1], 3, padding=1)
        self.norm13 = nn.GroupNorm(1, self.channels[1])
        self.conv14 = nn.Conv1d(self.channels[1], self.channels[1], 3, padding=1)
        self.norm14 = nn.GroupNorm(1, self.channels[1])
        self.pool1 = nn.MaxPool1d(self.pool)

        self.conv21 = nn.Conv1d(self.channels[1], self.channels[2], 3, padding=1)
        self.norm21 = nn.GroupNorm(1, self.channels[2])
        self.conv22 = nn.Conv1d(self.channels[2], self.channels[2], 3, padding=1)
        self.norm22 = nn.GroupNorm(1, self.channels[2])
        self.conv23 = nn.Conv1d(self.channels[2], self.channels[2], 3, padding=1)
        self.norm23 = nn.GroupNorm(1, self.channels[2])
        self.conv24 = nn.Conv1d(self.channels[2], self.channels[2], 3, padding=1)
        self.norm24 = nn.GroupNorm(1, self.channels[2])
        self.pool2 = nn.MaxPool1d(self.pool)

        self.conv31 = nn.Conv1d(self.channels[2], self.channels[3], 3, padding=1)
        self.norm31 = nn.GroupNorm(1, self.channels[3])
        self.conv32 = nn.Conv1d(self.channels[3], self.channels[3], 3, padding=1)
        self.norm32 = nn.GroupNorm(1, self.channels[3])
        self.conv33 = nn.Conv1d(self.channels[3], self.channels[3], 3, padding=1)
        self.norm33 = nn.GroupNorm(1, self.channels[3])
        self.conv34 = nn.Conv1d(self.channels[3], self.channels[3], 3, padding=1)
        self.norm34 = nn.GroupNorm(1, self.channels[3])
        self.pool3 = nn.MaxPool1d(self.pool)

        self.conv41 = nn.Conv1d(self.channels[3], self.channels[4], 3, padding=1)
        self.norm41 = nn.GroupNorm(1, self.channels[4])
        self.conv42 = nn.Conv1d(self.channels[4], self.channels[4], 3, padding=1)
        self.norm42 = nn.GroupNorm(1, self.channels[4])
        self.conv43 = nn.Conv1d(self.channels[4], self.channels[4], 3, padding=1)
        self.norm43 = nn.GroupNorm(1, self.channels[4])
        self.conv44 = nn.Conv1d(self.channels[4], self.channels[4], 3, padding=1)
        self.norm44 = nn.GroupNorm(1, self.channels[4])
        self.pool4 = nn.MaxPool1d(self.pool)

        self.lin5 = nn.ModuleList([nn.Linear(self.channels[4], embedding_dim) for i in range(self.n_targets)])
        self.norm5 = nn.ModuleList([torch.nn.BatchNorm1d(embedding_dim) for i in range(self.n_targets)])
        self.drop5 = nn.ModuleList([nn.Dropout(dropout) for i in range(self.n_targets)])
        self.lin6 = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim//2) for i in range(self.n_targets)])
        self.norm6 = nn.ModuleList([torch.nn.BatchNorm1d(embedding_dim//2) for i in range(self.n_targets)])
        self.drop6 = nn.ModuleList([nn.Dropout(dropout) for i in range(self.n_targets)])
        self.linout = nn.ModuleList([nn.Linear(embedding_dim//2, 1) for i in range(self.n_targets)])

    def forward(self, x, x_len=None):
        if x.dim() == 1: x = x.unsqueeze(0)
        if x.dim() == 2: x = x.unsqueeze(1)
        # audio framing with
        # x = self.frame(x, win_samples=self.win_samples, hop_samples=self.hop_samples)
        # treat frames as different samples. move to batch dimension
        # batch_size, n_frames = x.shape[0], x.shape[1]
        # x = x.reshape(x.shape[0] * x.shape[1], 1, -1)
        # process audio
        y, emb = self.nn_forward(x, x_len=x_len)

        # de-frame
        # y = y.reshape(batch_size, n_frames, self.n_targets)
        # y = torch.mean(y, dim=1)
        # y = torch.max_pool1d(y.transpose(1, -1), kernel_size=int(n_frames))[:, :, 0]

        return y, emb


    def nn_forward(self, x, x_len=None):
        if x_len is None:
            x_len = [x.shape[-1]] * x.shape[0]
        x_len_padded = x.shape[-1]
        x = self.conv_in(x)
        x = torch.relu(x)
        x = self.norm_in(x)
        x = self.conv11(x)
        x = torch.relu(x)
        self.norm_ = self.norm11(x)
        x = self.norm_
        x = self.conv12(x)
        x = torch.relu(x)
        x = self.norm12(x)
        x = self.conv13(x)
        x = torch.relu(x)
        x = self.norm13(x)
        x = self.conv14(x)
        x = torch.relu(x)
        x = self.norm14(x)
        x = self.pool1(x)

        x = self.conv21(x)
        x = torch.relu(x)
        x = self.norm21(x)
        x = self.conv22(x)
        x = torch.relu(x)
        x = self.norm22(x)
        x = self.conv23(x)
        x = torch.relu(x)
        x = self.norm23(x)
        x = self.conv24(x)
        x = torch.relu(x)
        x = self.norm24(x)
        x = self.pool2(x)

        x = self.conv31(x)
        x = torch.relu(x)
        x = self.norm31(x)
        x = self.conv32(x)
        x = torch.relu(x)
        x = self.norm32(x)
        x = self.conv33(x)
        x = torch.relu(x)
        x = self.norm33(x)
        x = self.conv34(x)
        x = torch.relu(x)
        x = self.norm34(x)
        x = self.pool3(x)

        x = self.conv41(x)
        x = torch.relu(x)
        x = self.norm41(x)
        x = self.conv42(x)
        x = torch.relu(x)
        x = self.norm42(x)
        x = self.conv43(x)
        x = torch.relu(x)
        x = self.norm43(x)
        x = self.conv44(x)
        x = torch.relu(x)
        x = self.norm44(x)
        x = self.pool4(x)

        # (batch, channels, time)
        # mask padded windows from average pooling
        x_per = [int(x.shape[-1] * (xl / x_len_padded)) for xl in x_len]
        x = torch.cat([torch.mean(x[j, :, :x_per[j]], dim=-1).unsqueeze(0) for j in range(x.shape[0])], dim=0)
        emb = x
        y = []
        for i in range(self.n_targets):
            x0 = self.drop5[i](torch.relu(self.lin5[i](x)))
            x0 = self.drop6[i](torch.relu(self.lin6[i](x0)))
            x0 = self.linout[i](x0)
            y.append(x0)
        x = torch.cat(y, dim=-1)

        return x, emb


    def frame(self, x, win_samples, hop_samples):
        padding = ((x.shape[-1] // hop_samples) +  1 * (x.shape[-1] % hop_samples > 0)) * hop_samples - x.shape[-1]
        x = F.pad(x, (0, padding))
        y = x.unfold(dimension=-1, size=win_samples, step=hop_samples)
        y = y[:, 0] # unfold introduced new dim
        return y


def pad_collate(batch):
    (xx, yy) = zip(*batch)
    xx_pad = torch.nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0.)
    xx_len = [len(x) for x in xx]
    # trim audio to reduce memory usage
    xx_pad = xx_pad[..., :int(16000 * sample_len)]
    return xx_pad, xx_len, torch.tensor(yy)

class SRRLoss(torch.nn.Module):
    def __init__(self):
        super(SRRLoss, self).__init__()
        self.l1loss = torch.nn.HuberLoss()

    def forward(self, y_hat, y):
        loss1 = self.l1loss(torch.exp(y_hat), torch.exp(y))
        # loss2 = 1 - torch.mean(self.cosine(y_hat, y))
        # loss = torch.sqrt(torch.mean(torch.square(torch.cos((2 * math.pi * 2/3) * y_hat) - torch.cos((2 * math.pi *2/3) * y))))

        # pearson correlation
        vy_hat = y_hat - torch.mean(y_hat)
        vy = y - torch.mean(y)
        loss2 = 1 - torch.sum(vy_hat * vy) / (torch.sqrt(torch.sum(vy_hat ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

        return loss1 + 0.5 * loss2

class VentagliaLoss(torch.nn.Module):
    def __init__(self):
        super(VentagliaLoss, self).__init__()
        self.l1loss = torch.nn.L1Loss()

    def forward(self, y_hat, y):
        loss = torch.mean(self.l1loss(y_hat, y) + 0.1 * (y_hat - y) ** 2) - torch.mean(torch.sqrt((y_hat - torch.mean(y)) ** 2))
        return  loss


class MSE_vrysis(torch.nn.Module):
    def __init__(self):
        super(MSE_vrysis, self).__init__()

    def forward(self, y_hat, y):
        return torch.mean((y_hat-y)**2 / (torch.relu(-(y_hat**2 - y_hat))) + 0.001)


class Trainer:
    def __init__(self, val_size=0.2, stimuli='speech', speech_subset_rate=2., param_size=30, sample_len=6., ):
        self.folder = create_folder('models\\' + time.strftime("%m-%d") + 'Rev_')
        shutil.copy('Reverb.py', self.folder + '/SRRestimation.txt')
        shutil.copy('ReverbDataset.py', self.folder + '/SRRDataset.txt')
        sys.stdout = Logger(self.folder + '/console.txt')
        self.writer = SummaryWriter(comment=os.path.split(self.folder)[-1])
        self.val_size = val_size
        self.param_size = param_size
        self.sample_len = sample_len
        self.speech_subset_rate = speech_subset_rate
        self.stimuli = stimuli
        self.vst_params = ['size', 'wet', 'diffuse'] # ['wet_db', 'reverb_db', 'decay_time_sec', 'room_size']
        self.dataset = SimReverbDataset(params=self.vst_params, param_size=self.param_size, type='train',
                                        speech_subset_rate=self.speech_subset_rate, noisy=False, stimuli=self.stimuli)
        # self.model = RevNet2D(n_targets=len(self.vst_params))
        # self.model = RevNet(n_targets=len(self.vst_params))
        self.model = RevNetGLU(n_targets=len(self.vst_params))

        self.criterion = nn.L1Loss().cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

        self.batch_size = self.model.batch_size
        self.fs = self.model.fs
        self.test_embeddings = []
        self.test_df = pd.DataFrame({})
        self.train_errors, self.valid_errors = [], []
        self.early_stopping = EarlyStopping(patience=20, delta=1e-3, path=os.path.join(self.folder, 'model_checkpoint.pt'),
                                       verbose=False)
        train_val_ids = np.arange(self.dataset.__len__())
        _, _, self.train_ids, self.val_ids = train_test_split(train_val_ids, train_val_ids, test_size=self.val_size,
                                                    shuffle=False)


    def generate_sim_dataset(self):
        # generate dataset offline because of VST parallelization problem
        self.dataset.generate_dataset()
        np.save('temp/meta.npy', self.dataset.meta)

    def train(self, epochs=100):
        if os.path.exists('temp/meta.npy'):
            meta = np.load('temp/meta.npy', allow_pickle=True)
        else:
            return

        dataset = SimLoader(meta=meta, cache_folder='E:\\FastDatasets\\temp_data')
        train_dataset = torch.utils.data.Subset(dataset, self.train_ids)
        val_dataset = torch.utils.data.Subset(dataset, self.val_ids)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True,
                                  num_workers=6, drop_last=True, collate_fn=pad_collate)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True,
                                num_workers=8, drop_last=False, collate_fn=pad_collate)
        print('Folder: ', self.folder)
        print('Fold batches:', train_loader.__len__(), val_loader.__len__())
        print('Model:', self.model.__str__())
        print('Speech samples:', len(dataset))
        assert torch.cuda.is_available()

        self.model.cuda()
        summary(self.model, input_size=(int(5. * self.fs),))
        self.writer.add_graph(self.model, (torch.randn(self.batch_size, int(self.fs * 5.)).cuda()))

        print(time.strftime("%H:%M:%S"), 'Training...')
        for epoch in range(1, epochs+1):
            t0 = time.time()
            self.model.train()
            running_loss, running_dist = 0., 0.
            target_labels, predicted_labels = [], []
            for i, (x, x_len, y) in tqdm(enumerate(train_loader), total=len(train_loader), desc='Epoch %d' % epoch):
                # overfit on batch
                # if epoch == 1 and i==0:
                #     x1, y1 = x, y
                # x, y = x1, y1
                # for j in range(x.shape[0]):
                #     print(x[j].shape, 'Label:\n', self.vst_params, '\n', y[j])
                #     plt.plot(x[j])
                #     plt.show()
                #     play_audio(x[j].numpy(), fs=16000)
                self.optimizer.zero_grad()
                y_hat, embedding = self.model(x.requires_grad_(True).cuda(), x_len)  # (batch, time)
                loss = self.criterion(y_hat, y.cuda())
                running_loss += loss.item()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2)
                self.optimizer.step()
                # distance per parameter
                running_dist += torch.mean(torch.abs(y_hat.cpu() - y)).detach()
                predicted_labels.extend(y_hat.clone().cpu().detach().numpy())
                target_labels.extend(y.clone().cpu().detach().numpy())

            train_dist = running_dist / (i + 1)
            train_loss = running_loss / (i + 1)

            df_pred = {name+' (Predicted)': value for (name, value) in zip(self.vst_params, np.array(predicted_labels).transpose().tolist())}
            df_true = {name+' (Target)': value for (name, value) in zip(self.vst_params, np.array(target_labels).transpose().tolist())}
            df = pd.DataFrame({**df_pred, **df_true})
            self.train_errors.append([np.mean(np.abs(df[name + ' (Target)'] - df[name + ' (Predicted)']))
                                      for name in self.vst_params])

            for name in self.vst_params:
                fig, ax = plt.subplots(figsize=(8, 6))
                y_label = name + ' (Predicted)'
                x_label = name + ' (Target)'
                ax.plot([np.amin(df[x_label]), np.amax(df[x_label])], [np.amin(df[x_label]), np.amax(df[x_label])], ls='--', color='r')
                sns.scatterplot(data=df, y=y_label, x=x_label, ax=ax)
                sns.lineplot(data=df, y=y_label, x=x_label, ax=ax, color='g')
                # sns.kdeplot(y='wet (Predicted)', x='wet (Target)', data=df, levels=5, linewidths=1, ax=ax)
                self.writer.add_figure(name+'/Train', fig, global_step=epoch)

            # validation
            with torch.no_grad():
                self.model.eval()
                running_loss, running_dist = 0., 0.
                target_labels, predicted_labels = [], []
                val_embeddings = []
                for i, (x, x_len, y) in tqdm(enumerate(val_loader), desc='Validation', total=len(val_loader)):
                    y_hat, embedding = self.model(x.cuda(), x_len)  # (batch, time)
                    loss = self.criterion(y_hat, y.cuda())
                    running_loss += loss.item()
                    running_dist += torch.mean(torch.abs(y_hat.cpu() - y)).detach()
                    predicted_labels.extend(y_hat.clone().cpu().detach().numpy())
                    target_labels.extend(y.clone().cpu().detach().numpy())
                    val_embeddings.append(embedding)

            r_sq = np.mean([r_squared(np.array(target_labels)[:, i], np.array(predicted_labels)[:, i])
                            for i in range(len(self.vst_params))])
            valid_loss = running_loss / (i + 1)
            valid_dist = running_dist / (i + 1)

            print('%ds - epoch: %s%d/%d - loss: %.4f - val_loss: %.4f - train_dist: %.3f - val_dist: %.3f - val_R²: %.2f' % (
                                                                        int(time.time() - t0), ' ' if epoch < 10 else '', epoch, epochs,
                                                                        train_loss, valid_loss, train_dist, valid_dist, r_sq))
            df_pred = {name + ' (Predicted)': value for (name, value) in
                       zip(self.vst_params, np.array(predicted_labels).transpose().tolist())}
            df_true = {name + ' (Target)': value for (name, value) in
                       zip(self.vst_params, np.array(target_labels).transpose().tolist())}
            df = pd.DataFrame({**df_pred, **df_true})

            self.valid_errors.append([np.mean(np.abs(df[name + ' (Target)'] - df[name + ' (Predicted)']))
                                      for name in self.vst_params])

            for name in self.vst_params:
                fig, ax = plt.subplots(figsize=(8, 6))
                y_label = name + ' (Predicted)'
                x_label = name + ' (Target)'
                ax.plot([np.amin(df[x_label]), np.amax(df[x_label])], [np.amin(df[x_label]), np.amax(df[x_label])],
                        ls='--', color='r')
                sns.scatterplot(data=df, y=y_label, x=x_label, ax=ax)
                sns.lineplot(data=df, y=y_label, x=x_label, ax=ax, color='g')
                self.writer.add_figure(name + '/Valid', fig, global_step=epoch)

            # self.writer.add_histogram('Valid/Error distribution', predicted_labels - target_labels, epoch)

            self.writer.add_embedding(torch.cat(val_embeddings), list(target_labels), global_step=epoch, tag='%s_%d'%(self.folder, epoch))
            self.writer.add_scalar('SRRLoss/train', train_loss, epoch)
            self.writer.add_scalar('SRRLoss/valid', valid_loss, epoch)
            self.writer.add_scalar('SRRDist/train', train_dist, epoch)
            self.writer.add_scalar('SRRDist/valid', valid_dist, epoch)
            self.writer.add_scalar('R2/valid', r_sq, epoch)

            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping!")
                break

        # load the last checkpoint with the best model
        self.model = torch.load(self.folder + '\\model_checkpoint.pt')
        torch.save(self.model, self.folder + '\\model.pt')
        self.save()
        self.writer.close()


    def test(self):
        print(time.strftime("%H:%M:%S"), 'Evaluating... ')
        t0 = time.time()

        test_dataset = SimReverbDataset(params=self.vst_params, param_size=self.param_size, cache_files=False, type='test',
                                        noisy=False, stimuli=self.stimuli)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=False,
                                 num_workers=0, drop_last=False, collate_fn=pad_collate)
        # test
        with torch.no_grad():
            self.model.eval()
            running_loss, running_dist = 0., 0.
            target_labels, predicted_labels = [], []
            self.test_embeddings = []
            for i, (x, x_len, y) in tqdm(enumerate(test_loader), desc='Test', total=len(test_loader)):
                y_hat, embedding = self.model(x.cuda(), x_len)  # (batch, time)
                loss = self.criterion(y_hat, y.cuda())
                running_loss += loss.item()
                running_dist += torch.mean(torch.abs(y_hat.cpu() - y)).detach()
                predicted_labels.extend(y_hat.clone().cpu().detach().numpy())
                target_labels.extend(y.clone().cpu().detach().numpy())
                self.test_embeddings.extend(embedding.cpu().detach().numpy())

        r_sq_full = [r_squared(np.array(target_labels)[:, i], np.array(predicted_labels)[:, i])
                        for i in range(len(self.vst_params))]
        mae_full = [np.mean(np.abs(np.array(target_labels)[:, i] - np.array(predicted_labels)[:, i]))
                     for i in range(len(self.vst_params))]

        df = pd.DataFrame(np.stack([mae_full, r_sq_full]).transpose(), columns=['MAE', 'R²'])
        df['Parameter'] = self.vst_params
        df.insert(0, 'Parameter', df.pop('Parameter'))
        print('Test Results:')
        print(df.to_string())
        df.to_csv(os.path.join(self.folder, 'test_results.csv'))

        r_sq = np.mean(r_sq_full)
        test_loss = running_loss / (i + 1)
        test_dist = running_dist / (i + 1)

        df_pred = {name + ' (Predicted)': value for (name, value) in
                   zip(self.vst_params, np.array(predicted_labels).transpose().tolist())}
        df_true = {name + ' (Target)': value for (name, value) in
                   zip(self.vst_params, np.array(target_labels).transpose().tolist())}
        df = pd.DataFrame({**df_pred, **df_true})
        self.test_df = df

        for name in self.vst_params:
            fig, ax = plt.subplots(figsize=(8, 6))
            y_label = name + ' (Predicted)'
            x_label = name + ' (Target)'
            ax.plot([np.amin(df[x_label]), np.amax(df[x_label])], [np.amin(df[x_label]), np.amax(df[x_label])],
                    ls='--', color='r')
            sns.scatterplot(data=df, y=y_label, x=x_label, ax=ax)
            sns.lineplot(data=df, y=y_label, x=x_label, ax=ax, color='g')
            self.writer.add_figure(name + '/Test', fig)

        self.writer.add_hparams({'lr': self.optimizer.defaults['lr'],
                                 'bsize': self.batch_size,
                                 'train_samples': len(self.dataset),
                                 'criterion': self.criterion.__str__(),
                                 'model': self.model.__str__(),
                                 'val_size': self.val_size,
                                 'sample_len': self.sample_len,
                                 }, {'test_dist':test_dist, 'test_R2': r_sq}, run_name='.')

        self.save()

    def plot_training_curve(self):
        ax = plt.figure().gca()
        plt.plot(range(1, len(self.train_errors) + 1), self.train_errors, 'b', label='Training loss')
        plt.plot(range(1, len(self.valid_errors) + 1), self.valid_errors, 'g', label='Validation loss')
        plt.axvline(self.early_stopping.best_epoch, linestyle='--', color='r', label='Early Stopping Checkpoint')
        plt.xlim(0, len(self.train_errors) + 1)  # consistent scale
        plt.grid(True)
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder, 'train_history.pdf'), dpi=600)  # , bbox_inches='tight')
        plt.clf()
        plt.cla()
        plt.close()

    def plot_training_curve_params(self):
        df_train = pd.DataFrame(np.array(self.train_errors), columns=self.vst_params)
        df_train['Epoch'] = np.arange(1, len(self.train_errors) + 1)
        df_train = df_train.melt(var_name='Parameter', value_name='Mean Absolute Error', id_vars=['Epoch'])
        df_train['Type'] = 'train'

        df_valid = pd.DataFrame(np.array(self.valid_errors), columns=self.vst_params)
        df_valid['Epoch'] = np.arange(1, len(self.valid_errors) + 1)
        df_valid = df_valid.melt(var_name='Parameter', value_name='Mean Absolute Error', id_vars=['Epoch'])
        df_valid['Type'] = 'valid'

        df = pd.concat([df_train, df_valid])
        sns.relplot(data=df, x="Epoch", y="Mean Absolute Error",
                    hue="Parameter", col="Type",
                    kind="line", size_order=["train", "valid"],
                    height=4, aspect=1.)
        plt.savefig(os.path.join(self.folder, 'train_history_param.pdf'), dpi=1200)


    def binary_test(self):
        print('\nEvaluating...')
        t0 = time.time()

        test_dataset = ReverbDataset(type='test', noisy=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=False,
                                 num_workers=6, drop_last=False, collate_fn=pad_collate)
        wet_blends = [test_dataset.meta[idx]['wet_blend'] for idx in range(len(test_dataset))]

        # Test
        with torch.no_grad():
            self.model.eval()
            self.model.cuda()
            running_loss, running_dist = 0., 0.
            target_labels, predicted_labels = [], []
            for i, (x, x_len, y) in tqdm(enumerate(test_loader), desc='Test', total=len(test_loader)):
                y_hat, embedding = self.model(x.cuda(), x_len)  # (batch, time)
                loss = self.criterion(y_hat, y[:, None].repeat(1, y_hat.shape[1]).cuda())
                running_loss += loss.item()
                running_dist += torch.mean(torch.abs(y_hat.cpu().detach() - y[:, None].repeat(1, y_hat.shape[1])))
                predicted_labels.extend(torch.mean(y_hat.cpu().detah(), dim=1).numpy())
                target_labels.extend(y.cpu().numpy())
        test_loss = running_loss / (i + 1)
        test_dist = running_dist / (i + 1)
        print('Evaluation: %ds - test_loss: %.4f - test_dist: %.3f' % (int(time.time() - t0), test_loss, test_dist))
        # plot results
        # predicted_labels = [np.mean(k, axis=1) for k in predicted_labels]
        predicted_labels = np.array(predicted_labels).flatten()
        target_labels = np.array(target_labels).flatten()
        self.plot_roc_curve(target_labels, predicted_labels)

        # error plot
        df = pd.DataFrame({'Wet Blend': np.array(wet_blends).round(decimals=1), 'Error':target_labels - predicted_labels})
        sns.lineplot(data=df, x='Wet Blend', y='Error')
        plt.show()

    def plot_roc_curve(self, target_labels, predicted_labels):
        from sklearn.metrics import roc_curve, roc_auc_score
        lr_auc = roc_auc_score(target_labels, predicted_labels)
        print('Logistic: ROC AUC=%.3f' % (lr_auc))
        lr_fpr, lr_tpr, _ = roc_curve(target_labels, predicted_labels)
        # plot the roc curve for the model
        plt.plot(lr_fpr, lr_tpr, marker='.', label='RevNet')
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.legend()
        # show the plot
        plt.show()

    def save(self, filepath=None):
        if filepath is None:
            filepath = os.path.join(self.folder, 'trainer.npy')
        np.save(filepath, [self.model, self.criterion, self.vst_params, self.folder,
                           self.train_errors, self.valid_errors, self.early_stopping, self.optimizer, self.val_size,
                           self.speech_subset_rate, self.train_ids, self.val_ids, self.test_embeddings, self.sample_len,
                           self.test_df, self.stimuli
                           ])

    def load(self, filepath):
        self.model, self.criterion, self.vst_params, self.folder, \
        self.train_errors, self.valid_errors, self.early_stopping, self.optimizer, self.val_size, \
        self.speech_subset_rate, self.train_ids, self.val_ids, self.test_embeddings, self.sample_len, self.test_df, self.stimuli = np.load(filepath, allow_pickle=True)

        self.batch_size = self.model.batch_size
        self.writer = SummaryWriter(comment=os.path.split(self.folder)[-1])
        self.early_stopping = EarlyStopping(patience=30, delta=1e-4, path=os.path.join(self.folder, 'model_checkpoint.pt'),
                                            verbose=False)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.fs = self.model.fs
        self.dataset = SimReverbDataset(params=self.vst_params, param_size=self.param_size, type='train', noisy=False,
                                        speech_subset_rate=self.speech_subset_rate, stimuli=self.stimuli)

    def load_model(self, filepath):
        if os.path.exists(filepath):
            self.model = torch.load(filepath)
        else:
            raise ValueError('Filepath does not exist:', filepath)

    def test_in_noisy(self):
        snrs = [0, 20, 40, 60, 80]
        print(time.strftime("%H:%M:%S"), 'Evaluating in Noisy conditions...')
        res_df = []
        snr_dists = []
        snr_r_sq = []
        df_all = []
        for snr in snrs:
            t0 = time.time()
            test_dataset = SimReverbDataset(params=self.vst_params, param_size=self.param_size, cache_files=False, type='test',
                                            noisy=True, snrs=[snr], stimuli=self.stimuli)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=False,
                                     num_workers=0, drop_last=False, collate_fn=pad_collate)
            # test
            with torch.no_grad():
                self.model.eval()
                running_loss, running_dist = 0., 0.
                target_labels, predicted_labels = [], []
                for i, (x, x_len, y) in tqdm(enumerate(test_loader), desc='Test %d dB' % snr, total=len(test_loader)):
                    y_hat, embedding = self.model(x.cuda(), x_len)  # (batch, time)
                    loss = self.criterion(y_hat, y.cuda())
                    running_loss += loss.item()
                    running_dist += torch.mean(torch.abs(y_hat.cpu() - y)).detach()
                    predicted_labels.extend(y_hat.clone().cpu().detach().numpy())
                    target_labels.extend(y.clone().cpu().detach().numpy())

            r_sq_full = [r_squared(np.array(target_labels)[:, i], np.array(predicted_labels)[:, i])
                            for i in range(len(self.vst_params))]
            mae_full = [np.mean(np.abs(np.array(target_labels)[:, i] - np.array(predicted_labels)[:, i]))
                         for i in range(len(self.vst_params))]

            df = pd.DataFrame(np.stack([mae_full, r_sq_full]).transpose(), columns=['MAE', 'R²'])
            df['Parameter'] = self.vst_params
            df.insert(0, 'Parameter', df.pop('Parameter'))
            print('Test Results: %d dB SNR' % snr)
            print(df.to_string())
            df['SNR (dB)'] = snr
            df.to_csv(os.path.join(self.folder, 'test_results_'+str(snr)+'dB.csv'))
            df_all.append(df)
            r_sq = np.mean(r_sq_full)
            test_dist = running_dist / (i + 1)
            res_df.append(test_dist)
            snr_dists.append(test_dist)
            snr_r_sq.append(r_sq)

        df_all = pd.concat(df_all)
        df_all_mae = df_all.drop(columns=['R²'])
        df_all_rsq = df_all.drop(columns=['MAE'])
        sns.relplot(data=df_all_mae.reset_index(), x='SNR (dB)', y='MAE', hue='Parameter', kind='line', height=4)
        plt.savefig(os.path.join(self.folder, 'robustness_mae.pdf'), dpi=1000)
        sns.relplot(data=df_all_rsq.reset_index(), x='SNR (dB)', y='R²', hue='Parameter', kind='line', height=4)
        plt.savefig(os.path.join(self.folder, 'robustness_rsq.pdf'), dpi=1000)


    def export_results(self):
        df = self.test_df
        fig_folder = create_folder(os.path.join(self.folder, 'plots'))
        trainer.plot_training_curve_params()
        for name in self.vst_params:
            y_label = name + ' (Predicted)'
            x_label = name + ' (Target)'
            if name in ['wet_db', 'early_reflections_db', 'reverb_db']:
                df[x_label] = df[x_label] * 60 - 60
                df[y_label] = df[y_label] * 60 - 60
            df['Error'] = np.abs(df[x_label] - df[y_label])
            # sns.scatterplot(data=df, y=y_label, x=x_label, ax=ax)
            plot1 = sns.jointplot(data=df, x=x_label, y=y_label, kind="hex", height=4,hue_norm=1)
            # plot1 =  sns.kdeplot(
            #     data=df, x=x_label, y=y_label, fill=True, thresh=0, levels=100, cmap="mako")
            plot1.ax_joint.plot([np.amin(df[x_label]), np.amax(df[x_label])], [np.amin(df[x_label]), np.amax(df[x_label])],
                    ls='--', color='r', linewidth=1)
            plt.savefig(os.path.join(fig_folder, name+'.pdf'))
            print('Saving:', os.path.join(fig_folder, name+'.pdf'))

        # plot embeddings with colored each class
        tsne = TSNE(learning_rate='auto', n_jobs=6, init='pca')
        embeddings = np.array([[self.test_embeddings[k][j] for j in range(len(self.test_embeddings[k]))]
                               for k in range(len(self.test_embeddings))])

        fig, axs = plt.subplots(nrows=1, ncols=len(self.vst_params), figsize=(4 * len(self.vst_params), 5))
        for name, ax in zip(self.vst_params, axs.ravel()):
            X_embedded = tsne.fit_transform(embeddings, y=self.test_df[name + ' (Target)'])
            sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=self.test_df[name + ' (Target)'], ax=ax)
            ax.set_title(name)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.savefig(os.path.join(fig_folder, 'embeddings.pdf'))
        print('Saving:', os.path.join(fig_folder, 'embeddings.pdf'))


sample_len = 6. # seconds
if __name__ == '__main__':
    trainer = Trainer(sample_len=sample_len)
    trainer.generate_sim_dataset()
    trainer.train(epochs=100)
    # trainer.load(filepath='models\\03-04Rev_4\\trainer.npy')
    # trainer.load_model(filepath='models\\03-04Rev_4\\model.pt')
    trainer.test()
    trainer.test_in_noisy()
    trainer.export_results()

