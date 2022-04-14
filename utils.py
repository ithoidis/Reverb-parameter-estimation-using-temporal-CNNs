import numpy as np
import scipy as sp
import librosa
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
import scipy
import random
from sklearn.metrics import mean_squared_error
import soundfile as sf
import pickle
import os
from tqdm import tqdm

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.best_epoch = 0
        self.current_epoch = 0

    def __call__(self, val_loss, model):

        score = -val_loss
        self.current_epoch += 1
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = self.current_epoch
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, self.path)
        self.val_loss_min = val_loss


def get_list_of_files(dirName):
    '''
        For the given path, get the List of all files in the directory tree
    '''
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + get_list_of_files(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

def create_folder(fd, add=0):
    import os
    if add != 0:
        if fd[-1].isdigit():
            le = len(fd.split('_')[-1])
            fd = fd[:-le] + str(add)
        else:
            fd = fd + str(add)
    if not os.path.exists(fd):
        os.makedirs(fd)
        return fd
    else:
        fd = create_folder(fd, add=add+1)
        return fd


def delete_folder(dirpath):
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)


def rms(y):
    """
    Root mean square value of signal
    :param y:
    :return:
    """
    return np.sqrt(np.mean(y ** 2, axis=-1))


def r_squared(actual, predicted):
    sse = np.sum(np.square(actual - predicted))
    sst = np.sum(np.square(actual - np.mean(actual)))
    return 1 - (sse / sst)

def split(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di


def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)


def resample_all(folder, target_fs):

    fpaths = [a for a in get_list_of_files(folder) if '.wav' in a]
    new_folder = create_folder(os.path.join(folder, 'resampled_%d' % target_fs))
    for fpath in tqdm(fpaths):
        fs, x = read_audio(fpath, target_fs=target_fs)
        fname = os.path.split(fpath)[-1]
        f_fold = os.path.split(os.path.split(fpath)[-2])[-1]
        write_audio(os.path.join(new_folder, f_fold +'_'+ fname), fs=target_fs, wav=x)

def frame_all(folder, sample_len=10):
    fpaths = [a for a in get_list_of_files(folder) if '.wav' in a]
    new_folder = create_folder(os.path.join(folder, 'framed_%dsec' % sample_len))
    for fpath in tqdm(fpaths):
        fs, x = read_audio(fpath)
        fname = os.path.split(fpath)[-1]
        x_len_sec = len(x) / fs
        for i, t in enumerate(np.arange(0, x_len_sec, sample_len)):
            write_audio(os.path.join(new_folder, fname[:-4] + '_%d.wav'%i), fs=fs, wav=x[int(t*fs):int((t+sample_len)*fs)])


def read_audio(filename, target_fs=None):
    # x, fs = librosa.load(filename, sr=target_fs) # was producing error is a few datasets
    if '.flac' in filename[-5:]:
        x, fs = sf.read(filename)
    elif '.wav' in filename[-5:]:
        fs, x = read(filename)
        # convert to mono
        if len(x.shape) == 2:
            if x.shape[0] == 2:
                x = np.sum(x, axis=0)
            else:
                x = np.sum(x, axis=1)
    else:
        pass
    if np.isnan(x).any() or np.isinf(x).any():
        raise ValueError('Error: NaN or Inf value. File path: %s.' % (filename))
    if not (isinstance(x[0], np.float32) or isinstance(x[0], np.float64)):
        x = x.astype('float32') / np.power(2, 15)
    if target_fs is not None and fs != target_fs:
        x = librosa.resample(x, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
        # print('resampling...')
    return fs, x

def write_audio(filename, fs, wav):
    if isinstance(wav[0], np.float32) or isinstance(wav[0], np.float64):
        wav = np.asarray(np.multiply(wav, 32768.0), dtype=np.int16)
    write(filename, fs, wav)
