### Module imports ###
import numpy as np
import pandas as pd
import cv2
import librosa

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms


### Global Variables ###
audio_parameters = {'sampling_rate': 48000, 'desired_sample': 48000}
melspectrogram_parameters = {'n_mels': 128, 'fmin': 20, 'fmax': 24000}


### Class declarations ###
class SpeechDataset(Dataset):

    def __init__(self, csv_path, audio_parameters, melspectrogram_parameters, img_size):
        self.df = pd.read_csv(csv_path)
        self.filepath = self.df['path']
        self.label = self.df['label']
        self.audio_parameters = audio_parameters
        self.melspectrogram_parameters = melspectrogram_parameters
        self.img_size = img_size
        self.transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.filepath[idx]
        label = np.asarray(self.label[idx])
        label = torch.from_numpy(label).long()

        y, sr = librosa.load(path, sr=self.audio_parameters['sampling_rate'])
        len_y = len(y)
        desired_sample = self.audio_parameters['desired_sample']
        if len_y < desired_sample:
            new_y = np.zeros(desired_sample, dtype=y.dtype)
            start = np.random.randint(desired_sample - len_y)
            new_y[start:start + len_y] = y
            y = new_y
        else:
            start = np.random.randint(len_y - desired_sample)
            y = y[start:start + desired_sample]

        melspec = librosa.feature.melspectrogram(y, sr=audio_parameters['sampling_rate'], **self.melspectrogram_parameters)
        melspec = librosa.power_to_db(melspec).astype(np.float32)
        image = mono_to_color(melspec)
        height, width, _ = image.shape
        image = cv2.resize(image, (int(width * self.img_size / height), self.img_size))
        # image = np.moveaxis(image, 2, 0)
        image = (image / 255.0).astype(np.float32)
        image = self.transforms(image)
        return image, label


### Function declarations ###
def mono_to_color(X: np.ndarray,
                  mean=None,
                  std=None,
                  norm_max=None,
                  norm_min=None,
                  eps=1e-6):
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V


if __name__ == '__main__':
    speech_dataset = SpeechDataset('../Train-Test-Data/train_colab.csv', audio_parameters, melspectrogram_parameters, img_size=224)
    a = speech_dataset.__getitem__(4)
    print(a[0].shape)
