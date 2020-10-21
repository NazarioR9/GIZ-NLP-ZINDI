import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import os
import cv2
import numpy as np
import librosa

__BASE__ = {
	'resnet': 'fc',
	'densenet': 'classifier',
	'xresnet': '',
	'efficientnet': ''
}

__FC__ = {
	'resnet': 
		{
			'resnet18': 512,
			'resnet34': 512,
			'resnet50': 2048,
			'resnet101': 2048,
			'resnet152': 2048,
		},
	'densenet':
		{
			'densenet121': None,
		    'densenet169': None,
		    'densenet201': None,
		    'densenet161': 2208,
		}
}



def preprocess_mfcc(signal, sr):
	spectrogram = librosa.feature.melspectrogram(signal, sr=sr, n_mels=40, hop_length=160, n_fft=480, fmin=20, fmax=4000)
	idx = [spectrogram > 0]
	spectrogram[idx] = np.log(spectrogram[idx])

	dct_filters = librosa.filters.dct(n_filters=40, n_input=40)
	mfcc = [np.matmul(dct_filters, x) for x in np.split(spectrogram, spectrogram.shape[1], axis=1)]
	mfcc = np.hstack(mfcc)
	mfcc = mfcc.astype(np.float32)
	return mfcc

def preprocess_mel(signal, sr):
	spectrogram = librosa.feature.melspectrogram(signal, sr=sr, n_mels=128, fmin=20, fmax=4000)
	spectrogram = librosa.power_to_db(spectrogram)
	spectrogram = spectrogram.astype(np.float32)

	return spectrogram

def preprocess_wav(wav, normalization=True):
	data = wav.reshape(1, -1)
	if normalization:
		mean = data.mean()
		data -= mean
	return data

def mono_to_color(wav, eps=1e-6, mean=None, std=None):
	wav = np.stack([wav, wav, wav], axis=-1)

	mean = mean or wav.mean()
	std = std or wav.std()
	wav = (wav - mean) / (std + eps)

	_min, _max = wav.min(), wav.max()

	if (_max - _min) > eps:
		wav = np.clip(wav, _min, _max)
		wav = 255 * (wav - _min) / (_max - _min)
		wav = wav.astype(np.uint8)

	return np.transpose(wav, (2, 0, 1))

def resize(img, size=None):
	if size is not None:
		return cv2.resize(img, (size, size))
	return img

def save_model(model, args):
	path = "{}{}/{}_{}/".format(args.model_hub, args.base_name, args.model_name, args.proc)
	os.makedirs(path, exist_ok=True)

	torch.save(model.state_dict(), f'{path}model.bin')