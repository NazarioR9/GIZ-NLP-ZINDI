import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tvm

import os
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

def load_model(args):
	"""
		Arguments that must be in args:
		- base_name: root name of the model
		- model_name : model name
		- drop_rate: percentage of droupout
		- n_classes : nb of output classes

		Optional arguments:
		- nb_layer : number of Dense blocks in the last layer
	"""

	model_name = args['model_name']
	base_name = args['base_name']
	drop_rate = args['drop_rate']
	n_classes = args['n_classes']

	fc_size = __FC__[base_name][model_name]
	fc_name = __BASE__[base_name]

	try:
		n = args['nb_layer']
	except KeyError:
		n = 1

	layers = []
	for _ in range(n):
		layers += [nn.Dense(fc_size, fc_size//2), nn.Dropout(drop_rate)]
		fc_size = fc_size//2


	layers += [nn.Dense(fc_size, n_classes)]

	model = getattr(tvm, model_name)(pretrained=True)
	setattr(model, nn.Sequential(*layers))

	return model


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
	spectrogram = librosa.feature.melspectrogram(signal, sr=sr, n_mels=40, hop_length=160, n_fft=480, fmin=20, fmax=4000)
	spectrogram = librosa.power_to_db(spectrogram)
	spectrogram = spectrogram.astype(np.float32)

	return spectrogram

def preprocess_wav(wav, normalization=True):
	data = wav.reshape(1, -1)
	if normalization:
		mean = data.mean()
		data -= mean
	return data

def save_model(model, args):
	path = "{}{}/{}_{}/".format(args.model_hub,args.base_name,args.model_name,args.proc)
	os.mkdir(path, exist_ok=True)

	torch.save(model.state_dict(), f'{path}model.bin')