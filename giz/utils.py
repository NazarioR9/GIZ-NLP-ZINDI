import torch
import os
import cv2
import numpy as np
import librosa


def preprocess_mfcc(signal, sr):
	mfcc = librosa.feature.mfcc(signal, sr=sr, n_mfcc=40)
	mfcc = mfcc.astype(np.float32)
	return mfcc

def preprocess_mel(signal, sr):
	spectrogram = librosa.feature.melspectrogram(signal, sr=sr, n_mels=128, fmin=20, fmax=8000)
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

def get_save_path(args):
	path = "{}{}/{}_{}/".format(args.model_hub, args.base_name, args.model_name, args.proc)
	
	if args.pretrain:
		path += 'pretrain/'

	return path

def save_model(model, args):
	path = get_save_path(args)

	os.makedirs(path, exist_ok=True)

	torch.save(model.state_dict(), f'{path}{args.save_model}')