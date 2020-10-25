import torch
import os
import cv2
import numpy as np
import librosa
import python_speech_features as psf


def preprocess_psf(signal, sr, window_length=0.05, window_step=0.0045, NFFT=2205):
  # preemphasis
  signal = psf.sigproc.preemphasis(signal, coeff=0.95)
  # get specrogram
  # Get the frames
  frames = psf.sigproc.framesig(signal, 
                              window_length*sr, 
                              window_step*sr, 
                              lambda x:np.ones((x,))) 
  # magnitude Spectrogram
  spectrogram = np.rot90(psf.sigproc.magspec(frames, NFFT))
  
  return spectrogram


def preprocess_mfcc(signal, sr):
	mfcc = librosa.feature.mfcc(signal, sr=sr)
	mfcc = mfcc.astype(np.float32)
	return mfcc

def preprocess_mel(signal, sr):
	spectrogram = librosa.feature.melspectrogram(signal, sr=sr)
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
	return np.stack([wav, wav, wav], axis=-1)


def normalize(wav, mean=None, std=None):
	mean = mean or wav.mean()
	std = std or wav.std()
	wav = (wav - mean) / (std + eps)

	_min, _max = wav.min(), wav.max()

	if (_max - _min) > eps:
		wav = np.clip(wav, _min, _max)
		wav = 255 * (wav - _min) / (_max - _min)
		wav = wav.astype(np.uint8)

	return wav

def resize(img, size=None):
	if size is not None:
		if isinstance(size, int):
			size = (size, size)
		return cv2.resize(img, size)
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


def get_loss(args):
	if args.loss == 'bce':
		return torch.nn.BCELossWithLogits()
	else:
		return torch.nn.CrossEntropyLoss()

def get_proc_func(args):
	if 'mel' in args.proc:
		proc_fun = preprocess_mel
	elif 'mfcc' in args.proc:
		proc_fun = preprocess_mfcc
	else:
		proc_fun = preprocess_psf

	return proc_fun