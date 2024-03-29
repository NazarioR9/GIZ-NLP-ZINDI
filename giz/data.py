import librosa
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from .utils import *


class GIZDataset(Dataset):
	def __init__(self, df, proc_fun, phase='train', size=224, **args):
		super(GIZDataset, self).__init__()

		self.df = df
		self.phase = phase
		self.size = size
		self.sr = 44100
		self.secs = 3
		self.length = self.sr * self.secs
		self.proc_fun = proc_fun
		self.classes = self.df.target.unique()
		self.map = dict(zip(self.classes, range(len(self.classes))))
		self.args = args
		
	def __len__(self):
		return len(self.df)

	def read_wav(self, fn):
		wav = librosa.load(fn, sr=self.sr, mono=True)[0]
		return wav

	def pad(self, wav):
		if len(wav) < self.length:
			wav = np.pad(wav, (0, self.length-len(wav)), 'constant')
		return wav[:self.length]

	def __getitem__(self, idx):
		fn = self.df.loc[idx, 'fn']
		wav = self.read_wav(fn)
		wav = self.proc_fun(wav, self.sr)
		if self.args['mono']:
			wav = resize(wav, self.size)
			wav = np.expand_dims(wav, axis=0)
		else:
			wav = mono_to_color(wav)
			wav = resize(wav, self.size)
			wav = np.transpose(wav, (2, 0, 1))
		wav = normalize(wav)
		

		out = {
			'wav': torch.from_numpy(wav).float(),
		}

		if self.phase!='test':
			y = self.df.loc[idx, 'target']
			y = self.map[y]

			dtype = torch.long

			if self.args['loss'].startswith('bce'):
				dtype = torch.float


			out.update(
				{'target': torch.tensor(y, dtype=dtype)}
				)

		return out


def load_dataset(args):
	proc_fun = get_proc_func(args)

	train = pd.read_csv(args.data + 'Train.csv')
	trainset = GIZDataset(train, proc_fun, size=args.size, loss=args.loss, mono=args.mono)
	trainloader = DataLoader(trainset, batch_size=args.bs, shuffle=True)

	valoader = None

	if not args.pretrain:
		val = pd.read_csv(args.data + 'Val.csv')
		valset = GIZDataset(val, proc_fun, size=args.size, phase='val', loss=args.loss, mono=args.mono)
		valoader = DataLoader(valset, batch_size=args.bs//2)

	return trainloader, valoader