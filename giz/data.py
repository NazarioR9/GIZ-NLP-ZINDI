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
		self.sr = 16000
		self.proc_fun = proc_fun
		self.classes = self.df.target.unique()
		self.map = dict(zip(self.classes, range(len(self.classes))))
		
	def __len__(self):
		return len(self.df)

	def read_wav(self, fn):
		wav = librosa.load(fn, sr=self.sr)[0]
		if len(wav) < self.sr:
			wav = np.pad(wav, (0, self.sr-len(wav)), 'constant')
		return wav[:self.sr]

	def __getitem__(self, idx):
		fn = self.df.loc[idx, 'fn']
		wav = self.read_wav(fn)
		wav = self.proc_fun(wav, self.sr)
		wav = mono_to_color(wav)
		wav = resize(wav)

		out = {
			'wav': torch.from_numpy(wav).float(),
		}

		if self.phase=='train':
			y = self.df.loc[idx, 'target']
			y = self.map[y]

			out.update(
				{'target': torch.tensor(y, dtype=torch.long)}
				)

		return out


def load_dataset(args):
	if 'mel' in args.proc:
		proc_fun = preprocess_mel
	else:
		proc_fun = preprocess_mfcc

	train = pd.read_csv(args.data + 'Train.csv')
	dataset = GIZDataset(train, proc_fun, size=args.size)

	dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True)

	return dataloader