class GIZDataset(Dataset):
	def __init__(self, df, proc_fun, phase='train', **args):
		self.df = df
		self.phase = phase
		self.sr = args.sr ifargs.sr else 16000
		self.proc_fun = proc_fun
		self.classes = self.df.target.unique()
		self.map = dict(self.classes, range(len(self.classes)))


	def __len__(self):
		return len(self.df)

	def read_wav(fn):
		wav = librosa.load(fn, sr=self.sr)[0]
	    if len(wav) < self.sr:
	        wav = np.pad(wav, (0, self.sr - len(wav)), 'constant')
	    return wav[:self.sr]

	def __getitem__(self, idx):
		fn = self.df.loc[idx, 'fn']
		wav = self.read_wav(fn)
		wav = self.proc_fun(wav, self.sr)

		out = {
			'wav': torch.from_numpy(wav_numpy).float(),
		}

		if self.phase=='train':
			y = self.df.lco[idx, 'target']
			y = self.map[y]

			out.update(
				'target': torch.tensor(y, dtype=torch.float)
				)

		return out