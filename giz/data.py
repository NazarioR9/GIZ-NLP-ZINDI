class TemplateDataset(Dataset):
	def __init__(self, df, imgsize=(224, 224), phase='train', **args):
		self.df = df
		self.phase = phase
		self.imgsize = imgsize

		self.base_transforms = tfms.compose(
			tfms.ToTensor(),
			)

		self.transforms = None

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		x = self.df.loc[idx, 'img']
		img = cv2.imread(x)
		img = cv2.resize(img, self.imgsize)

		if self.transforms:
			img = self.transforms(img)


		out = {
			'image': self.base_transforms(img),
		}

		if self.phase=='train':
			y = self.df.lco[idx, 'target']
			out.update(
				'target': torch.tensor(y, dtype=torch.float)
				)

		return out