import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from .data import load_dataset
from .models import GIZModel
from .utils import save_model


def train(args):
	print(f"\n[{args.proc.upper()}]Started training ...")

	dataloader = load_dataset(args)
	model = GIZModel(args)

	criterion = nn.CrossEntropyLoss()
	opt = AdamW(model.parameters(), lr=args.lr)
	device = 'cuda' if args.cuda else 'cpu'

	model.train()
	model.to(device)

	pbar = tqdm(range(args.epochs), desc='Training ... ')
	size = len(dataloader)

	for epoch in pbar:
		print(f'\nEpoch {epoch+1} : ')

		epoch_loss = 0

		for i, data in enumerate(dataloader):
			opt.zero_grad()

			data['wav'] = data['wav'].to(device)
			data['target'] = data['target'].to(device)

			output = model(data['wav'])
			loss = criterion(output, data['target'])

			epoch_loss += loss.item()

			loss.backward()
			opt.step()

			print(f"\r[{i+1}/{size}] Loss : {epoch_loss/(i+1)}", end='')
		print()

	save_model(model, args)