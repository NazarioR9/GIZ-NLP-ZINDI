import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
from .data import load_dataset
from .models import get_model
from .utils import save_model


def train(args):
	print(f"\n[{args.proc.upper()}]Started training ...")

	best_loss = np.inf

	trainloader, valoader = load_dataset(args)
	model = get_model(args)

	criterion = nn.CrossEntropyLoss()
	opt = AdamW(model.parameters(), lr=args.lr)
	device = 'cuda' if args.cuda else 'cpu'

	model.to(device)

	pbar = tqdm(range(args.epochs), desc='Training ... ')

	for epoch in pbar:
		print(f'\nEpoch {epoch+1} : ')

		train_loss = one_epoch(model, trainloader, device, opt, criterion)

		if args.pretrain:
			save_model(model, args)
			continue

		val_loss = one_epoch(model, valoader, device, opt, criterion, phase="val")

		if val_loss < best_loss:
			best_loss = val_loss
			save_model(model, args)


def one_epoch(model, dataloader, device, opt=None, criterion=None, phase="train"):

	if phase=="train":
		model.train()
		opt.zero_grad()
	else:
		model.eval()
	
	epoch_loss = 0
	size = len(dataloader)

	for i, data in enumerate(dataloader):
		
		data['wav'] = data['wav'].to(device)
		data['target'] = data['target'].to(device)

		output = model(data['wav'])

		try:
			loss = criterion(output, data['target'])
			epoch_loss += loss.item()
		except Exception:
			if phase=='train':
				raise ValueError("Make sure that **data as key 'target'** or **phase is correct**.")

		if phase=="train":
			loss.backward()
			opt.step()
			opt.zero_grad()


		if epoch_loss:
			print(f"\r[{i+1}/{size}] {phase} loss : {epoch_loss/(i+1)}", end='')
	print()

	return epoch_loss