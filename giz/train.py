import torch
import torch.nn as nn
from torch.optim import AdamW, Adam
from tqdm import tqdm
import numpy as np
from .data import load_dataset
from .models import get_model
from .utils import save_model, get_loss


def train(args):
	print(f"\n[{args.proc.upper()}]Started training ...")

	best_loss = np.inf

	trainloader, valoader = load_dataset(args)
	model = get_model(args)

	criterion = get_loss(args)
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

		val_loss = eval(model, valoader, device, criterion, phase="val")

		if val_loss < best_loss:
			best_loss = val_loss
			save_model(model, args)


def one_epoch(model, dataloader, device, opt=None, criterion=None, phase="train"):
	model.train()

	epoch_loss = 0
	size = len(dataloader)


	for i, data in enumerate(dataloader):
		opt.zero_grad()
		
		data['wav'] = data['wav'].to(device)
		data['target'] = data['target'].to(device)

		output = model(data['wav'])

		loss = criterion(output, data['target'])
		epoch_loss += loss.item()
	
		loss.backward()
		opt.step()

		print(f"\r[{i+1}/{size}] {phase} loss : {epoch_loss/(i+1)}", end='')
	print()

	return epoch_loss

def eval(model, dataloader, device, criterion=None, phase="val"):
	model.eval()

	epoch_loss = 0
	preds = []
	size = len(dataloader)

	with torch.no_grad():
		for i, data in enumerate(dataloader):

			data['wav'] = data['wav'].to(device)
			data['target'] = data['target'].to(device)

			output = model(data['wav'])

			if phase=='val':
				loss = criterion(output, data['target'])
				epoch_loss += loss.item()
				
				print(f"\r[{i+1}/{size}] {phase} loss : {epoch_loss/(i+1)}", end='')
			else:
				preds += output.detach().cpu().numpy().tolist()

		print()

	if phase=='val':
		return epoch_loss

	return preds