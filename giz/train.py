def train(args):
	print('Started training ....')

	dataloder = load_dataset(args)
	model = GIZModel(args)

	criterion = nn.CrossEntropyLoss()
  	opt = AdamW(model.parameters(), lr=args.lr)

  	device = 'cuda' if args.cuda else 'cpu'

  	model.train()
  	model.to(device)

  	pbar = tqdm(range(args.epochs), desc='Training ...')

  	for epoch in pbar:
  		print(f'Epoch {i+1} : ')

  		epoch_loss = 0

  		for i, data in enumerate(dataloder):
  			opt.zero_grad()

  			data['wav'] = data['wav'].to(device)
  			data['target'] = data['target'].to(device)

  			output = model(data['wav'])
  			loss = criterion(output, data['target'])

  			epoch_loss += loss.item()

  			loss.backward()
  			opt.step()

  			print(f"\rLoss : {epoch_loss/(i+1)}")

  	save_model(model)