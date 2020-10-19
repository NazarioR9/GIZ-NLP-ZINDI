__BASE__ = {
	'resnet': 'fc',
	'densenet': 'classifier',
	'xresnet': '',
	'efficientnet': ''
}

__FC__ = {
	'resnet': 
		{
			'base': 512,
			'large': 2048
		},
	'densenet':
		{
			'densenet121': None,
		    'densenet169': None,
		    'densenet201': None,
		    'densenet161': 2208,
		}
}

def load_model(**args):
	"""
		Arguments that must be in args:
		- base_name: root name of the model
		- model_name : model name
		- drop_rate: percentage of droupout
		- n_classes : nb of output classes

		Optional arguments:
		- nb_layer : number of Dense blocks in the last layer
	"""

	model_name = args['model_name']
	base_name = args['base_name']
	drop_rate = args['drop_rate']
	n_classes = args['n_classes']

	fc_size = __FC__[base_name][model_name]
	fc_name = __BASE__[base_name]

	try:
		n = args['nb_layer']
	except ValueError:
		n = 1


	layers = []
	for _ in range(n):
		layers += [nn.Dense(fc_size, fc_size//2), nn.Dropout(drop_rate)]
		fc_size = fc_size//2


	layers += [nn.Dense(fc_size, n_classes)]

	model = getattr(tvm, model_name)
	setattr(model, nn.Sequential(*layers))

	return model