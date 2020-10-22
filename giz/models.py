import torch
import torch.nn as nn
import torchvision.models as tvm
from .utils import *


__BASE__ = {
	'resnet': 'fc',
	'densenet': 'classifier',
	'xresnet': '',
	'efficientnet': ''
}

__FC__ = {
	'resnet': 
		{
			'resnet18': 512,
			'resnet34': 512,
			'resnet50': 2048,
			'resnet101': 2048,
			'resnet152': 2048,
		},
	'densenet':
		{
			'densenet121': None,
		    'densenet169': None,
		    'densenet201': None,
		    'densenet161': 2208,
		}
}


class GIZModel(nn.Module):
	def __init__(self, args):
		super(GIZModel, self).__init__()

		self.model = create_model(args)

	def forward(self, x):
		return self.model(x)


class DenseBlock(nn.Module):
	def __init__(self, fc_size, dp_rate, n_classes=None):
		super(DenseBlock, self).__init__()

		n = n_classes or fc_size//2

		self.linear = nn.Linear(fc_size, n)
		self.dropout = nn.Dropout(dp_rate)

		nn.init.xavier_normal_(self.linear.weight)

	def forward(self, x):
		x = self.dropout(x)
		return self.linear(x)



def create_model(args):
	"""
		Arguments that must be in args:
		- base_name: root name of the model
		- model_name : model name
		- drop_rate: percentage of droupout
		- n_classes : nb of output classes

		Optional arguments:
		- nb_layer : number of Dense blocks in the last layer
	"""

	model_name = args.model_name
	base_name = args.base_name
	drop_rate = args.drop_rate
	n_classes = args.n_classes

	fc_size = __FC__[base_name][model_name]
	fc_name = __BASE__[base_name]

	try:
		n = args.nb_layer
	except Exception:
		n = 1

	layers = []
	for _ in range(n):
		layers += [DenseBlock(fc_size, drop_rate)]
		fc_size = fc_size//2

	layers += [nn.AdaptiveAvgPool2d(1), DenseBlock(fc_size, drop_rate, n_classes)]

	model = getattr(tvm, model_name)(pretrained=True)
	setattr(model, fc_name, nn.Sequential(*layers))

	return model


def get_model(args):
	path = get_save_path(args) + 'pretrain/'

	model = GIZModel(args)

	if os.path.exists(path):
		model.load_state_dict(torch.load(f'{path}{args.save_model}'))

	model