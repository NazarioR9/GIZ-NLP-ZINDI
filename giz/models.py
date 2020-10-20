from .utils import load_model
import torch.nn as nn

class GIZModel(nn.Module):
	def __init__(self, args):
		super(GIZModel, self).__init__()

		self.model = load_model(args)

	def forward(self, x):
		return self.model(x)
