from utils import *

class TemplateModel(nn.Module):
	def __init__(self, model_name, **args):
		self.args = args
		self.model = load_model(args)

	def forward(self, x):
		return self.model(x)
