from utils import load_model

class GIZModel(nn.Module):
	def __init__(self, args):
		self.model = load_model(args)

	def forward(self, x):
		return self.model(x)
