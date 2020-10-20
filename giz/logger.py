

class Logger:
	def __init__(self, args):
		self.args = args

	def _verbose(self):
		if 


	def log(self, msg):
		if self.args.v:
			print(msg)