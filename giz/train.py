import argparse

parser = argparse.ArgumentParser(description='Training phase parser')

parser.add_argument('-data', type=str, default='data/raw/', help='path to the data folder')
parser.add_argument('-model_name', type=str, default='resnet18', help='model name')
parser.add_argument('-base_name', type=str, default='resnet', help='model root name')
parser.add_argument('-bs', type=int, default=32, help='batch size')
parser.add_argument('-epochs', type=int, default=10, help='number of training unit')
parser.add_argument('-size', type=int, default=224, help='image size')
parser.add_argument('-nb_layer', type=int, default=1, help='')
parser.add_argument('-pseudo', action='store_true', help='use pseudo labels for training')
parser.add_argument('-model_hub', type=str, default='model_hub/', help='')
parser.add_argument('-pseudo_dir', type=str, default='pseudo/', help='')

parser.add_argument('-verbose', '--v', action='store_true', help='verbosity')

def main():
	args = parser.parse_args()
	print(args)
	while 1:
		continue


if __name__ == '__main__':
	main()