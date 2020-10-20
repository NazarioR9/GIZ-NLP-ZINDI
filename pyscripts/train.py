from giz.imports import *
import argparse

parser = argparse.ArgumentParser(description='Training phase parser')

parser.add_argument('-data', type=str, default='data/raw/', help='path to the data folder')
parser.add_argument('-model_name', type=str, default='resnet18', help='model name')
parser.add_argument('-base_name', type=str, default='resnet', help='model root name')
parser.add_argument('-n_classes', type=int, default=194, help='')
parser.add_argument('-proc', type=str, default='mel', help='')
parser.add_argument('-bs', type=int, default=32, help='batch size')
parser.add_argument('-lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('-epochs', type=int, default=1, help='number of training unit')
parser.add_argument('-size', type=int, default=224, help='image size')
parser.add_argument('-nb_layer', type=int, default=1, help='')
parser.add_argument('-drop_rate', type=float, default=0.5, help='')
parser.add_argument('-pseudo', action='store_true', help='use pseudo labels for training')
parser.add_argument('-model_hub', type=str, default='model_hub/', help='')
parser.add_argument('-pseudo_dir', type=str, default='pseudo/', help='')
parser.add_argument('-cuda', action='store_true', help='')


parser.add_argument('-verbose', '--v', action='store_true', help='verbosity')

def main():
	args = parser.parse_args()

	train(args)

	args.proc = 'mfcc'

	train(args)


if __name__ == '__main__':
	main()