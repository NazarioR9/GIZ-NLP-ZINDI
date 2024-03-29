import os, sys, gc
import numpy as np
import pandas as pd
import argparse

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from giz.imports import *


parser = argparse.ArgumentParser(description="Sanity check test")
parser.add_argument("-data", type=str, default="data/processed/", help="")

# parser.add_argument('-model_name', type=str, default='resnet18', help='model name')
# parser.add_argument('-base_name', type=str, default='resnet', help='model root name')
# parser.add_argument('-save_model', type=str, default='model.bin', help='saved model name')
# parser.add_argument('-n_classes', type=int, default=193, help='')
parser.add_argument('-proc', choices=['mel', 'mfcc'], default='mel', help='')
# parser.add_argument('-bs', type=int, default=32, help='batch size')
# parser.add_argument('-lr', type=float, default=1e-4, help='learning rate')
# parser.add_argument('-epochs', type=int, default=1, help='number of training unit')
parser.add_argument('-size', type=int, default=224, help='image size')
# parser.add_argument('-nb_layer', type=int, default=0, help='')
# parser.add_argument('-drop_rate', type=float, default=0.5, help='')
parser.add_argument('-loss', choices=['ce', 'bce'], default='ce', help='')
# parser.add_argument('--pseudo', action='store_true', help='use pseudo labels for training')
parser.add_argument('--pretrain', action='store_true', default=True, help='')
# parser.add_argument('-model_hub', type=str, default='model_hub/', help='')
# parser.add_argument('-pseudo_dir', type=str, default='pseudo/', help='')
parser.add_argument('--mono', action='store_true', help='')
parser.add_argument('--mel', action='store_true', default=True, help='')
# parser.add_argument('--mfcc', action='store_true', help='')
# parser.add_argument('--cuda', action='store_true', help='')

# parser.add_argument("", action="", type="", choices=[], default="", help="")


def testDataset(args):
	nrows = 3

	proc_fun = get_proc_func(args)
	train = pd.read_csv(args.data + 'Train.csv')
	ds = GIZDataset(train, proc_fun, size=args.size, loss=args.loss, mono=args.mono)

	rands = np.random.randint(0, len(ds), nrows**2)

	fig = plt.figure(figsize=(12,10))

	for i, n in enumerate(rands):
	    out = ds[i]
	    img, y = out.values()
	    img = img.numpy().squeeze(0)
	    y = y.numpy()

	    ax = plt.subplot(nrows, nrows, i+1)
	    plt.imshow(img)
	    plt.title(ds.classes[y])
	plt.show()


if __name__ == '__main__':
	args = parser.parse_args()

	testDataset(args)