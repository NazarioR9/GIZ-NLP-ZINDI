import os, sys, gc, glob
import pandas as pd
import numpy as np
from zindi import user as zuser
import argparse



parser = argparse.ArgumentParser(description='Logging phase')

parser.add_argument('-username', type=str, help='Your Zindi username')
parser.add_argument('-store', type=str, default='data/raw/', help="***")

parser.add_argument('-download', action='store_true', help="")
parser.add_argument('-pp', action='store_true', help="")


def download(args):
	user = zuser.Zindian(args.username)
	user.which_challenge
	user.select_a_challenge()
	user.download_dataset(args.store)

def preprocessing():
	additionnal = glob.glob("data/latest_keywords/*/*.wav")

	add = pd.DataFrame({'fn': additionnal})
	add['target'] = add['fn'].apply(lambda x: x.split('/')[-2])

	train = pd.read_csv('data/raw/Train.csv')
	train['fn'] = 'data/' + train['fn']
	train.rename(columns = {'label': 'target'}, inplace=True)

	train = pd.concat([train, add], axis=0)
	train.to_csv('data/processed/Train.csv', index=False)

	subs = pd.read_csv('data/raw/SampleSubmission.csv')
	subs['fn'] = 'data/' + subs['fn']
	subs.to_csv('data/processed/SampleSubmission.csv', index=False)



def main(parser):
	args = parser.parse_args()

	assert args.download != args.pp, "Can't use --download and --pp together."

	if args.download: download(args)

	if args.pp: preprocessing()


if __name__ == '__main__':
	main(parser)

