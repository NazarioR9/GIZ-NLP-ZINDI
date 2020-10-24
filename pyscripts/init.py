import os, sys, gc, glob
import argparse
import pandas as pd
import numpy as np
from zindi import user as zuser
from sklearn.model_selection import train_test_split, StratifiedKFold



parser = argparse.ArgumentParser(description='Logging phase')

parser.add_argument('-username', type=str, help='Your Zindi username')
parser.add_argument('-data', type=str, default='data/raw/', help="***")
parser.add_argument('-to', type=str, default='data/processed/', help="***")
parser.add_argument('-base', type=str, default='data/processed/base/', help="***")
parser.add_argument('-add', type=str, default='data/processed/add/', help="***")
parser.add_argument('-use', type=str, default='data/processed/add/', help="data to use for split")
parser.add_argument('-strategy', choices=['tts', 'skf'], default='tts', help="split strategy")
parser.add_argument('-n_splits', type=int, default=5, help="nb of splits. Is used with kf/skf")
parser.add_argument('-split', type=float, default=0.2, help="test size portion")
parser.add_argument('-seed', type=int, default=42, help='randomness factor')

parser.add_argument('-download', action='store_true', help="")
parser.add_argument('-pp', action='store_true', help="")


def download(args):
	user = zuser.Zindian(args.username)
	user.which_challenge
	user.select_a_challenge()
	user.download_dataset(args.data)

def preprocessing(args):
	additionnal = glob.glob("data/latest_keywords/*/*.wav")

	add = pd.DataFrame({'fn': additionnal})
	add['target'] = add['fn'].apply(lambda x: x.split('/')[-2])
	add.to_csv(args.add + 'Train.csv', index=False)

	train = pd.read_csv(args.data + 'Train.csv')
	train['fn'] = 'data/' + train['fn']
	train.rename(columns = {'label': 'target'}, inplace=True)
	train.to_csv(args.base + 'Train.csv', index=False)


	train = pd.concat([train, add], axis=0)
	train.to_csv(args.to + 'Train.csv', index=False)

	subs = pd.read_csv(args.data + 'SampleSubmission.csv')
	subs['fn'] = 'data/' + subs['fn']
	subs.to_csv(args.to + 'SampleSubmission.csv', index=False)

def info(args):
	train = pd.read_csv(args.base + 'Train.csv')
	full = pd.read_csv(args.to + 'Train.csv')
	add = pd.read_csv(args.add + 'Train.csv')

	lung_words = add.target.unique()
	eng_words = [w for w in train.target.unique() if w not in lung_words]

	dicts = {"base data": train, "add data": add, "full data": full}

	with open("data/info.txt", "wb") as f:
		for name, df in dicts.items():
			info = f"##{name}##. \nIt contains {n} unique classes .\n"
			f.write(info)
			f.write("\n")
		f.close()



def splitter(args):
	df = pd.read_csv(args.use + 'Train.csv')
	path = args.use + args.strategy + '/'

	if args.strategy=='tts':
		train, val = train_test_split(df, test_size=args.split, stratify=df['target'], random_state=args.seed)
		
		os.makedirs(path)

		train.to_csv(path + 'Train.csv', index=False)
		val.to_csv(path + 'Val.csv', index=False)
	else:
		skf = StratifiedKFold(args.n_splits)
		for i, (tr,vr) in enumerate(skf.split(df, df['target'])):
			train = df.loc[tr].reset_index(drop=True)
			val = df.loc[vr].reset_index(drop=True)

			path += f'fold_{i}/'
			os.makedirs(path, exist_ok=True)

			train.to_csv(path + 'Train.csv', index=False)
			val.to_csv(path + 'Val.csv', index=False)


def main(parser):
	args = parser.parse_args()

	assert args.download != args.pp, "Can't use --download and --pp together."

	if args.download: download(args)

	if args.pp: 
		preprocessing(args)
		splitter(args)
		info(args)



if __name__ == '__main__':
	main(parser)

