alias pip=pip3
alias python=python3

pip install git+https://github.com/eaedk/testing-zindi-package.git -q
pip install -q python_speech_features
# pip install glob3 tqdm librosa -q
pip install -e .

mkdir data/raw/ data/processed/ data/processed/base/ data/processed/add/

python pyscripts/init.py -username $1 -download #Connnects the user and download the dataset from zindi
unzip -q data/raw/audio_files.zip -d data/
unzip -q data/raw/AdditionalUtterances.zip -d data/