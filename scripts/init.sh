alias pip=pip3
alias python=python3

pip install git+https://github.com/eaedk/testing-zindi-package.git
pip install glob3 tqdm librosa

python pyscripts/init.py -username $1 -download #Connnects the user and download the dataset from zindi
unzip -q data/raw/audio_files.zip -d data/
unzip -q data/raw/AdditionalUtterances.zip -d data/

python pyscripts/init.py -pp