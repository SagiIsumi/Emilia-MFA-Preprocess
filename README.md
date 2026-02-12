# Emilia Audio Timestamping

## Comments
Even though I tried modelscope/FunASR developed by Alibaba, the performance is not as precise as expected. 
Besides, it labels the timestamping based on ASR, that is to say its transcript might be different from the ground truth.

## Monteral Forced-Aligner

### Objective

This code focus on do the timestamping on Emilia Chinese Audio for the preparation of E2E Chinese Conversation Spoken Model.
We need to align each word in the text with its corresponding phoneme, enabling the model to learn how to pronounce words 
based on the acoustic features it encounters.

### Environment

#### 1. settin Monteral Forced-Aligner
[Official Tutorial](https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html#installation)

```
conda create -n aligner -c conda-forge montreal-forced-aligner
conda activate aligner
conda update -c conda-forge montreal-forced-aligner kalpy kaldi=*=cpu* --update-deps
```
#### 2. setup other packages by uv
please run `uv sync` to sychronize envs by pyproject.toml.
make sure you always run script by `uv run python ...` in conda aligner environment.
your command line should looks like `(aligner)user@root: uv run python ...`

### Execution
1. Please download the Emilia Dataset first, of course you are allowed to use other dataset.
However, you have to make sure the format aligned with Emilia style, the details you refer to Emilia Huggingface model card.

run 
```
uv run python ./src/Audio_pretrin/download.py
```
2. execute 

```
uv run python ./src/Audio_pretrin/timestamping/MFA.py
```

REMEBER! Please changes arguments in setting area in MFA.py.
I know the hard-code is not good-looking, but it is a tailored script for Emilia preprocessing, 
I am lazy use argparse or OmegaConf. That is all.