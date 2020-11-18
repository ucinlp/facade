# Facade

This repository contains code for showing the manipulation of gradients in NLP models as described in the following paper:

> [Gradient-based Analysis of NLP Models is Manipulable](https://arxiv.org/pdf/2010.05419.pdf)
> Junlin Wang, Jens Tuyls, Eric Wallace, Sameer Singh Findings of EMNLP 2020

Bibtex for citations:
```
@inproceedings{facade:emnlp20},  
 author = {Junlin Wang and Jens Tuyls and Eric Wallace and Sameer Singh},  
 title = {Gradient-based Analysis of NLP Models is Manipulable},  
 booktitle = {Findings of Empirical Methods in Natural Language Processing},  
 year = {2020} 
```

# Setup

Clone this repo, and then run `export PYTHONPATH="$PWD"` in the directory that you cloned the repo in.

In addition, install the following packages
```
conda install matplotlib nltk
pip install git+https://github.com/allenai/allennlp-models.git@baf3a1ec3b74273a4ffa2112d37fb88e8b3dd39c
```

Also, we need to clone a modified version of allennlp and install it from source. To do this, run
```
git clone git@github.com:Eric-Wallace/allennlp.git
git checkout -b gradient-regularization
git pull origin gradient-regularization
cd allennlp
pip install .
``` 

# Repository structure

The repository is divided as follows:
- Each of the four tasks we consider have their own folder (SA = Sentiment Analysis, NLI = Natural Language Inference, QA = Question Answering, bios = Biosbias). Each task has three files:
    - `train_facade.py`: used to train the facade model
    - `train_predictive.py`: used to train an off-the-shelf classifier for the task
    - `train-rp.py`: used to finetune a regularized predictive model for the task (see section 3.4 in the paper)

- The analysis folder contains code to analyze the merged model in terms of manipulation of saliency maps, and effects on input reduction and hotflip.