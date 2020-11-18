# Facade

This repository contains code for showing the manipulation of gradients in NLP models as described in the following paper:

> [Gradient-based Analysis of NLP Models is Manipulable](https://arxiv.org/pdf/2010.05419.pdf)
> Junlin Wang, Jens Tuyls, Eric Wallace, Sameer Singh Findings of EMNLP 2020

Bibtex for citations:
```bibtex
@inproceedings{wang-etal-2020-gradient,
    title = "Gradient-based Analysis of {NLP} Models is Manipulable",
    author = "Wang, Junlin  and
      Tuyls, Jens  and
      Wallace, Eric  and
      Singh, Sameer",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.findings-emnlp.24",
    pages = "247--258",
    abstract = "Gradient-based analysis methods, such as saliency map visualizations and adversarial input perturbations, have found widespread use in interpreting neural NLP models due to their simplicity, flexibility, and most importantly, the fact that they directly reflect the model internals. In this paper, however, we demonstrate that the gradients of a model are easily manipulable, and thus bring into question the reliability of gradient-based analyses. In particular, we merge the layers of a target model with a Facade Model that overwhelms the gradients without affecting the predictions. This Facade Model can be trained to have gradients that are misleading and irrelevant to the task, such as focusing only on the stop words in the input. On a variety of NLP tasks (sentiment analysis, NLI, and QA), we show that the merged model effectively fools different analysis tools: saliency maps differ significantly from the original model{'}s, input reduction keeps more irrelevant input tokens, and adversarial perturbations identify unimportant tokens as being highly important.",
}
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