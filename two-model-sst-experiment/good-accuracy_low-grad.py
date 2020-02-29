import sys
import argparse 
import os.path
import argparse
import torch
import math
import operator
from collections import defaultdict

import matplotlib.pyplot as plt
import torch.optim as optim
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import \
    StanfordSentimentTreeBankDatasetReader
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model, BasicClassifier
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper, CnnEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.trainer import Trainer
from allennlp.common.util import lazy_groups_of
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.nn.util import move_to_device
from allennlp.interpret.saliency_interpreters import SaliencyInterpreter, SimpleGradient, IntegratedGradient, SmoothGradient
from allennlp.predictors import Predictor
from allennlp.data.dataset import Batch
import pickle
from allennlp.nn import util
import numpy as np
EMBEDDING_TYPE = "glove" # what type of word embeddings to use
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
def enablePrint():
    sys.stdout = sys.__stdout__
def get_rank(arr):
  arr_idx = sorted([(idx, grad) for idx, grad in enumerate(arr)], key=lambda t: t[1], reverse=True)
  arr_rank = [0 for _ in range(len(arr_idx))]
  for i, (idx, grad) in enumerate(arr_idx):
    arr_rank[idx] = i + 1
  return arr_rank,arr_idx
def get_custom_hinge_loss():
    def custom_hinge_loss(x,k,rank):
        if rank > k:
            return x-x
        else:
            return x
    return custom_hinge_loss
def get_custom_log_loss():
    def custom_log_loss(x):
        return torch.log(x)
    return custom_log_loss
def unfreeze_embed(modules, requiregrad):
  for module in modules:
    if isinstance(module, Embedding):
      module.weight.requires_grad = requiregrad
def get_salient_words(training_instance,pos100,neg100):
    top100 = None
    if training_instance.fields["label"].label == "1":
        # pmi_dict = pmi_ent 
        top100 = pos100
    elif training_instance.fields["label"].label == '0':
        # pmi_dict = pmi_neu
        top100 = neg100
    return top100


class PriorsFineTuner:
  def __init__(self,pmi_pos, pmi_neg, model, reader,train_data,dev_dataset,vocab,args, pos_data,neg_data):
    self.model = model
    self.reader = reader
    self.dev_dataset = dev_dataset
    self.predictor = Predictor.by_name('text_classifier')(self.model, reader)  
    self.simple_gradient_interpreter = SimpleGradient(self.predictor)
    self.args = args
    self.loss = args.loss 
    self.lmbda = args.lmbda
    self.nepochs = args.epochs
    self.batch_size = args.batch_size
    self.outdir = args.outdir
    self.name = args.name
    self.cuda = args.cuda
    self.normal_loss = args.normal_loss
    self.autograd = args.autograd
    self.all_low = args.all_low
    self.lr = args.learning_rate
    self.embedding_operator = args.embedding_operator
    self.normalization = args.normalization
    self.normalization2 = args.normalization2
    self.softmax = args.softmax
    if self.loss == "MSE":
      self.loss_function = torch.nn.MSELoss()
    elif self.loss == "Hinge":
      self.loss_function = get_custom_hinge_loss()
    elif self.loss == "L1":
      self.loss_function = torch.nn.L1Loss()
    if self.cuda == "True":
      self.model.cuda()
    self.pos100 = []
    self.neg100 = []
    trainable_modules = []
    # $$$$$ Create Saving Directory $$$$$
    metadata = "epochs: " + str(self.nepochs) + \
            "\nbatch_size: " + str(self.batch_size) + \
            "\nloss: " + self.loss + \
            "\nlmbda: " + str(self.lmbda) + \
            "\nlr: " + str(self.lr) + \
            "\ncuda: " + self.cuda + \
            "\nautograd: " + str(self.autograd) + \
            "\nall_low: " + str(self.all_low) + \
            "\nembedding_operator: " + str(self.embedding_operator) + \
            "\nnormalization: " + str(self.normalization) + \
            "\nsoftmax: " + str(self.softmax) 
    dir_name = self.name
    self.outdir = os.path.join(self.args.outdir, dir_name)
    print(self.outdir)
    try:
        os.mkdir(self.outdir)
    except:
        print('directory already created')
    trainable_modules = torch.nn.ModuleList(trainable_modules)  
    self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr) #model.parameters()
    # self.optimizer = torch.optim.SGD(self.model.parameters(),lr=self.lr) 
    torch.autograd.set_detect_anomaly(True)
    
    self.train_dataset = train_data
    # about 52% is positive
    self.batched_training_instances = [self.train_dataset[i:i + self.batch_size] for i in range(0, len(self.train_dataset), self.batch_size)]
    self.batched_training_instances_test = [self.train_dataset[i:i + 128] for i in range(0, len(self.train_dataset), 128)]
    self.batched_dev_instances = [self.dev_dataset[i:i + 128] for i in range(0, len(self.dev_dataset), 128)]
    self.vocab = vocab
    # self.iterator = BucketIterator(batch_size=32, sorting_keys=[("tokens", "num_tokens")])
    # self.iterator.index_with(vocab)
    self.acc = []
    self.grad_mags = []
    self.mean_grads = []
    self.high_grads = []
    self.ranks = []
    self.take_notes(-1,0)
    self.get_avg_grad(-1,-1,self.model,self.vocab,self.outdir)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model.train()
    f1 = open(os.path.join(self.outdir,"highest_grad.txt"), "w")
    f1.close()
    f1 = open(os.path.join(self.outdir,"gradient_mags.txt"), "w")
    f1.close()
    f1 = open(os.path.join(self.outdir,"accuracy_pmi.txt"), "w")
    f1.close()
    f1 = open(os.path.join(self.outdir,"ranks.txt"), "w")
    f1.close()
    with open(os.path.join(self.outdir,"metadata.txt"), "w") as myfile:
      myfile.write(metadata)
    print(self.model.get_metrics())
  def fine_tune(self):  
    propagate = True
    unfreeze_embed(self.model.modules(),True) # unfreeze the embedding  
    np.random.seed(42)
    np.random.shuffle(self.batched_training_instances)
    for ep in range(self.nepochs):
      for idx, training_instances  in enumerate(self.batched_training_instances):
        # grad_input_1 => hypothesis
        # grad_input_2 => premise 
        print()
        print()
        print(idx)
        data = Batch(training_instances)
        data.index_instances(self.vocab)
        model_input = data.as_tensor_dict()
        print(model_input)
        if self.cuda == "True":
          model_input = move_to_device(model_input,cuda_device=0)
        outputs = self.model(**model_input)
        new_instances = []
        for instance, output in zip(training_instances , outputs['probs']):
          new_instances.append(self.predictor.predictions_to_labeled_instances(instance, { 'probs': output.cpu().detach().numpy() })[0])
        variables = {"fn":get_salient_words, "fn2":get_rank, "lmbda":self.lmbda,"pos100":self.pos100,"neg100":self.neg100,"training_instances":training_instances}
        print("----------")
        # blockPrint()
        summed_grad, grad_mag, highest_grad,mean_grad = self.simple_gradient_interpreter.saliency_interpret_from_instances_2_model_sst(new_instances, self.embedding_operator, self.normalization,self.normalization2, self.softmax, self.cuda, self.autograd,self.all_low)
        self.grad_mags.append(grad_mag)
        self.high_grads.append(highest_grad)
        self.mean_grads.append(mean_grad)
        for gradient in grad_mag:
          temp = [(idx, grad) for idx, grad in enumerate(gradient)]
          temp.sort(key=lambda t: t[1], reverse=True)
          rank = [i for i, (idx, grad) in enumerate(temp) if idx == 0][0]
          self.ranks.append(rank)
        # # enablePrint()
        print("----------")
        print(summed_grad)
        print("lambda * regularized loss:",float(self.lmbda)*summed_grad.cpu().detach().numpy(), "+ model loss:",outputs["loss"].cpu().detach().numpy())
        # print(self.vocab.get_token_index("0", "labels"))
        # print(outputs["logits"],outputs['loss'])
        # print(outputs["probs"], training_instances[0]["label"].label,training_instances[0]["label"]._label_id)
        # print(self.model.get_metrics())
        if self.all_low == "False":
          summed_grad = self.loss_function(summed_grad.unsqueeze(0), torch.ones(1).cuda() if self.cuda =="True" else torch.ones(1))
          print("MSEd gradloss:",summed_grad)
        embedding_layer = util.find_embedding_layer(self.model)
        regularized_loss =  outputs["loss"] +float(self.lmbda)*summed_grad
        print("final loss:",regularized_loss.cpu().detach().numpy())
        self.model.train()
        if propagate:
          self.optimizer.zero_grad()
          regularized_loss.backward()
          # print("after pt ...............")
          # for module in self.model.parameters():
          #   print("parameter gradient is:")
          #   print(module.grad)
          # count = 0
          # for j in embedding_layer.weight.grad:
          #   for z in j:
          #     if (z.cpu().detach().numpy() != 0.):
          #       count += 1
          # print("how many grads change:",count)
          # exit(0)
          self.optimizer.step()
        # unfreeze_embed(self.model.modules(),True) # unfreeze the embedding  

        if (idx % (600//self.batch_size)) == 0:
            self.take_notes(ep,idx)
        # print(torch.cuda.memory_summary())

      des = "attack_ep" + str(ep)
      model_path = "models/small_grad_high_acc_lstm/" + des + "model.th"
      vocab_path = "models/small_grad_high_acc_lstm/" + des + "sst_vocab"
      with open(model_path, 'wb') as f:
        torch.save(self.model.state_dict(), f)
      self.vocab.save_to_files(vocab_path)    
      self.take_notes(ep,idx)
      self.get_avg_grad(ep,idx,self.model,self.vocab,self.outdir)
    

  def take_notes(self,ep,idx):
    #   self.get_avg_grad(ep,idx,self.model, self.vocab,self.outdir)
    self.get_accuracy(self.model, self.dev_dataset, self.vocab, self.acc,self.outdir)
    with open(os.path.join(self.outdir,"accuracy_pmi.txt"), "a") as myfile:
        myfile.write("\nEpoch#%d Iteration%d # Accuracy: %f"%(ep,idx,self.acc[-1]))
    mean_grad = 0
    if len(self.grad_mags) != 0:
      with open(os.path.join(self.outdir,"gradient_mags.txt"), "a") as myfile:
          for each_group in self.grad_mags:
            for each in each_group:
              written = " ".join([str(x) for x in each])
              myfile.write("\nEpoch#%d Batch#%d gradients: %s"%(ep,idx,written))
          self.grad_mags = []
      high_grad = np.max(self.high_grads)
      mean_grad = np.mean(self.mean_grads)
      with open(os.path.join(self.outdir,"highest_grad.txt"), "a") as myfile:
          myfile.write("\nEpoch#%d mean gradients: %s, highest gradient: %s"%(ep,str(mean_grad), str(high_grad)))
          self.high_grads = []
          self.mean_grads = []
      with open(os.path.join(self.outdir,"ranks.txt"), "a") as myfile:
        for each_r in self.ranks:
          myfile.write("\nEpoch#%d Batch#%d rank: %d"%(ep,idx,each_r))
        self.ranks = []


  def get_accuracy(self,model, dev_data, vocab, acc,outdir):       
    model.get_metrics(reset=True)
    model.eval() # model should be in eval() already, but just in case
    iterator = BucketIterator(batch_size=128, sorting_keys=[("tokens", "num_tokens")])
    iterator.index_with(vocab)        
    for batch in lazy_groups_of(iterator(dev_data, num_epochs=1, shuffle=False), group_size=1):
      if self.cuda == "True":
        batch = move_to_device(batch[0], cuda_device=0)
      else:
        batch = batch[0]
      model(batch['tokens'], batch['label'])
    acc.append(model.get_metrics(True)['accuracy'])
    model.train()
    print("Accuracy:", acc[-1])

  def get_avg_grad(self,ep,idx,model, vocab,outdir):       
    model.get_metrics(reset=True)
    model.eval() # model should be in eval() already, but just in case
    highest_grad_dev = []
    mean_grad_dev = np.float(0)
    highest_grad_train = []
    mean_grad_train = np.float(0)
    for i, training_instances  in enumerate(self.batched_dev_instances):
      data = Batch(training_instances)
      data.index_instances(self.vocab)
      model_input = data.as_tensor_dict()
      if self.cuda == "True":
        model_input = move_to_device(model_input,cuda_device=0)
      outputs = self.model(**model_input)
      new_instances = []
      for instance, output in zip(training_instances , outputs['probs']):
        new_instances.append(self.predictor.predictions_to_labeled_instances(instance, { 'probs': output.cpu().detach().numpy() })[0])
      summed_grad, grad_mag, highest_grad,mean_grad = self.simple_gradient_interpreter.saliency_interpret_from_instances_2_model_sst(new_instances, self.embedding_operator, self.normalization, self.normalization2, self.softmax, self.cuda, self.autograd,self.all_low)
      summed_grad.detach()
      highest_grad_dev.append(highest_grad)
      mean_grad_dev += mean_grad
      self.optimizer.zero_grad()

    highest_grad_dev = np.max(highest_grad_dev)
    mean_grad_dev /= len(self.batched_dev_instances)

    for i, training_instances  in enumerate(self.batched_training_instances_test):
      data = Batch(training_instances)
      data.index_instances(self.vocab)
      model_input = data.as_tensor_dict()
      if self.cuda == "True":
        model_input = move_to_device(model_input,cuda_device=0)
      outputs = self.model(**model_input)
      new_instances = []
      for instance, output in zip(training_instances , outputs['probs']):
        new_instances.append(self.predictor.predictions_to_labeled_instances(instance, { 'probs': output.cpu().detach().numpy() })[0])
      summed_grad, grad_mag, highest_grad,mean_grad = self.simple_gradient_interpreter.saliency_interpret_from_instances_2_model_sst(new_instances, self.embedding_operator, self.normalization, self.normalization2, self.softmax, self.cuda, self.autograd,self.all_low)
      summed_grad.detach()
      highest_grad_train.append(highest_grad)
      mean_grad_train += mean_grad
      self.optimizer.zero_grad()
    highest_grad_train = np.max(highest_grad_train)
    mean_grad_train /= len(self.batched_training_instances_test)
    model.train()
    with open(os.path.join(self.outdir,"highest_grad_dataset.txt"), "a") as myfile:
      myfile.write("\nEpoch#{} Iteration{} # highest/mean grad mag: {:.8E} ; {:.8E} ; {:.8E} ; {:.8E}".format(ep,idx,highest_grad_dev,mean_grad_dev, highest_grad_train,mean_grad_train))
    torch.cuda.empty_cache()
def main():
    args = argument_parsing()
    # load the binary SST dataset.
    single_id_indexer = SingleIdTokenIndexer(lowercase_tokens=True) # word tokenizer
    # use_subtrees gives us a bit of extra data by breaking down each example into sub sentences.
    reader = StanfordSentimentTreeBankDatasetReader(granularity="2-class",                                  
                                                    token_indexers={"tokens": single_id_indexer},
                                                    add_synthetic_bias=False)

    train_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/train.txt')

    dev_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/dev.txt')
    
    vocab = Vocabulary.from_instances(train_data)

    # Randomly initialize vectors
    if EMBEDDING_TYPE == "None":
        # token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=300)
        # word_embedding_dim = 300
        token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=10)
        word_embedding_dim = 10

    # Load word2vec vectors
    elif EMBEDDING_TYPE == "glove":
        embedding_path = "embeddings/glove.840B.300d.txt"
        embedding_path = os.path.join(os.getcwd(),embedding_path)
        weight = _read_pretrained_embeddings_file(embedding_path,
                                                  embedding_dim=300,
                                                  vocab=vocab,
                                                  namespace="tokens")
        token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                    embedding_dim=300,
                                    weight=weight,
                                    trainable=True)
        word_embedding_dim = 300
    # Initialize model, cuda(), and optimizer
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    # encoder = CnnEncoder(embedding_dim=word_embedding_dim,
    #                      num_filters=100,
    #                      ngram_filter_sizes=(1,2,3))
    encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(word_embedding_dim,
                                                   hidden_size=512,
                                                   num_layers=2,
                                                   batch_first=True))
    model = BasicClassifier(vocab, word_embeddings, encoder)
    # model.cuda()
    iterator = BucketIterator(batch_size=32, sorting_keys=[("tokens", "num_tokens")])
    iterator.index_with(vocab)
    # # where to save the model
    model_path = "/tmp/" + EMBEDDING_TYPE + "_" + "model_rnn.th"
    vocab_path = "/tmp/" + EMBEDDING_TYPE + "_" + "vocab3"
    # if the model already exists (its been trained), load the pre-trained weights and vocabulary
    if os.path.isfile(model_path):
        vocab = Vocabulary.from_files(vocab_path)
        model = BasicClassifier(vocab, word_embeddings, encoder)
        with open(model_path, 'rb') as f:
            model.load_state_dict(torch.load(f))
    # otherwise train model from scratch and save its weights
    else:
        optimizer = optim.Adam(model.parameters())
        trainer = Trainer(model=model,
                          optimizer=optimizer,
                          iterator=iterator,
                          train_dataset=train_data,
                          validation_dataset=dev_data,
                          num_epochs=8,
                          patience=1)
        trainer.train()
        with open(model_path, 'wb') as f:
            torch.save(model.state_dict(), f)
        vocab.save_to_files(vocab_path)    

    pmi_pos = []
    pmi_neg = []
    pos_data = []
    neg_data = []
    fine_tuner = PriorsFineTuner(pmi_pos,pmi_neg,model, reader, train_data, dev_data, vocab, args, pos_data,neg_data)
    fine_tuner.fine_tune()
    
def argument_parsing():
    parser = argparse.ArgumentParser(description='One argparser')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--lmbda', type=float, help='Lambda of regularized loss')
    parser.add_argument('--loss', type=str, help='Loss function')
    parser.add_argument('--normal_loss', type=str, help='Decide to use normal loss or not')
    parser.add_argument('--outdir', type=str, help='Output dir')
    parser.add_argument('--name', type=str, help='name')
    parser.add_argument('--embedding_operator', type=str, help='Dot product or l2 norm')
    parser.add_argument('--normalization', type=str, help='L1 norm or l2 norm')
    parser.add_argument('--normalization2', type=str, help='L2 norm or l2 norm')
    parser.add_argument('--softmax', type=str, help='Decide to use softmax or not')
    parser.add_argument('--cuda', type=str, help='Use cuda')
    parser.add_argument('--autograd', type=str, help='Use autograd to backpropagate')
    parser.add_argument('--all_low', type=str, help='want to make all gradients low?')
    args = parser.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    main()
