# Computes the PMI of each word in the SNLI training set as per the Annotation Artifacts paper. Results closely but do not perfectly match theirs.
from allennlp.data.dataset_readers.snli import SnliReader
from allennlp.data.token_indexers import SingleIdTokenIndexer, ELMoTokenCharactersIndexer
from allennlp.data.dataset_readers.hans import HansReader
from allennlp.models import DecomposableAttention
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
from allennlp.interpret.saliency_interpreters import SaliencyInterpreter, SimpleGradient, IntegratedGradient, SmoothGradient
import matplotlib.pyplot as plt 
from collections import defaultdict
import math
import operator
import pickle
import os 
import scipy
import torch
import argparse 
from allennlp.modules.text_field_embedders.basic_text_field_embedder import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
import numpy as np
from allennlp.data.dataset import Batch
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.util import move_to_device
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.common.util import lazy_groups_of
from torch.utils.hooks import RemovableHandle
from torch import Tensor
from typing import List
from allennlp.nn import util
def get_rank(arr):
  arr_idx = sorted([(idx, grad) for idx, grad in enumerate(arr)], key=lambda t: t[1], reverse=True)
  arr_rank = [0 for _ in range(len(arr_idx))]
  for i, (idx, grad) in enumerate(arr_idx):
    arr_rank[idx] = i + 1
  return arr_rank,arr_idx

class PriorsFineTuner:
  def __init__(self,pmi_ent, pmi_neu, pmi_con, dev_dataset, reader,hans_data,hans_vocab,args):
    # self.model = load_archive('https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-2017.09.04.tar.gz').model
    self.model = load_archive('https://allennlp.s3-us-west-2.amazonaws.com/models/esim-glove-snli-2019.04.23.tar.gz').model
    self.reader = reader
    self.dev_dataset = dev_dataset
    self.predictor = Predictor.by_name('textual-entailment')(self.model, reader)  
    self.simple_gradient_interpreter = SimpleGradient(self.predictor)
    self.args = args
    self.loss = args.loss 
    self.lmbda = args.lmbda
    self.nepochs = args.epochs
    self.batch_size = args.batch_size
    self.outdir = args.outdir
    self.cuda = args.cuda
    if self.loss == "MSE":
      self.loss_function = torch.nn.MSELoss()
    elif self.loss == "Hinge":
      # self.loss_function = torch.nn.MarginRankingLoss()
      self.loss_function = get_custom_hinge_loss()
    elif self.loss == "L1":
      self.loss_function = torch.nn.L1Loss()
  # ig_interpreter = IntegratedGradient(predictor)
    trainable_modules = []
    # for module in model.modules():
    #   # print(module)
    #   # print(type(module))
    #   if not isinstance(module, torch.nn.Embedding):    
    #     if not isinstance(module,BasicTextFieldEmbedder):   
    #       if not isinstance(module,Embedding):   
    #         trainable_modules.append(module)
    # print(len(trainable_modules))
    # print(model.parameters())
    self.hans_data = hans_data
    self.hans_vocab = hans_vocab
    trainable_modules = torch.nn.ModuleList(trainable_modules)  
    self.optimizer = torch.optim.Adam(self.model.parameters()) #model.parameters()
    #loss_func = torch.nn.MSELoss()
    prem_corr = []
    hyp_corr = []
    x = []

    ent100 = sorted(pmi_ent.items(), key=operator.itemgetter(1),reverse=True)[:100]
    self.ent100 = {x[0]:[] for x in ent100}
    con100 = sorted(pmi_con.items(), key=operator.itemgetter(1),reverse=True)[:100]
    self.con100 =  {x[0]:[] for x in con100}
    neu100 = sorted(pmi_neu.items(), key=operator.itemgetter(1),reverse=True)[:100]
    self.neu100 =  {x[0]:[] for x in neu100}

    torch.autograd.set_detect_anomaly(True)
    
    self.train_dataset = reader.read('data/snli_1.0_train.jsonl') 
    self.batched_training_instances = [self.train_dataset[i:i + self.batch_size] for i in range(0, len(self.train_dataset), self.batch_size)]
    if self.cuda == "True":
      self.model.cuda()
    # vocab = Vocabulary.from_instances(train_dataset)
    self.vocab = self.model.vocab
    self.acc = []
    self.get_accuracy(self.model, dev_dataset, self.vocab, self.acc,self.cuda)
    # with open("sanity_checks/accuracy_pmi.txt", "w") as myfile:
    #   myfile.write("\nEpoch#%d Iteration%d # Accuracy: %f"%(0,0,acc[-1]))
    print(len(self.train_dataset))

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model.train()
    f1 = open("snli_hans_acc.txt","w")
    f1.close()
    f2 = open("sanity_checks/embedding_check.txt", "w")
    f2.close()
    
  def fine_tune(self):    
    for ep in range(self.nepochs):
      for idx, training_instances  in enumerate(self.batched_training_instances):
        # grad_input_1 => hypothesis
        # grad_input_2 => premise 
        print()
        print()
        print(idx)
        freeze_embed(self.model.modules(),True)
        # self.optimizer.zero_grad()
        data = Batch(training_instances)
        data.index_instances(self.vocab)
        model_input = data.as_tensor_dict()
        if self.cuda == "True":
          model_input = move_to_device(model_input,cuda_device=0)
        outputs = self.model(**model_input)
        print("label_logits:",outputs["label_logits"].cpu().detach().numpy())
        # print("outputs:",outputs["label_probs"])
        # sanitize_instances(predictor, outputs, instance)  
        # new_instances = predictor.predictions_to_labeled_instances(instance[0], outputs)
        new_instances = []
        for instance, output in zip(training_instances , outputs['label_logits']):
          new_instances.append(self.predictor.predictions_to_labeled_instances(instance, { 'label_logits': output.cpu().detach().numpy() })[0])

        
        # get the variables ready
        variables = {"fn":get_salient_words, "fn2":get_rank, "lmbda":self.lmbda,"ent100":self.ent100,"con100":self.con100,"neu100":self.neu100,"training_instances":training_instances}
        # get gradient (this times the embedding and has all the normalization)
        print("----------")
        summed_grad, model_losses, grad_mag,gradient,gradient2,embedding_gradients = self.simple_gradient_interpreter.saliency_interpret_from_instances_autograd(new_instances, "dot_product", "None", variables,"None", "False",self.cuda)
        print("----------")
        print(instance["hypothesis"])

        # summed_gradient: List[Tensor] = []
        # def hook_layers(module, grad_in, grad_out):
        #     summed_gradient.append(grad_out[0])
        embedding_layer = util.find_embedding_layer(self.model)
        # print(embedding_layer)
        # efhooks: List[RemovableHandle] =[ embedding_layer.register_backward_hook(hook_layers)]

        # print(summed_gradient)

        propagate = True
        print("regularized loss:",summed_grad.cpu().detach().numpy(), "model loss:",outputs["loss"].cpu().detach().numpy())
        if (summed_grad.cpu().detach().numpy() == np.array([0])) or (np.isnan(summed_grad.cpu().detach().numpy()[0])):
          print("no regularized loss")
          regularized_loss = outputs["loss"]
        else:
          if idx>62:
            print("this loss")
            regularized_loss = outputs["loss"]
          else:
            # regularized_loss = (1-self.lmbda)*summed_grad + self.lmbda*outputs["loss"]
            regularized_loss = 100*summed_grad + outputs["loss"]
        print("final loss:",regularized_loss.cpu().detach().numpy())
        if propagate:
          print("backward begains...")
          # print("embedding layer shape",embedding_layer.weight.shape)
          # print("before embedding layer grad",embedding_layer.weight.grad)
          # for module in self.model.parameters():
          #   print("parameter gradient is:")
          #   print(module.grad)

          self.optimizer.zero_grad()
          # freeze_embed(self.model.modules(),False)
          # print("mid pt.................")
          # for module in self.model.parameters():
          #   print("parameter gradient is:")
          #   print(module.grad)
          regularized_loss.backward()

          # print("after pt ...............")
          # for module in self.model.parameters():
          #   print("parameter gradient is:")
          #   print(module.grad)
          count = 0
          # for j in embedding_layer.weight.grad:
          #   for z in j:
          #     if (z.cpu().detach().numpy() != 0.):
          #       print(j)
          #       count += 1
          # print("how many grads change:",count)
          self.optimizer.step()
        if (idx % (1000//self.batch_size)) == 0:
          self.get_accuracy(self.model, self.dev_dataset, self.vocab, self.acc,self.cuda)
          with open("sanity_checks/accuracy_pmi.txt", "a") as myfile:
            myfile.write("\nEpoch#%d Iteration%d # Accuracy: %f"%(ep,idx,self.acc[-1]))
          self.record(self.ent100,self.con100,self.neu100)
          self.get_hans_accuracy(self.model, self.hans_data, self.hans_vocab,self.cuda)
    self.record(self.ent100,self.con100,self.neu100)

  def get_accuracy(self,model, dev_data, vocab, acc,cuda):       
    model.get_metrics(reset=True)
    model.eval() # model should be in eval() already, but just in case
    iterator = BucketIterator(batch_size=128, sorting_keys=[("premise", "num_tokens"), ("hypothesis", "num_tokens")])
    iterator.index_with(vocab)        
    for batch in lazy_groups_of(iterator(dev_data, num_epochs=1, shuffle=False), group_size=1):
      if cuda == "True":
        batch = move_to_device(batch[0], cuda_device=0)
      else:
        batch = batch[0]
      model(batch['premise'], batch['hypothesis'], batch['label'], batch['metadata'])
    acc.append(model.get_metrics()['accuracy'])
    model.train()
    print("Accuracy:", acc[-1])
  def record(self,ent100,con100,neu100):
    with open("sanity_checks/gradient_change_ent.txt", "w") as myfile:
      for each in ent100:
        myfile.write("\n%s: "%(each))
        for num,rank,grads in ent100[each]:
          grads = [str(x) for x in grads]
          grads_str = " ".join(grads)
          myfile.write("%f,%d" %(num,int(rank)))
      
    with open("sanity_checks/gradient_change_con.txt", "w") as myfile:
      for each in con100:
        myfile.write("\n%s: "%(each))
        for num,rank,grads in con100[each]:
          grads = [str(x) for x in grads]
          grads_str = " ".join(grads)
          myfile.write("%f,%d; " %(num,int(rank)))
    with open("sanity_checks/gradient_change_neu.txt", "w") as myfile:
      for each in neu100:
        myfile.write("\n%s: "%(each))
        for num,rank,grads in neu100[each]:
          grads = [str(x) for x in grads]
          grads_str = " ".join(grads)
          myfile.write("%f,%d; " %(num,int(rank)))
  def get_hans_accuracy(self,model, hans_data, hans_vocab,cuda):
    model.eval()
    iterator = BucketIterator(batch_size=128, sorting_keys=[("premise", "num_tokens"), ("hypothesis", "num_tokens")])
    iterator.index_with(hans_vocab)  
    # [entailment, contradiction, neutral]

    const_ent = 0
    const_ent_total = 0
    const_non_ent = 0
    const_non_ent_total = 0

    lex_ent = 0
    lex_ent_total = 0
    lex_non_ent = 0
    lex_non_ent_total = 0

    sub_ent = 0
    sub_ent_total = 0
    sub_non_ent = 0
    sub_non_ent_total = 0

    for batch in lazy_groups_of(iterator(hans_data, num_epochs=1, shuffle=False), group_size=1):
      batch = batch[0]
      if cuda == "True":
        batch = move_to_device(batch, cuda_device=0)
      output = model(batch['premise'], batch['hypothesis'], batch['label'])
      max_idx = torch.argmax(output['label_probs'], axis=1)
      for top_label, true_label, heuristic in zip(max_idx, batch['label'], batch['heuristic']):
        top_label = int(top_label.item())
        heuristic = hans_vocab.get_token_from_index(heuristic.item(), 'labels')
        true_label = hans_vocab.get_token_from_index(true_label.item(), 'labels')
        if heuristic == "lexical_overlap":
          if true_label == "entailment":
            lex_ent += 1 if top_label == 0 else 0
            lex_ent_total += 1
          else:
            lex_non_ent += 1 if top_label != 0 else 0
            lex_non_ent_total += 1
        elif heuristic == "subsequence":
          if true_label == "entailment":
            sub_ent += 1 if top_label == 0 else 0
            sub_ent_total += 1
          else:
            sub_non_ent += 1 if top_label != 0 else 0
            sub_non_ent_total += 1 
        elif heuristic == "constituent":
          if true_label == "entailment":
            const_ent += 1 if top_label == 0 else 0
            const_ent_total += 1
          else:
            const_non_ent += 1 if top_label != 0 else 0
            const_non_ent_total += 1 
        else:
          raise Exception("Did not recognize heuristic!")

    const_ent_acc = const_ent/const_ent_total
    const_non_ent_acc = const_non_ent/const_non_ent_total

    lex_ent_acc = lex_ent/lex_ent_total
    lex_non_ent_acc = lex_non_ent/lex_non_ent_total 

    sub_ent_acc = sub_ent/sub_ent_total
    sub_non_ent_acc = sub_non_ent/sub_non_ent_total 
    model.train()
    with open("snli_hans_acc.txt", "a") as f:
      f.write("Constituent Entailment: %f\n" % (const_ent_acc))
      f.write("Constituent Non-Entailment: %f\n" % (const_non_ent_acc))
      f.write("Lexical Overlap Entailment: %f\n" % (lex_ent_acc))
      f.write("Lexical Overlap Non-Entailment: %f\n" % (lex_non_ent_acc))
      f.write("Subsequence Entailment: %f\n" % (sub_ent_acc))
      f.write("Subsequence Non-Entailment: %f\n" % (sub_non_ent_acc))

  # outputs = model.forward_on_instances(dev_dataset)
  # total = len(dev_dataset)
  # num_right = 0
  # for j,each in enumerate(outputs):
  #   if np.argmax(each["label_probs"]) == dev_dataset[j].fields["label"]._label_id:
  #     num_right+=1
  # print("After Accuracy:",num_right/total)
def freeze_embed(modules, requiregrad):
  for module in modules:
    if isinstance(module, Embedding):
      module.weight.requires_grad = requiregrad
def get_salient_words(training_instance,ent100,con100,neu100):
    top100 = None
    if training_instance.fields["label"].label == 'entailment':
      # pmi_dict = pmi_ent 
      top100 = ent100
    elif training_instance.fields["label"].label == 'neutral':
      # pmi_dict = pmi_neu
      top100 = neu100
    elif training_instance.fields["label"].label == 'contradiction':
      # pmi_dict = pmi_con
      top100 = con100
    return top100


def sanitize_instances(predictor, outputs, training_instances):
    outputs["label_logits"] = outputs["label_logits"].cpu().detach().numpy()
    # new_instances = []
    # for idx,instance in enumerate(training_instances):
    #     tmp = {"label_logits": move_to_device(outputs["label_logits"][idx],0)}
    #     new_instances.append(predictor.predictions_to_labeled_instances(instance,tmp)[0])
    # return new_instances
def get_custom_hinge_loss():
    def custom_hinge_loss(x,k,rank):
        if rank > k:
            return x-x
        else:
            return x
    return custom_hinge_loss


def main():       
    args = argument_parsing()            
    p_word = defaultdict(lambda: 100.0) # add 100 smoothing
    p_class = defaultdict(lambda: 0.0) 
    p_word_class = defaultdict(lambda: 0.0)

    total_words = 0.0
    total_examples = 0.0

    single_id = SingleIdTokenIndexer(lowercase_tokens=True)
    reader = SnliReader(token_indexers={'tokens': single_id})    
    dev_dataset = reader.read('data/snli_1.0_dev.jsonl')

    hans_reader = HansReader(token_indexers={'tokens': single_id})
    hans_data = hans_reader.read('data/heuristics_evaluation_set.txt')
    hans_vocab = Vocabulary.from_instances(hans_data)
    pmi_ent = {}    
    pmi_con = {}    
    pmi_neu = {}

    if os.path.isfile('ent.pkl') and os.path.isfile('neu.pkl') and os.path.isfile('con.pkl'):
      with open('ent.pkl', 'rb') as ent_file:
        pmi_ent = pickle.load(ent_file)
      with open('con.pkl', 'rb') as con_file:
        pmi_con = pickle.load(con_file)
      with open('neu.pkl', 'rb') as neu_file:
        pmi_neu = pickle.load(neu_file)  

    tt = PriorsFineTuner(pmi_ent, pmi_neu, pmi_con, dev_dataset, reader,hans_data,hans_vocab,args)
    tt.fine_tune()
def argument_parsing():
  parser = argparse.ArgumentParser(description='One argparser')
  parser.add_argument('--batch_size', type=int, help='Batch size')
  parser.add_argument('--epochs', type=int, help='Number of epochs')
  parser.add_argument('--lmbda', type=float, help='Lambda of regularized loss')
  parser.add_argument('--loss', type=str, help='Loss function')
  parser.add_argument('--outdir', type=str, help='Output dir')
  parser.add_argument('--cuda', type=str, help='Use cuda')
  # parser.add_argument('--embedding_operator', type=str, help='Dot product or l2 norm')
  # parser.add_argument('--normalization', type=str, help='L1 norm or l2 norm')
  # parser.add_argument('--normalization2', type=str, help='L2 norm or l2 norm')
  args = parser.parse_args()
  print(args)
  return args
if __name__ == '__main__':
    main()
