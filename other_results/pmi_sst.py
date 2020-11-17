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
# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
def get_rank(arr):
  arr_idx = sorted([(idx, grad) for idx, grad in enumerate(arr)], key=lambda t: t[1], reverse=True)
  arr_rank = [0 for _ in range(len(arr_idx))]
  for i, (idx, grad) in enumerate(arr_idx):
    arr_rank[idx] = i + 1
  return arr_rank,arr_idx

class PriorsFineTuner:
  def __init__(self,pmi_pos, pmi_neg, model, reader,train_data,dev_dataset,vocab,args, pos_data,neg_data):
    self.model = model
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
    self.normal_loss = args.normal_loss
    self.autograd = args.autograd
    self.all_low = args.all_low
    self.lr = args.learning_rate
    if self.loss == "MSE":
      self.loss_function = torch.nn.MSELoss()
    elif self.loss == "Hinge":
      # self.loss_function = torch.nn.MarginRankingLoss()
      self.loss_function = get_custom_hinge_loss()
    elif self.loss == "L1":
      self.loss_function = torch.nn.L1Loss()
    if self.cuda == "True":
      self.model.cuda()
  # ig_interpreter = IntegratedGradient(predictor)
    trainable_modules = []
    dir_name = "pmi-cnn-batch_size" + str(self.batch_size) + \
            "__lmbda-" + str(self.lmbda) + \
            "__loss-" + self.loss + \
              "__cuda-" + self.cuda + \
              "lr-" + str(self.lr)
    dir_name = "test_autograd"
    self.outdir = os.path.join(self.args.outdir, dir_name)
    print(self.outdir)
    try:
        os.mkdir(self.outdir)
    except:
        print('directory already created')

    trainable_modules = torch.nn.ModuleList(trainable_modules)  
    self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr) #model.parameters()
    # self.optimizer = torch.optim.SGD(self.model.parameters(),lr=self.lr) 
    #loss_func = torch.nn.MSELoss()

    pos100 = sorted(pmi_pos.items(), key=operator.itemgetter(1),reverse=True)[:100]
    self.pos100 = {x[0]:[] for x in pos100}
    neg100 = sorted(pmi_neg.items(), key=operator.itemgetter(1),reverse=True)[:100]
    self.neg100 =  {x[0]:[] for x in neg100}

    self.pos_op = {x:[] for x in pos_data}
    self.neg_op = {x:[] for x in neg_data}

    # self.pos100 = self.pos_op
    # self.neg100 = self.neg_op
    # print(self.pos100)
    # print(self.neg100)
    torch.autograd.set_detect_anomaly(True)
    
    self.train_dataset =train_data
    self.batched_training_instances = [self.train_dataset[i:i + self.batch_size] for i in range(0, len(self.train_dataset), self.batch_size)]
    self.batched_training_instances_test = [self.train_dataset[i:i + 128] for i in range(0, len(self.train_dataset), 128)]
    self.batched_dev_instances = [self.dev_dataset[i:i + 128] for i in range(0, len(self.dev_dataset), 128)]
    # vocab = Vocabulary.from_instances(train_dataset)
    self.vocab = vocab
    self.acc = []
    self.get_accuracy(self.model, dev_dataset, self.vocab, self.acc,self.outdir,0)
    with open(os.path.join(self.outdir,"accuracy_pmi.txt"), "w") as myfile:
      myfile.write("\nEpoch#%d Iteration%d # Accuracy: %f"%(0,0,self.acc[-1]))
    self.get_avg_grad(0,0,self.model, self.vocab,self.outdir)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model.train()

    f1 = open(os.path.join(self.outdir,"highest_grad.txt"), "w")
    f1.close()
    f2 = open("sanity_checks/embedding_check.txt", "w")
    f2.close()
    
  def fine_tune(self):  
    unfreeze_embed(self.model.modules(),True) # unfreeze the embedding  
    for ep in range(self.nepochs):
      for idx, training_instances  in enumerate(self.batched_training_instances):
        # grad_input_1 => hypothesis
        # grad_input_2 => premise 
        print()
        print()
        print(idx)
        # self.optimizer.zero_grad()
        data = Batch(training_instances)
        data.index_instances(self.vocab)
        model_input = data.as_tensor_dict()
        if self.cuda == "True":
          model_input = move_to_device(model_input,cuda_device=0)
        # print(model_input)
        outputs = self.model(**model_input)
        # print("label_logits:",outputs["label_logits"].cpu().detach().numpy())
        # print("outputs:",outputs["label_probs"])
        # sanitize_instances(predictor, outputs, instance)  
        # new_instances = predictor.predictions_to_labeled_instances(instance[0], outputs)
        new_instances = []
        for instance, output in zip(training_instances , outputs['probs']):
          new_instances.append(self.predictor.predictions_to_labeled_instances(instance, { 'label_logits': output.cpu().detach().numpy() })[0])

        # get the variables ready
        variables = {"fn":get_salient_words, "fn2":get_rank, "lmbda":self.lmbda,"pos100":self.pos100,"neg100":self.neg100,"training_instances":training_instances}
        # get gradient (this times the embedding and has all the normalization)
        print("----------")
        blockPrint()
        summed_grad, model_losses, grad_mag,gradient,gradient2,embedding_gradients,highest_grad = self.simple_gradient_interpreter.saliency_interpret_from_instances_pmi_sst(new_instances, "l2_norm", "None", variables,"None", "False", self.cuda,self.autograd,self.all_low)
        enablePrint()
        print("----------")
        # print(instance["hypothesis"])
        # summed_gradient: List[Tensor] = []
        # def hook_layers(module, grad_in, grad_out):
        #     summed_gradient.append(grad_out[0])
        # embedding_layer = util.find_embedding_layer(self.model)
        # print(embedding_layer)
        # efhooks: List[RemovableHandle] =[ embedding_layer.register_backward_hook(hook_layers)]
        # print(summed_gradient)
        if self.normal_loss == "True":
          propagate = True
        else:
          propagate = False
        print("regularized loss:",summed_grad.cpu().detach().numpy(), "model loss:",outputs["loss"].cpu().detach().numpy())
        if (summed_grad.cpu().detach().numpy() == np.array([0])) or (np.isnan(summed_grad.cpu().detach().numpy()[0])):
          print("no regularized loss")
          regularized_loss = outputs["loss"]
        else:
          # if idx>80:
          #   print("this loss")
          #   regularized_loss = outputs["loss"]
          # else:
            # regularized_loss = (1-self.lmbda)*summed_grad + self.lmbda*outputs["loss"]
          regularized_loss = float(self.lmbda)*summed_grad + outputs["loss"]
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
          # count = 0
          # for j in embedding_layer.weight.grad:
          #   for z in j:
          #     if (z.cpu().detach().numpy() != 0.):
          #       print(j)
          #       count += 1
          # print("how many grads change:",count)
          self.optimizer.step()
          
        unfreeze_embed(self.model.modules(),True) # unfreeze the embedding  
        if (idx % (600//self.batch_size)) == 0:
          self.get_avg_grad(ep,idx,self.model, self.vocab,self.outdir)
          self.get_accuracy(self.model, self.dev_dataset, self.vocab, self.acc,self.outdir)
          with open(os.path.join(self.outdir,"accuracy_pmi.txt"), "a") as myfile:
            myfile.write("\nEpoch#%d Iteration%d # Accuracy: %f"%(ep,idx,self.acc[-1]))
          self.record(self.pos100,self.neg100,highest_grad, self.outdir,ep)
        # print(torch.cuda.memory_summary())
    des = "autograd_"
    model_path = "/tmp/fine_tuned/pmi_sst/" + des + "model.th"
    vocab_path = "/tmp/fine_tuned/pmi_sst/" + des + "sst_vocab"
    with open(model_path, 'wb') as f:
      torch.save(self.model.state_dict(), f)
    self.vocab.save_to_files(vocab_path)    
    self.record(self.pos100,self.neg100,highest_grad,self.outdir,ep)
    self.get_avg_grad(ep,idx,self.model, self.vocab,self.outdir)
    self.get_accuracy(self.model, self.dev_dataset, self.vocab, self.acc,self.outdir)
    with open(os.path.join(self.outdir,"accuracy_pmi.txt"), "a") as myfile:
      myfile.write("\nEpoch#%d Iteration%d # Accuracy: %f"%(ep,idx,self.acc[-1]))
  
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
    acc.append(model.get_metrics()['accuracy'])
    model.train()
    print("Accuracy:", acc[-1])

  def get_avg_grad(self,ep,idx,model, vocab,outdir):       
    model.get_metrics(reset=True)
    # model.eval() # model should be in eval() already, but just in case
    highest_grad_dev = np.float(0)
    mean_grad_dev = np.float(0)
    highest_grad_train = np.float(0)
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
        new_instances.append(self.predictor.predictions_to_labeled_instances(instance, { 'label_logits': output.cpu().detach().numpy() })[0])
      highest, mean,_ = self.simple_gradient_interpreter.saliency_interpret_from_instances_highest(new_instances, "l2_norm", "None", [],"None", "False", self.cuda,self.autograd,self.all_low,True)
      highest_grad_dev += highest
      mean_grad_dev += mean
      outputs["loss"].backward()
      self.optimizer.zero_grad()
      
      
    self.optimizer.zero_grad()
    
    highest_grad_dev /= len(self.dev_dataset)
    mean_grad_dev /= len(self.dev_dataset)
    for i, training_instances  in enumerate(self.batched_training_instances_test):
      data = Batch(training_instances)
      data.index_instances(self.vocab)
      model_input = data.as_tensor_dict()
      if self.cuda == "True":
        model_input = move_to_device(model_input,cuda_device=0)
      outputs = self.model(**model_input)
      new_instances = []
      for instance, output in zip(training_instances , outputs['probs']):
        new_instances.append(self.predictor.predictions_to_labeled_instances(instance, { 'label_logits': output.cpu().detach().numpy() })[0])
      highest, mean,_ = self.simple_gradient_interpreter.saliency_interpret_from_instances_highest(new_instances, "l2_norm", "None", [],"None", "False", self.cuda,self.autograd,self.all_low,True)
      highest_grad_train += highest
      mean_grad_train += mean
      outputs["loss"].backward()
      self.optimizer.zero_grad()
    highest_grad_train /= len(self.dev_dataset)
    mean_grad_train /= len(self.dev_dataset)
    model.train()
    with open(os.path.join(self.outdir,"highest_grad.txt"), "a") as myfile:
      myfile.write("\nEpoch#{} Iteration{} # highest/mean grad mag: {:.8E} ; {:.8E} ; {:.8E} ; {:.8E}".format(ep,idx,highest_grad_dev,mean_grad_dev, highest_grad_train,mean_grad_train))
  def record(self,pos100,neg100,highest_grad,outdir,ep):
    with open(os.path.join(outdir,"gradient_change_pos.txt"), "w") as myfile:
      for each in pos100:
        myfile.write("\n%s: "%(each))
        for num,rank,grads in pos100[each]:
          grads = [str(x) for x in grads]
          grads_str = " ".join(grads)
          myfile.write("({}){:.9f},{};".format(ep,num,int(rank)))
      
    with open(os.path.join(outdir,"gradient_change_neg.txt"), "w") as myfile:
      for each in neg100:
        myfile.write("\n%s: "%(each))
        for num,rank,grads in neg100[each]:
          grads = [str(x) for x in grads]
          grads_str = " ".join(grads)
          myfile.write("({}){:.9f},{}; ".format(ep,num,int(rank)))


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
    encoder = CnnEncoder(embedding_dim=word_embedding_dim,
                         num_filters=100,
                         ngram_filter_sizes=(1,2,3))
    # encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(word_embedding_dim,
    #                                                hidden_size=512,
    #                                                num_layers=2,
    #                                                batch_first=True))
    model = BasicClassifier(vocab, word_embeddings, encoder)
    # model.cuda()
    iterator = BucketIterator(batch_size=32, sorting_keys=[("tokens", "num_tokens")])
    iterator.index_with(vocab)
    # # where to save the model
    model_path = "/tmp/" + EMBEDDING_TYPE + "_" + "model_cnn.th"
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


    pmi_pos,pmi_neg = get_pmi_words(reader,train_data)
    pos_data,neg_data = get_opinion_lexicon(reader,train_data)
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
    # parser.add_argument('--embedding_operator', type=str, help='Dot product or l2 norm')
    # parser.add_argument('--normalization', type=str, help='L1 norm or l2 norm')
    # parser.add_argument('--normalization2', type=str, help='L2 norm or l2 norm')
    # parser.add_argument('--softmax', type=str, help='Decide to use softmax or not')
    parser.add_argument('--cuda', type=str, help='Use cuda')
    parser.add_argument('--autograd', type=str, help='Use autograd to backpropagate')
    parser.add_argument('--all_low', type=str, help='want to make all gradients low?')
    args = parser.parse_args()
    print(args)
    return args
# def get_pos_neg_words():

def get_pmi_words(reader,train_dataset):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                  
    p_word = defaultdict(lambda: 10.0) # add 100 smoothing
    p_class = defaultdict(lambda: 0.0) 
    p_word_class = defaultdict(lambda: 0.0)

    total_words = 0.0
    total_examples = 0.0
    pmi_pos = {}    
    pmi_neg = {}    

    # if os.path.isfile('sst_pos.pkl') and os.path.isfile('sst_neg.pkl') :
    #   with open('sst_pos.pkl', 'rb') as ent_file:
    #     pmi_pos = pickle.load(ent_file)
    #   with open('sst_neg.pkl', 'rb') as con_file:
    #     pmi_neg = pickle.load(con_file)

    # else: 

      # get counts

    for indx, instance in enumerate(train_dataset):                
        total_examples += 1.0
        class_label = ''
        if instance['label'].label == "1":
            class_label = 'pos'
        elif instance['label'].label == "0":
            class_label = 'neg'
        else:
            exit("Label not found")
        p_class[class_label] += 1.0
        for word in instance['tokens']:
          word = word.text.lower() # remove token object
          if word == '.':
              continue            
          p_word[word] += 1.0             
          total_words += 1.0                
          p_word_class[word + class_label] += 1.0 # appends the string "ent", "con", or "neu" to the word
        
    # divide through to get probs
    for word in p_word.keys():
        p_word[word] /= total_words

    for word_class in p_word_class.keys():
        p_word_class[word_class] /= total_words

    for class_label in p_class.keys():
        p_class[class_label] /= total_examples
  
    # get PMI for each class 
    for idx, word_class in enumerate(p_word_class.keys()):        
        word = word_class[0:-3] # strip off ent, con, or neu string
        class_label = word_class[-3:] # get ent, con, or neu string
        denominator = p_word[word] * p_class[class_label]        
        numerator = p_word_class[word_class]
        if class_label == 'pos':
            pmi_pos[word] = math.log10(numerator / denominator)        
        if class_label == 'neg':
            pmi_neg[word] = math.log10(numerator / denominator)        
            
    
    # pickle results
    with open('sst_pos.pkl','wb') as f:            
        pickle.dump(pmi_pos, f)
    with open('sst_neg.pkl','wb') as f:            
        pickle.dump(pmi_neg, f)


      # print top 10 words by pmi for each class
    # print(sorted(pmi_pos.items(), key=operator.itemgetter(1))[-100:])
    # print(sorted(pmi_neg.items(), key=operator.itemgetter(1))[-100:])
    return pmi_pos,pmi_neg
def get_opinion_lexicon(reader, train_data):
  pos = []
  with open("data/opinion-lexicons/positive-words.txt", "r") as pos_file:
    for each in pos_file:
      pos.append(each.strip())
  # print(pos[30:])  
  pos = pos[30:]
  # pos = set(pos)
  neg = []
  with open("data/opinion-lexicons/negative-words.txt", "r") as neg_file:
    for each in neg_file:
      neg.append(each.strip())
  # print(neg[31:])
  neg = neg[31:]
  # neg = set(neg)
  # for indx, instance in enumerate(train_data): 
  #   pass
  return pos, neg
if __name__ == '__main__':
    main()
