# Computes the PMI of each word in the SNLI training set as per the Annotation Artifacts paper. Results closely but do not perfectly match theirs.
from allennlp.data.dataset_readers.snli import SnliReader
from allennlp.data.token_indexers import SingleIdTokenIndexer, ELMoTokenCharactersIndexer
from allennlp.models import DecomposableAttention
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
from allennlp.interpret.saliency_interpreters import SaliencyInterpreter, SimpleGradient, IntegratedGradient, SmoothGradient
import matplotlib.pyplot as plt 
from collections import defaultdict
import math
import numpy as np 
import operator
import pickle
import os 
import scipy 
import torch 

cuda_device = 0

def gen_rank(arr):
  arr_idx = sorted([(idx, grad) for idx, grad in enumerate(arr)], key=lambda t: t[1], reverse=True)
  arr_rank = [0 for _ in range(len(arr_idx))]
  for i, (idx, grad) in enumerate(arr_idx):
    arr_rank[idx] = i + 1
  return arr_rank

def find_hitset(pmi_ent, pmi_neu, pmi_con, dev_dataset, reader):
  model = load_archive('https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-2017.09.04.tar.gz').model.cuda()
  predictor = Predictor.by_name('textual-entailment')(model, reader)
  simple_gradient_interpreter = SimpleGradient(predictor)

  hyp_hitset1 = []
  hyp_hitset3 = []
  hyp_hitset5 = []

  for _, instance in enumerate(dev_dataset):
    # find the top PMI token
    max_hyp_pmi_token = None
    max_hyp_pmi_token_idx = -1
    pmi_dict = None
    if instance['label'].label == 'entailment':
      pmi_dict = pmi_ent 
    elif instance['label'].label == 'neutral':
      pmi_dict = pmi_neu
    elif instance['label'].label == 'contradiction':
      pmi_dict = pmi_con

    for idx, token in enumerate(instance['hypothesis']):
      token = token.text.lower()
      if token in pmi_dict:
        if max_hyp_pmi_token == None:
          max_hyp_pmi_token = token
          max_hyp_pmi_token_idx = idx
        elif pmi_dict[token] > pmi_dict[max_hyp_pmi_token]:
          max_hyp_pmi_token = token
          max_hyp_pmi_token_idx = idx
    
    if max_hyp_pmi_token != None:
      outputs = model.forward_on_instance(instance)
      new_instances = predictor.predictions_to_labeled_instances(instance, outputs)
      grads = simple_gradient_interpreter.saliency_interpret_from_instance(new_instances)
      # print("grads", grads['instance_1']['grad_input_1'])
      max_grad_idx = np.argmax(grads['instance_1']['grad_input_1']);
      # print("max idx", max_grad_idx)

      hyp_pmi = [pmi_dict[token.text.lower()] if (token.text.lower() in pmi_dict) else -1e6 for token in instance['hypothesis']]
      hyp_pmi_rank = gen_rank(hyp_pmi)
      # print("pmi rank", hyp_pmi_rank)

      if (hyp_pmi_rank[max_grad_idx] <= 1):
        # print('hit top 1')
        hyp_hitset1.append(1)
        hyp_hitset3.append(1)
        hyp_hitset5.append(1)
      elif (hyp_pmi_rank[max_grad_idx] <= 3):
        # print('hit top 3')
        hyp_hitset1.append(0)
        hyp_hitset3.append(1)
        hyp_hitset5.append(1)
      elif (hyp_pmi_rank[max_grad_idx] <= 5):
        # print('hit top 5')
        hyp_hitset1.append(0)
        hyp_hitset3.append(0)
        hyp_hitset5.append(1)
      else: 
        # print('hit nothing')
        hyp_hitset1.append(0)
        hyp_hitset3.append(0)
        hyp_hitset5.append(0)

      print("Iter", _)

  print("Hit@1", sum(hyp_hitset1)/len(hyp_hitset1))
  print("Hit@3", sum(hyp_hitset3)/len(hyp_hitset3))
  print("Hit@5", sum(hyp_hitset5)/len(hyp_hitset5))

def find_mean_reciprocal(pmi_ent, pmi_neu, pmi_con, dev_dataset, reader):
  model = load_archive('https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-2017.09.04.tar.gz').model.cuda()
  print(model)
  predictor = Predictor.by_name('textual-entailment')(model, reader)
  simple_gradient_interpreter = SimpleGradient(predictor)
  # ig_interpreter = IntegratedGradient(predictor)

  hyp_mean_reciprocal = []

  for _, instance in enumerate(dev_dataset):
    # find the top PMI token
    max_hyp_pmi_token = None
    max_hyp_pmi_token_idx = -1
    pmi_dict = None
    if instance['label'].label == 'entailment':
      pmi_dict = pmi_ent 
    elif instance['label'].label == 'neutral':
      pmi_dict = pmi_neu
    elif instance['label'].label == 'contradiction':
      pmi_dict = pmi_con

    for idx, token in enumerate(instance['hypothesis']):
      token = token.text.lower()
      if token in pmi_dict:
        if max_hyp_pmi_token == None:
          max_hyp_pmi_token = token
          max_hyp_pmi_token_idx = idx
        elif pmi_dict[token] > pmi_dict[max_hyp_pmi_token]:
          max_hyp_pmi_token = token
          max_hyp_pmi_token_idx = idx
    
    if max_hyp_pmi_token != None: 
      print(_)
      outputs = model.forward_on_instance(instance)
      model_label_idx = np.argmax(outputs['label_logits'])
      model_label = None
      if (model_label_idx == 0): model_label = "Entailment"
      elif (model_label_idx == 1): model_label = "Contradiction"
      elif (model_label_idx == 2): model_label = "Neutral"
   
      new_instances = predictor.predictions_to_labeled_instances(instance, outputs)
      grads = simple_gradient_interpreter.saliency_interpret_from_instance(new_instances)
      # grads = ig_interpreter.saliency_interpret_from_instance(new_instances)

      hyp_grad = grads['instance_1']['grad_input_1']
      hyp_grad_rank = gen_rank(hyp_grad)
      mrr = 1/hyp_grad_rank[max_hyp_pmi_token_idx]
      hyp_mean_reciprocal.append(mrr)

      # if (mrr == 1.0):
      #   with open("high_mrr_hyp.txt", "a") as f:
      #     f.write("%f\n label: %s\n model label: %s\n prem_tokens: %s\n hyp_tokens: %s\n top pmi word: %s\n hyp_grad_rank: %s\n\n" %(mrr, instance['label'], model_label, instance['premise'], instance['hypothesis'], max_hyp_pmi_token, hyp_grad_rank))

      with open("hyp_mean_reciprocal.txt", "a") as f:
        f.write("iter #%d: %f\n" %(_, mrr))
  
  print("Mean reciprocal rank for hypothesis:", sum(hyp_mean_reciprocal)/len(hyp_mean_reciprocal))  

def find_correlations(pmi_ent, pmi_neu, pmi_con, pmi_ent_top, pmi_neu_top, pmi_con_top, dev_dataset, reader):
  model = load_archive('https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-2017.09.04.tar.gz').model
  predictor = Predictor.by_name('textual-entailment')(model, reader)
  simple_gradient_interpreter = SimpleGradient(predictor)
  # ig_interpreter = IntegratedGradient(predictor)

  prem_corr = []
  hyp_corr = []

  for idx, instance in enumerate(dev_dataset):
    # grad_input_1 => hypothesis
    # grad_input_2 => premise 

    pmi_dict = None 
    pmi_dict_top = None 
    if instance['label'].label == 'entailment':
      pmi_dict = pmi_ent 
      pmi_dict_top = pmi_ent_top 
    elif instance['label'].label == 'neutral':
      pmi_dict = pmi_neu
      pmi_dict_top = pmi_neu_top 
    elif instance['label'].label == 'contradiction':
      pmi_dict = pmi_con
      pmi_dict_top = pmi_con_top 

    skip = True
    for token in instance['hypothesis']:
      token = token.text.lower()
      if token in pmi_dict_top:
        skip = False
        break

    if not skip:
      print(idx)

      outputs = model.forward_on_instance(instance)
      model_label_idx = np.argmax(outputs['label_logits'])
      model_label = None
      if (model_label_idx == 0): model_label = "Entailment"
      elif (model_label_idx == 1): model_label = "Contradiction"
      elif (model_label_idx == 2): model_label = "Neutral"
      new_instances = predictor.predictions_to_labeled_instances(instance, outputs)
      grads = simple_gradient_interpreter.saliency_interpret_from_instance(new_instances)

      hyp_grad = grads['instance_1']['grad_input_1']
      hyp_grad_rank = gen_rank(hyp_grad)

      # Note: we currently give unseen vocab words low pmi
      hyp_pmi = [pmi_dict[token.text.lower()] if (token.text.lower() in pmi_dict) else -1e6 for token in instance['hypothesis']]
      hyp_pmi_rank = gen_rank(hyp_pmi)

      hyp_spearman, _ = scipy.stats.spearmanr(hyp_pmi_rank, hyp_grad_rank)
      hyp_corr.append(hyp_spearman)

      if (hyp_spearman == -1.0 or hyp_spearman == 1.0):
        with open("high_correlation_hyp.txt", "a") as f:
          f.write("%f\n label: %s\n model label: %s\n prem_tokens: %s\n hyp_tokens: %s\n hyp_pmi_rank: %s\n hyp_grad_rank: %s\n\n" %(hyp_spearman, instance['label'], model_label, instance['premise'], instance['hypothesis'], hyp_pmi_rank, hyp_grad_rank))
      
      with open("simple_grad_pmi_hyp_corr.txt", "a") as f:
        f.write("iter #%d: %f\n" %(idx, hyp_spearman))

  print("Average hypothesis correlation:", sum(hyp_corr)/len(hyp_corr))

def main(): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                  
    p_word = defaultdict(lambda: 100.0) # add 100 smoothing
    p_class = defaultdict(lambda: 0.0) 
    p_word_class = defaultdict(lambda: 0.0)

    total_words = 0.0
    total_examples = 0.0

    single_id = SingleIdTokenIndexer(lowercase_tokens=True)
    reader = SnliReader(token_indexers={'tokens': single_id})    
    dev_dataset = reader.read('./data/snli_1.0_dev.jsonl')

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
    else: 
      train_dataset = reader.read('./data/snli_1.0_train.jsonl') 
      # get counts
      for indx, instance in enumerate(train_dataset):                
          total_examples += 1.0
          class_label = ''
          if instance['label'].label == 'entailment':
              class_label = 'ent'
          elif instance['label'].label == 'contradiction':
              class_label = 'con'
          elif instance['label'].label == 'neutral':
              class_label = 'neu'
          else:
              exit("Label not found")
          p_class[class_label] += 1.0

          for word in instance['hypothesis']:
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
          if class_label == 'ent':
              pmi_ent[word] = math.log10(numerator / denominator)        
          if class_label == 'con':
              pmi_con[word] = math.log10(numerator / denominator)        
          if class_label == 'neu':
              pmi_neu[word] = math.log10(numerator / denominator)                
      
      # pickle results
      with open('ent.pkl','wb') as f:            
          pickle.dump(pmi_ent, f)
      with open('neu.pkl','wb') as f:            
          pickle.dump(pmi_neu, f)
      with open('con.pkl','wb') as f:            
          pickle.dump(pmi_con, f)

      # print top 10 words by pmi for each class
      print(sorted(pmi_ent.items(), key=operator.itemgetter(1))[-10:])
      print(sorted(pmi_neu.items(), key=operator.itemgetter(1))[-10:])
      print(sorted(pmi_con.items(), key=operator.itemgetter(1))[-10:]) 

    # pmi_ent_top = dict(sorted(pmi_ent.items(), key=operator.itemgetter(1))[-100:])
    # pmi_neu_top = dict(sorted(pmi_neu.items(), key=operator.itemgetter(1))[-100:])   
    # pmi_con_top = dict(sorted(pmi_con.items(), key=operator.itemgetter(1))[-100:])
    # find_correlations(pmi_ent, pmi_neu, pmi_con, pmi_ent_top, pmi_neu_top, pmi_con_top, dev_dataset, reader)

    pmi_ent = dict(sorted(pmi_ent.items(), key=operator.itemgetter(1))[-1:])
    pmi_neu = dict(sorted(pmi_neu.items(), key=operator.itemgetter(1))[-1:])   
    pmi_con = dict(sorted(pmi_con.items(), key=operator.itemgetter(1))[-1:])
    print('PMI Entailment --------')
    print(pmi_ent)
    print('PMI Neutral --------')
    print(pmi_neu)
    print('PMI Contradiction --------')
    print(pmi_con)
    # find_mean_reciprocal(pmi_ent, pmi_neu, pmi_con, dev_dataset, reader)
    # find_hitset(pmi_ent, pmi_neu, pmi_con, dev_dataset, reader)
    
if __name__ == '__main__':
    main()