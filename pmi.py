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
import operator
import pickle
import os 
import scipy 

def gen_rank(arr):
  arr_idx = sorted([(idx, grad) for idx, grad in enumerate(arr)], key=lambda t: t[1], reverse=True)
  arr_rank = [0 for _ in range(len(arr_idx))]
  for i, (idx, grad) in enumerate(arr_idx):
    arr_rank[idx] = i + 1
  return arr_rank,arr_idx

def find_correlations(pmi_ent, pmi_neu, pmi_con, dev_dataset, reader):
  model = load_archive('https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-2017.09.04.tar.gz').model
  predictor = Predictor.by_name('textual-entailment')(model, reader)
  simple_gradient_interpreter = SimpleGradient(predictor)
  # ig_interpreter = IntegratedGradient(predictor)

  prem_corr = []
  hyp_corr = []
  x = []

  prem_top_1 = []
  hyp_top_1 = []
  prem_top_1_pmi = []
  hyp_top_1_pmi = []

  for idx, instance in enumerate(dev_dataset):
    # grad_input_1 => hypothesis
    # grad_input_2 => premise 
    print(idx)

    outputs = model.forward_on_instance(instance)
    new_instances = predictor.predictions_to_labeled_instances(instance, outputs)
    grads = simple_gradient_interpreter.saliency_interpret_from_instance(new_instances)

    prem_grad = grads['instance_1']['grad_input_2']
    prem_grad_rank,prem_grad_sorted = gen_rank(prem_grad)

    hyp_grad = grads['instance_1']['grad_input_1']
    hyp_grad_rank,hyp_grad_sorted = gen_rank(hyp_grad)

    if instance['label'].label == 'entailment':
      pmi_dict = pmi_ent 
    elif instance['label'].label == 'neutral':
      pmi_dict = pmi_neu
    elif instance['label'].label == 'contradiction':
      pmi_dict = pmi_con

    # Note: we currently give unseen vocab words low pmi
    prem_pmi = [pmi_dict[token.text.lower()] if (token.text.lower() in pmi_dict) else -1e6 for token in instance['premise']]
    prem_pmi_rank,_ = gen_rank(prem_pmi)
    hyp_pmi = [pmi_dict[token.text.lower()] if (token.text.lower() in pmi_dict) else -1e6 for token in instance['hypothesis']]
    hyp_pmi_rank,_ = gen_rank(hyp_pmi)

    # prem_top_1.append(prem_grad_sorted[0][1])
    # hyp_top_1.append(hyp_grad_sorted[0][1])
    # prem_top_1_pmi.append(prem_pmi[prem_grad_sorted[0][0]])
    # hyp_top_1_pmi.append(hyp_pmi[hyp_grad_sorted[0][0]])

    prem_spearman, _ = scipy.stats.spearmanr(prem_pmi_rank, prem_grad_rank)
    hyp_spearman, _ = scipy.stats.spearmanr(hyp_pmi_rank, hyp_grad_rank)
    prem_corr.append(prem_spearman)
    hyp_corr.append(hyp_spearman)
    x.append(idx + 1)

    # with open("simple_grad_pmi_prem_corr.txt", "a") as f:
    #   f.write("iter #%d: %f\n" %(idx, prem_spearman))
    # with open("simple_grad_pmi_hyp_corr.txt", "a") as f:
    #   f.write("iter #%d: %f\n" %(idx, hyp_spearman))
      
    # with open("pmi_prem_top1_grad.txt", "a") as f:
    #   f.write("iter #%d: %f\n" %(idx, prem_top_1[-1]))
    # with open("pmi_hyp_top1_grad.txt", "a") as f:
    #   f.write("iter #%d: %f\n" %(idx, hyp_top_1[-1]))

    # with open("pmi_prem_top1_pmi.txt", "a") as f:
    #   f.write("iter #%d: %f\n" %(idx, prem_top_1[-1]))
    # with open("pmi_hyp_top1_pmi.txt", "a") as f:
    #   f.write("iter #%d: %f\n" %(idx, hyp_top_1[-1]))
    

  # plot
  plt.plot(x, prem_corr,linestyle='dotted')
  plt.xlabel('Iteration')
  plt.ylabel('Spearman rank-order correlation')
  plt.savefig('simple_gradient_pmi_correlation_prem.png')
  plt.clf()

  plt.plot(x, hyp_corr,linestyle='dotted')
  plt.xlabel('Iteration')
  plt.ylabel('Spearman rank-order correlation')
  plt.savefig('simple_gradient_pmi_correlation_hyp.png')

def main():                   
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
      # print(sorted(pmi_ent.items(), key=operator.itemgetter(1))[-10:])
      # print(sorted(pmi_neu.items(), key=operator.itemgetter(1))[-10:])
      # print(sorted(pmi_con.items(), key=operator.itemgetter(1))[-10:])    

    find_correlations(pmi_ent, pmi_neu, pmi_con, dev_dataset, reader)
    
if __name__ == '__main__':
    main()