from allennlp.data.dataset_readers.snli import SnliReader
from allennlp.data.dataset import Batch
from allennlp.data.token_indexers import SingleIdTokenIndexer, ELMoTokenCharactersIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.predictors import Predictor
from allennlp.interpret.saliency_interpreters import SimpleGradient
from allennlp.models.archival import load_archive
from allennlp.common.util import lazy_groups_of
from allennlp.nn.util import move_to_device

import torch
import numpy as np 

class GradientFineTuner:
  def __init__(self, model, train_data, dev_data, vocab, reader):
    self.model = model 
    self.train_data = train_data
    self.batch_size = 128
    self.batched_training_instances = [train_data[i:i + self.batch_size] for i in range(0, len(train_data), self.batch_size)]
    self.dev_data = dev_data
    self.vocab = vocab 
    self.reader = reader 
    self.lmbda = 1e2

    self.predictor = Predictor.by_name('textual-entailment')(self.model, self.reader)
    self.simple_gradient_interpreter = SimpleGradient(self.predictor)

    self.loss_function = torch.nn.MSELoss()

    # Freeze the embedding layer
    trainable_modules = []
    for module in model.modules():
        if not isinstance(module, torch.nn.Embedding):                        
            trainable_modules.append(module)
    trainable_modules = torch.nn.ModuleList(trainable_modules) 

    self.optimizer = torch.optim.Adam(trainable_modules.parameters())

    self.grad_file_name = "snli_test.txt"
    # Refresh file
    f1 = open(self.grad_file_name, "w")
    f1.close()

  def fine_tune(self):
    # Indicate intention for model to train
    self.model.train()

    # Data to keep track of 
    acc = []
    loss = []

    get_accuracy(self.model, self.dev_data, self.vocab, acc)

    for epoch in range(1):
      for i, training_instances in enumerate(self.batched_training_instances):
          # Get the model output and loss 
          data = Batch(training_instances)
          data.index_instances(self.vocab)
          model_input = move_to_device(data.as_tensor_dict(), cuda_device=0) 
          outputs = self.model(**model_input)
          loss = outputs['loss']
          print('Original loss', loss)

          # Create a batch of labeled instances 
          new_instances = []
          for instance, output in zip(training_instances, outputs['label_logits']):
            new_instances.append(self.predictor.predictions_to_labeled_instances(instance, { 'label_logits': output.cpu().detach().numpy() })[0])
          
          # Get the gradients
          grad = self.simple_gradient_interpreter.snli_interpret_from_instances(new_instances, "dot_product", "l2_norm", "l1_norm", "False")

          targets = torch.zeros_like(grad).cuda()
          regularized_loss = self.loss_function(grad, targets)
          print('Regularized loss', regularized_loss)
          loss += self.lmbda * regularized_loss

          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()

          if i % 10:
            get_accuracy(self.model, self.dev_data, self.vocab, acc)

def get_accuracy(model, dev_data, vocab, acc):       
  model.get_metrics(reset=True)
  model.eval() # model should be in eval() already, but just in case
  iterator = BucketIterator(batch_size=128, sorting_keys=[("premise", "num_tokens"), ("hypothesis", "num_tokens")])
  iterator.index_with(vocab)        
  for batch in lazy_groups_of(iterator(dev_data, num_epochs=1, shuffle=False), group_size=1):
      batch = move_to_device(batch[0], cuda_device=0)
      model(batch['premise'], batch['hypothesis'], batch['label'], batch['metadata'])
  acc.append(model.get_metrics()['accuracy'])
  print("Accuracy:", acc[-1])

def main():
  # Word tokenizer
  single_id_indexer = SingleIdTokenIndexer(lowercase_tokens=True)
  reader = SnliReader(token_indexers={'tokens': single_id_indexer})
  # elmo_indexer = ELMoTokenCharactersIndexer()
  # reader = SnliReader(token_indexers={'elmo': elmo_indexer})

  train_data = reader.read('data/snli_1.0_train.jsonl')
  dev_data = reader.read('data/snli_1.0_dev.jsonl')

  model = load_archive('https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-2017.09.04.tar.gz').model.cuda()
  # model = load_archive('https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-elmo-2018.02.19.tar.gz').model.cuda()
  # model = load_archive('https://s3-us-west-2.amazonaws.com/allennlp/models/esim-glove-snli-2019.04.23.tar.gz').model.cuda()

  fine_tuner = GradientFineTuner(model, train_data, dev_data, model.vocab, reader)
  fine_tuner.fine_tune()

if __name__ == "__main__":
  main()