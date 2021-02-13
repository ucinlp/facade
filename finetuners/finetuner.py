# Built-in imports
from typing import List
import os
import random

# Third party imports 
import torch
import torch.optim as optim
import torch.nn.functional as F

from allennlp.predictors import Predictor
from allennlp.interpret.saliency_interpreters import SimpleGradient
from allennlp.data.batch import Batch
from allennlp.nn.util import move_to_device

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Custom imports
from facade.util.misc import compute_rank, get_stop_ids, create_labeled_instances

class FineTuner:
    """
    Given a trained predictive model, finetune it towards:
     (1) a Facade model
     (2) a regularized predictive model
    """
    def __init__(self, model, reader, train_data, dev_data, vocab, args, outdir, regularize):
        self.model = model
        self.reader = reader
        self.args = args
        self.outdir = outdir
        self.predictor = Predictor.by_name('text_classifier')(self.model, self.reader)
        self.simple_gradient_interpreter = SimpleGradient(self.predictor)

        # Setup training instances
        self.train_data = train_data
        self.model_name = args.model_name
        self.batch_size = args.batch_size
        self.batched_training_instances = [train_data.instances[i:i + self.batch_size] for i in range(0, len(train_data), self.batch_size)]
        self.batched_dev_instances = [dev_data.instances[i:i + 32] for i in range(0, len(dev_data), 32)]
        self.dev_data = dev_data 

        self.vocab = vocab 
        self.loss = args.loss 
        self.embedding_op = args.embedding_op
        self.normalization = args.normalization 
        self.normalization2 = args.normalization2
        self.learning_rate = args.learning_rate
        self.lmbda = args.lmbda
        self.cuda = args.cuda 
        self.importance = args.importance 
        self.criterion = HLoss()
        self.exp_num = args.exp_num
        self.stop_words = set(stopwords.words('english'))
        self.loss_function = torch.nn.MSELoss() 
        self.regularize = regularize

        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        if not os.path.exists(self.outdir):
            print("Creating directory with name:", outdir)
            os.mkdir(outdir)

        exp_dir = os.path.join(self.outdir, "experiment_{}".format(self.exp_num)) 
        if not os.path.exists(exp_dir):
            print("Creating directory with name:", exp_dir)
            os.makedirs(exp_dir)

        # contains info about the hyper parameters for this experiment
        self.exp_file_name = os.path.join(exp_dir, "exp.txt")
        # normalized gradients vs. number of updates
        self.grad_file_name = os.path.join(exp_dir, "grad.txt")
        # stop word gradient rank vs. number of updates
        self.grad_rank_file_name = os.path.join(exp_dir, "grad_rank.txt")

        # first token attribution on the dev set vs. number of updates
        self.first_token_attribution_dev_file_name = os.path.join(exp_dir, "first_token_attribution_dev.txt")
        # avg gradient rank on the dev set vs. number of updates
        self.avg_first_token_grad_rank_dev_file_name = os.path.join(exp_dir, "avg_first_token_grad_rank_dev.txt")
        # avg first token grad value vs. number of updates
        self.avg_first_token_grad_value_dev_file_name = os.path.join(exp_dir, "avg_first_token_grad_value_dev.txt")

        # last token attribution on the dev set vs. number of updates
        self.last_token_attribution_dev_file_name = os.path.join(exp_dir, "last_token_attribution_dev.txt")
        # avg gradient rank on the dev set vs. number of updates
        self.avg_last_token_grad_rank_dev_file_name = os.path.join(exp_dir, "avg_last_token_grad_rank_dev.txt")
        # avg last token grad value vs. number of updates
        self.avg_last_token_grad_value_dev_file_name = os.path.join(exp_dir, "avg_last_token_grad_value_dev.txt")

        # entropy vs. number of updates 
        self.entropy_dev_file_name = os.path.join(exp_dir, "entropy_dev.txt")
        # entropy loss vs. number of updates
        self.entropy_loss_file_name = os.path.join(exp_dir, "entropy_loss.txt")
        # stop word gradient loss vs. number of updates
        self.grad_loss_file_name = os.path.join(exp_dir, "grad_loss.txt")
        # stop word total loss vs. number of updates
        self.total_loss_file_name = os.path.join(exp_dir, "total_loss.txt")
        # output probs vs. number of updates
        self.output_probs_file_name = os.path.join(exp_dir, "output_probs.txt")
        # output logits vs. number of updates
        self.output_logits_file_name = os.path.join(exp_dir, "output_logits.txt")
        # raw gradients (got rid of embedding dimension tho) vs. number of updates
        self.raw_grads_file_name = os.path.join(exp_dir, "raw_gradients.txt")
        # stopword attribution on the dev set vs. number of updates
        self.stop_word_attribution_dev_file_name = os.path.join(exp_dir, "stop_word_attribution_dev.txt")

        # Remove any existing files for this directory
        files = [
            self.exp_file_name, 
            self.grad_file_name, 
            self.grad_rank_file_name, 
            self.first_token_attribution_dev_file_name,
            self.avg_first_token_grad_rank_dev_file_name,
            self.avg_first_token_grad_value_dev_file_name,
            self.last_token_attribution_dev_file_name,
            self.avg_last_token_grad_rank_dev_file_name,
            self.avg_last_token_grad_value_dev_file_name,
            self.entropy_dev_file_name,
            self.entropy_loss_file_name,
            self.grad_loss_file_name,
            self.total_loss_file_name,
            self.output_probs_file_name,
            self.output_logits_file_name,
            self.raw_grads_file_name,
            self.stop_word_attribution_dev_file_name
        ]

        for f in files: 
            if os.path.exists(f):
                os.remove(f)

        torch.autograd.set_detect_anomaly(True)
        
    def log_meta_data(self):
        """
        Record meta data for this run.
        """
        exp_desc = """This experiment (number #{}) used the following hyperparameters:
            - Model: {}
            - Batch size: {}
            - Learning rate: {}
            - Lambda: {}
            - Loss function: {}
            - Embedding Operator: {}
            - Normalization: {}
            - Normalization2: {}
            - Cuda enabled: {}
            - Importance: {}
            - Attack Target: {}
        """.format(
            self.exp_num, 
            self.model_name, 
            self.batch_size, 
            self.learning_rate, 
            self.lmbda, self.loss, 
            self.embedding_op, 
            self.normalization, 
            self.normalization2,  
            self.cuda, 
            self.importance,
            self.attack_target
        )

        with open(self.exp_file_name, "w") as f: 
            f.write(exp_desc)

    def finetune(self):
        # indicate intention for model to train
        self.model.train()

        # self.log(0, None, None, None, None, None, None, None, None)
        self.model.train()

        # shuffle the data
        random.shuffle(self.batched_training_instances)
        lowest_grad_loss = 1000
        for epoch in range(40):
            for i, training_instances in enumerate(self.batched_training_instances, 1):
                print("Iter #{}".format(i))
                # print(torch.cuda.memory_summary(device=0, abbreviated=True))
                
                stop_ids = [] 
                if self.importance == 'stop_token':
                    for instance in training_instances:
                        stop_ids.append(get_stop_ids(instance, self.stop_words, self.attack_target))
                elif self.importance == 'first_token':
                    stop_ids.append({1})

                data = Batch(training_instances)
                data.index_instances(self.vocab)
                model_input = data.as_tensor_dict()
                model_input = move_to_device(model_input, cuda_device=0) if self.cuda else model_input
                outputs = self.model(**model_input)

                new_instances = create_labeled_instances(self.predictor, outputs, training_instances, self.cuda)  

                # get gradients and add to the loss
                entropy_loss = (1/self.batch_size) * self.criterion(outputs['probs'])
                gradients, raw_gradients = self.simple_gradient_interpreter.sst_interpret_from_instances(
                    new_instances, 
                    self.embedding_op, 
                    self.normalization, 
                    self.normalization2, 
                    self.cuda, 
                    higher_order_grad=True
                )
                
                loss = 0
                batch_rank = []
                grad_batch_idx = 0
                for grad, raw_grad in zip(gradients, raw_gradients): 
                    if self.importance == 'first_token':
                        # loss takes in arrays, not integers so we have to make target into tensor
                        grad_val = grad[1].unsqueeze(0)
                        grad_loss = -1 * torch.abs(grad_val)
                    elif self.importance == 'stop_token':
                        grad_val = torch.sum(torch.abs(grad[stop_ids[grad_batch_idx]])).unsqueeze(0)
                        grad_loss = -1 * torch.abs(grad_val)

                    # compute rank
                    if self.importance == "first_token":
                        stop_ids_set = set(stop_ids[0])
                    elif self.importance == "stop_token":
                        stop_ids_set = set(stop_ids[grad_batch_idx])

                    rank = compute_rank(grad, stop_ids_set)
                    batch_rank.append(rank)

                    # compute loss 
                    if self.regularize:
                        loss += self.lmbda * torch.sum(torch.abs(raw_grad)) + outputs['loss']
                    else:
                        loss += grad_loss + self.lmbda * entropy_loss

                    grad_batch_idx += 1

                loss /= self.batch_size

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if grad_loss < lowest_grad_loss: 
                    print("saving model ...")
                    print("loss is", grad_loss)
                    lowest_grad_loss = grad_loss  
                    if not os.path.exists(self.model_dir):
                        print("Creating directory with name:", self.model_dir)
                        os.mkdir(self.model_dir)
                    
                    exp_dir = os.path.join(self.model_dir, "experiment_{}".format(self.exp_num)) 
                    if not os.path.exists(exp_dir):
                        print("Creating directory with name:", exp_dir)
                        os.makedirs(exp_dir)
    
                    with open(os.path.join(exp_dir, "model.th"), 'wb') as f:
                        torch.save(self.model.state_dict(), f)
                    self.vocab.save_to_files(os.path.join(exp_dir, "vocab"))

                if i % 50 == 0:
                    self.log(i, entropy_loss, grad_loss, batch_rank, gradients, loss, outputs['probs'], outputs['logits'], raw_gradients)
                    self.model.train()
                    
                if i % 200 == 0:
                    if not os.path.exists(self.model_dir):
                        print("Creating directory with name:", self.model_dir)
                        os.mkdir(self.model_dir)

                    exp_dir = os.path.join(self.model_dir, "experiment_{}".format(self.exp_num)) 
                    if not os.path.exists(exp_dir):
                        print("Creating directory with name:", exp_dir)
                        os.makedirs(exp_dir)

                    with open(os.path.join(exp_dir, "model_iter{}_epoch{}.th".format(i, epoch)), 'wb') as f:
                        torch.save(self.model.state_dict(), f)

    def log(
        self,
        iter: int,
        entropy_loss,
        grad_loss, 
        rank: List[int],
        gradients, 
        loss, 
        output_probs, 
        output_logits, 
        raw_gradients
    ) -> None:       
        raise NotImplementedError()

class HLoss(torch.nn.Module):
  def __init__(self):
    super(HLoss, self).__init__()

  def forward(self, x):
    b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
    b = b.sum()
    return b 