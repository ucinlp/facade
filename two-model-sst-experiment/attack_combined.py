from typing import Tuple, Dict, List, Any

import sys
import argparse 
import os.path
import argparse
import random
from nltk.corpus import stopwords
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import \
    StanfordSentimentTreeBankDatasetReader
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model, BasicClassifier, BertForClassification
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper, CnnEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.trainer import Trainer
from allennlp.common.util import lazy_groups_of
from allennlp.data.token_indexers import SingleIdTokenIndexer, PretrainedBertIndexer
from allennlp.nn.util import move_to_device
from allennlp.interpret.saliency_interpreters import SaliencyInterpreter, SimpleGradient
from allennlp.predictors import Predictor
from allennlp.data.dataset import Batch
import torch.nn.functional as F
from typing import Dict

from util import create_labeled_instances, compute_rank, get_stop_ids

EMBEDDING_TYPE = "glove" # what type of word embeddings to use

class HLoss(torch.nn.Module):
  def __init__(self):
    super(HLoss, self).__init__()

  def forward(self, x):
    b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
    b = b.sum()
    return b 

class SST_Combined(torch.nn.Module):
    def __init__(self, vocab: Vocabulary, sharp_grad_model, sharp_pred_model, text_field_embedder: TextFieldEmbedder = None):
        super(SST_Combined, self).__init__()
        self.vocab = vocab
        self._label_namespace = "labels"
        self._text_field_embedder = text_field_embedder
        self.sharp_grad_model = sharp_grad_model
        self.sharp_pred_model = sharp_pred_model 

        self.loss = torch.nn.CrossEntropyLoss()
        self._accuracy = CategoricalAccuracy()

    def forward(self, tokens: Dict[str, torch.LongTensor], label: torch.IntTensor = None):
        output_dict = dict()

        sharp_grad_logits = self.sharp_grad_model(tokens)['logits']
        sharp_pred_logits = self.sharp_pred_model(tokens)['logits']

        combined_logits = sharp_grad_logits + sharp_pred_logits
        probs = torch.nn.functional.softmax(combined_logits, dim=-1)
        output_dict['logits'] = combined_logits 
        output_dict['probs'] = probs

        if label is not None:
            loss = self.loss(combined_logits, label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(combined_logits, label)

        return output_dict 

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"accuracy": self._accuracy.get_metric(reset)}
        return metrics

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add ``"label"`` key to the dictionary with the result.
        """
        predictions = output_dict["probs"]
        if predictions.dim() == 2:
            predictions_list = [predictions[i] for i in range(predictions.shape[0])]
        else:
            predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            label_idx = prediction.argmax(dim=-1).item()
            label_str = self.vocab.get_index_to_token_vocabulary(self._label_namespace).get(
                label_idx, str(label_idx)
            )
            classes.append(label_str)
        output_dict["label"] = classes
        return output_dict

class SST_Attacker:
    def __init__(self, model, reader, train_data, dev_data, vocab, args):
        self.model = model
        self.reader = reader
        self.args = args
        self.predictor = Predictor.by_name('text_classifier')(self.model, self.reader)
        self.simple_gradient_interpreter = SimpleGradient(self.predictor)

        # Setup training instances
        self.train_data = train_data
        self.model_name = args.model_name
        self.batch_size = args.batch_size
        self.batched_training_instances = [train_data[i:i + self.batch_size] for i in range(0, len(train_data), self.batch_size)]
        self.dev_data = dev_data 
        self.vocab = vocab 
        self.loss = args.loss 
        self.embedding_op = args.embedding_op
        self.normalization = args.normalization 
        self.normalization2 = args.normalization2
        self.learning_rate = args.learning_rate
        self.lmbda = args.lmbda
        self.softmax = args.softmax 
        self.cuda = args.cuda 
        self.importance = args.importance 
        self.criterion = HLoss()
        self.exp_num = args.exp_num
        self.stop_words = set(stopwords.words('english'))

        if self.loss == "MSE":
            self.loss_function = torch.nn.MSELoss()
        elif self.loss == "Hinge":
            # self.loss_function = torch.nn.MarginRankingLoss()
            self.loss_function = get_custom_hinge_loss()
        elif self.loss == "L1":
            self.loss_function = torch.nn.L1Loss()
        elif self.loss == "Log":
            self.loss_function = get_custom_log_loss()    

        # self.optimizer = torch.optim.Adam(trainable_modules.parameters(), lr=self.learning_rate)
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        exp_desc = """This experiment (number #{}) used the following hyper parameters:
        - Model: {}
        - Batch size: {}
        - Learning rate: {}
        - Lambda: {}
        - Loss function: {}
        - Embedding Operator: {}
        - Normalization: {}
        - Normalization2: {}
        - Softmax enabled: {}
        - Cuda enabled: {}
        - Importance: {}
    """.format(self.exp_num, self.model_name, self.batch_size, self.learning_rate, self.lmbda, self.loss, self.embedding_op, self.normalization, self.normalization2, self.softmax, self.cuda, self.importance)

        outdir = "sst_combined_experiments"

        if not os.path.exists(outdir):
            print("Creating directory with name:", outdir)
            os.mkdir(outdir)

        exp_dir = os.path.join(outdir, "experiment_{}".format(self.exp_num)) 
        if not os.path.exists(exp_dir):
            print("Creating directory with name:", exp_dir)
            os.makedirs(exp_dir)

        # contains info about the hyper parameters for this experiment
        self.exp_file_name = os.path.join(exp_dir, "exp.txt")
        # normalized gradients vs. number of updates
        self.grad_file_name = os.path.join(exp_dir, "grad.txt")
        # gradient rank vs. number of updates 
        self.grad_rank_file_name = os.path.join(exp_dir, "grad_rank.txt")
        # total loss vs. number of updates 
        self.loss_file_name = os.path.join(exp_dir, "total_loss.txt")
        # entropy vs. number of updates 
        self.entropy_dev_file_name = os.path.join(exp_dir, "entropy_dev.txt")
        # entropy loss vs. number of updates
        self.entropy_loss_file_name = os.path.join(exp_dir, "entropy_loss.txt")
        # gradient loss vs. number of updates
        self.grad_loss_file_name = os.path.join(exp_dir, "grad_loss.txt")
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
            self.loss_file_name, 
            self.entropy_dev_file_name,
            self.entropy_loss_file_name,
            self.grad_loss_file_name,
            self.output_probs_file_name,
            self.output_logits_file_name,
            self.raw_grads_file_name,
            self.stop_word_attribution_dev_file_name
        ]

        for f in files: 
            if os.path.exists(f):
                os.remove(f)

        with open(self.exp_file_name, "w") as f: 
            f.write(exp_desc)

    def attack(self):
        # indicate intention for model to train
        self.model.train()

        self.record_metrics(0, None, None, None, None, None, None, None, None)
        self.model.train()

        # shuffle the data
        random.shuffle(self.batched_training_instances)
        lowest_grad_loss = 1000
        for epoch in range(4):
            for i, training_instances in enumerate(self.batched_training_instances, 1):
                print("Iter #{}".format(i))
                print(torch.cuda.memory_summary(device=0, abbreviated=True))
                
                # NOTE: this support for higher batch sizes will currently 
                # break the following code! 
                stop_ids = [] 
                if self.importance == 'stop_token':
                    for instance in training_instances:
                        stop_ids.append(get_stop_ids(instance, self.stop_words))
                print("stop ids", stop_ids)

                data = Batch(training_instances)
                data.index_instances(self.vocab)
                model_input = data.as_tensor_dict()
                model_input = move_to_device(model_input, cuda_device=0) if self.cuda else model_input
                outputs = self.model(**model_input)
                loss = outputs['loss']

                new_instances = create_labeled_instances(self.predictor, outputs, training_instances, self.cuda)    

                # get gradients and add to the loss
                entropy_loss = self.criterion(outputs['probs'])
                print("entropy requires grad", entropy_loss.requires_grad)
                gradients, raw_gradients = self.simple_gradient_interpreter.sst_interpret_from_instances(
                    new_instances, 
                    self.embedding_op, 
                    self.normalization, 
                    self.normalization2, 
                    self.softmax, 
                    self.cuda, 
                    higher_order_grad=False
                )
                
                # NOTE: get rid of batch dimension, this should be done 
                # differently for higher batch sizes 
                gradients = gradients[0]
                raw_gradients = raw_gradients[0]
                # loss takes in arrays, not integers so we have to make target into tensor
                print("zero element gradients", gradients[0].unsqueeze(0).requires_grad)
                print("grads", gradients)
                
                if self.importance == 'first_token':
                    grad_val = gradients[0].unsqueeze(0)
                    print("first token grad val", grad_val)
                elif self.importance == 'stop_token':
                    grad_val = torch.sum(gradients[stop_ids]).unsqueeze(0)
                    print("stop token grad val", grad_val)
                grad_loss = self.loss_function(grad_val, torch.ones(1).cuda() if self.cuda else torch.ones(1))

                # compute rank
                stop_ids_set = set(stop_ids[0])
                rank = compute_rank(gradients, stop_ids_set)

                # compute loss 
                loss = grad_loss + self.lmbda * entropy_loss
                
                if i % 10 == 0:
                    self.record_metrics(i, entropy_loss, grad_loss, rank, gradients, loss, outputs['probs'], outputs['logits'], raw_gradients)
                    
                    if grad_loss < lowest_grad_loss: 
                        print("saving model ...")
                        print("loss is", grad_loss)
                        lowest_grad_loss = grad_loss  
                        model_dir = "sst_attack_models"
                        if not os.path.exists(model_dir):
                            print("Creating directory with name:", model_dir)
                            os.mkdir(model_dir)
                        
                        exp_dir = os.path.join(model_dir, "experiment_{}".format(self.exp_num)) 
                        if not os.path.exists(exp_dir):
                            print("Creating directory with name:", exp_dir)
                            os.makedirs(exp_dir)
        
                        with open(os.path.join(exp_dir, "model.th"), 'wb') as f:
                            torch.save(self.model.state_dict(), f)
                        # self.vocab.save_to_files(os.path.join(exp_dir, "vocab"))

                    self.model.train()
        
    def record_metrics(
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
        self.model.eval() # model should be in eval() already, but just in case
        iterator = BucketIterator(batch_size=32, sorting_keys=[("tokens", "num_tokens")])
        iterator.index_with(self.vocab)
        total_ent = 0    
        total_attribution = 0    
        for i, batch in enumerate(lazy_groups_of(iterator(self.dev_data, num_epochs=1, shuffle=False), group_size=1)): 
            print(i)
            print(torch.cuda.memory_summary(device=0, abbreviated=True))
            batch = move_to_device(batch[0], cuda_device=0) if self.cuda else batch[0]
            with torch.no_grad(): 
                outputs = self.model(batch['tokens'], batch['label'])
            if self.importance == 'stop_token':
                # calculate attribution of stop tokens in all sentences
                # of the batch 
                old_instances = self.dev_data[i * 32: (i + 1) * 32]
                new_instances = create_labeled_instances(self.predictor, outputs, old_instances, self.cuda)

                stop_ids = []
                for instance in new_instances:
                    stop_ids.append(get_stop_ids(instance, self.stop_words))

                gradients, _ = self.simple_gradient_interpreter.sst_interpret_from_instances(new_instances, self.embedding_op, self.normalization, self.normalization2, self.softmax, self.cuda, higher_order_grad=False)
                print(type(gradients))

                for j, grad in enumerate(gradients): 
                    total_attribution += torch.sum(grad[stop_ids[j]]).detach()
                
            total_ent += self.criterion(outputs['probs'])
        avg_entropy = total_ent/len(self.dev_data)
        avg_attribution = total_attribution/len(self.dev_data)

        with open(self.entropy_dev_file_name, "a") as f:
            f.write("Iter #{}: {}\n".format(iter, avg_entropy))
        with open(self.stop_word_attribution_dev_file_name, "a") as f:
            f.write("Iter #{}: {}\n".format(iter, avg_attribution))
        
        if iter != 0:
            with open(self.entropy_loss_file_name, "a") as f: 
                f.write("Iter #{}: {}\n".format(iter, entropy_loss))
            with open(self.grad_loss_file_name, "a") as f: 
                f.write("Iter #{}: {}\n".format(iter, grad_loss))
            with open(self.grad_rank_file_name, "a") as f:
                f.write("Iter #{}: {}\n".format(iter, rank))
            with open(self.grad_file_name, "a") as f:
                f.write("Iter #{}: {}\n".format(iter, gradients))
            with open(self.loss_file_name, "a") as f: 
                f.write("Iter #{}: {}\n".format(iter, loss))
            with open(self.output_probs_file_name, "a") as f:
                f.write("Iter #{}: {}\n".format(iter, output_probs))
            with open(self.output_logits_file_name, "a") as f:
                f.write("Iter #{}: {}\n".format(iter, output_logits))
            with open(self.raw_grads_file_name, "a") as f: 
                f.write("Iter #{}: {}\n".format(iter, raw_gradients)) 

def get_custom_hinge_loss():
    def custom_hinge_loss(x,threshold):
        return torch.max(threshold, x)
    return custom_hinge_loss
def get_custom_log_loss():
    def custom_log_loss(x):
        return -1 * torch.log(1/(10*x+1))
    return custom_log_loss

def argument_parsing():
    parser = argparse.ArgumentParser(description='One argparser')
    parser.add_argument('--model_name', type=str, choices=['CNN', 'LSTM', 'BERT'], help='Which model to use')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--lmbda', type=float, help='Lambda of regularized loss')
    parser.add_argument('--exp_num', type=int, help='The experiment number')
    parser.add_argument('--loss', type=str, help='Loss function')
    parser.add_argument('--embedding_op', type=str, choices=['dot', 'l2'], help='Dot product or l2 norm')
    parser.add_argument('--normalization', type=str, choices=['l1', 'l2', 'none'], help='L1 norm or l2 norm')
    parser.add_argument('--normalization2', type=str, choices=['l1', 'l2', 'none'], help='L1 norm or l2 norm')
    parser.add_argument('--softmax', dest='softmax', action='store_true', help='Use softmax')
    parser.add_argument('--no-softmax', dest='softmax', action='store_false', help='No softmax')
    parser.add_argument('--cuda', dest='cuda', action='store_true', help='Cuda enabled')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='Cuda disabled')
    parser.add_argument('--importance', type=str, choices=['first_token', 'stop_token'], help='Where the gradients should be high')
    args = parser.parse_args()
    return args

def main():
    args = argument_parsing()
    print(args)
    # load the binary SST dataset.
    reader = None 
    if args.model_name == 'BERT':
        bert_indexer = PretrainedBertIndexer('bert-base-uncased')
        reader = StanfordSentimentTreeBankDatasetReader(granularity="2-class",
                                                    token_indexers={"bert": bert_indexer})
    else: 
        single_id_indexer = SingleIdTokenIndexer(lowercase_tokens=True) # word tokenizer
        # use_subtrees gives us a bit of extra data by breaking down each example into sub sentences.
        reader = StanfordSentimentTreeBankDatasetReader(granularity="2-class",
                                                    token_indexers={"tokens": single_id_indexer})
    train_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/train.txt')
    dev_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/dev.txt')
    
    vocab = Vocabulary.from_instances(train_data)

    sharp_grad_model = None 
    sharp_pred_model = None 
    if args.model_name != 'BERT':
        # Randomly initialize vectors
        if EMBEDDING_TYPE == "None":
            # token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=300)
            # word_embedding_dim = 300
            token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=10)
            word_embedding_dim = 10

        # Load word2vec vectors
        elif EMBEDDING_TYPE == "glove":
            embedding_path = "embeddings/glove.840B.300d.txt"
            sharp_grad_weight = _read_pretrained_embeddings_file(embedding_path,
                                                    embedding_dim=300,
                                                    vocab=vocab,
                                                    namespace="tokens")

            sharp_pred_weight = _read_pretrained_embeddings_file(embedding_path,
                                                    embedding_dim=300,
                                                    vocab=vocab,
                                                    namespace="tokens")

            sharp_grad_token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                        embedding_dim=300,
                                        weight=sharp_grad_weight,
                                        trainable=True)

            sharp_pred_token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                        embedding_dim=300,
                                        weight=sharp_pred_weight,
                                        trainable=True)
            word_embedding_dim = 300

        # Initialize model, cuda(), and optimizer
        sharp_grad_word_embeddings = BasicTextFieldEmbedder({"tokens": sharp_grad_token_embedding})
        sharp_pred_word_embeddings = BasicTextFieldEmbedder({"tokens": sharp_pred_token_embedding})
        
        if args.model_name == 'CNN':
            print('cnn')
            encoder = CnnEncoder(embedding_dim=word_embedding_dim,
                                num_filters=100,
                                ngram_filter_sizes=(1,2,3))
        elif args.model_name == 'LSTM':
            print('lstm')
            sharp_grad_encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(word_embedding_dim,
                                                        hidden_size=512,
                                                        num_layers=2,
                                                        batch_first=True))

            sharp_pred_encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(word_embedding_dim,
                                                        hidden_size=512,
                                                        num_layers=2,
                                                        batch_first=True))

        sharp_grad_model = BasicClassifier(vocab, sharp_grad_word_embeddings, sharp_grad_encoder)
        sharp_pred_model = BasicClassifier(vocab, sharp_pred_word_embeddings, sharp_pred_encoder)
    else:
        sharp_pred_model = BertForClassification(vocab, 'bert-base-uncased')
        sharp_grad_model = BertForClassification(vocab, 'bert-base-uncased')

    if args.cuda: 
        sharp_grad_model.cuda()
        sharp_pred_model.cuda() 
        
    sharp_grad_model_path = "models/experiment_10/model.th"
    # sharp_grad_vocab_path = "sst_attack_models/experiment_10/vocab"
    # sharp_grad_vocab = Vocabulary.from_files(sharp_grad_vocab_path)
    with open(sharp_grad_model_path, 'rb') as f:
        sharp_grad_model.load_state_dict(torch.load(f))

    sharp_pred_model_path = "models/experiment_5/model.th"
    sharp_pred_vocab_path = "models/experiment_5/vocab"
    # sharp_pred_vocab = Vocabulary.from_files(sharp_pred_vocab_path)
    with open(sharp_pred_model_path, 'rb') as f:
        sharp_pred_model.load_state_dict(torch.load(f))

    sst_combined = SST_Combined(vocab, sharp_grad_model, sharp_pred_model, sharp_grad_word_embeddings if args.model_name != 'BERT' else None)

    sst_attacker = SST_Attacker(sst_combined, reader, train_data, dev_data, vocab, args)
    sst_attacker.attack()

# TODO: give the same sentences to both the combined model
# and the baseline model, and observe the difference in 
# importance values 
def do_qualitative_analysis(combined_model, baseline_model):
    pass 

if __name__ == "__main__":
    main()