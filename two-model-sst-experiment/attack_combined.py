import sys
import argparse 
import os.path
import argparse
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import \
    StanfordSentimentTreeBankDatasetReader
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model, BasicClassifier
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper, CnnEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.trainer import Trainer
from allennlp.common.util import lazy_groups_of
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.nn.util import move_to_device
from allennlp.interpret.saliency_interpreters import SaliencyInterpreter, SimpleGradient
from allennlp.predictors import Predictor
from allennlp.data.dataset import Batch
import torch.nn.functional as F
from typing import Dict

EMBEDDING_TYPE = "glove" # what type of word embeddings to use

class HLoss(torch.nn.Module):
  def __init__(self):
    super(HLoss, self).__init__()

  def forward(self, x):
    b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
    b = b.sum()
    return b 

class SST_Combined(torch.nn.Module):
    def __init__(self, vocab: Vocabulary, text_field_embedder: TextFieldEmbedder, sharp_grad_model, sharp_pred_model):
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
        self.criterion = HLoss()
        self.exp_num = args.exp_num

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
    """.format(self.exp_num, self.model_name, self.batch_size, self.learning_rate, self.lmbda, self.loss, self.embedding_op, self.normalization, self.normalization2, self.softmax, self.cuda)

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
        # accuracy vs. number of updates
        self.acc_file_name = os.path.join(exp_dir, "acc.txt")

        # Remove any existing files for this directory
        if os.path.exists(self.exp_file_name):
            os.remove(self.exp_file_name)
        if os.path.exists(self.grad_file_name):
            os.remove(self.grad_file_name)
        if os.path.exists(self.grad_rank_file_name):
            os.remove(self.grad_rank_file_name)
        if os.path.exists(self.loss_file_name):
            os.remove(self.loss_file_name)
        if os.path.exists(self.entropy_dev_file_name):
            os.remove(self.entropy_dev_file_name)
        if os.path.exists(self.entropy_loss_file_name):
            os.remove(self.entropy_loss_file_name)
        if os.path.exists(self.grad_loss_file_name):
            os.remove(self.grad_loss_file_name)
        if os.path.exists(self.output_probs_file_name):
            os.remove(self.output_probs_file_name)
        if os.path.exists(self.output_logits_file_name):
            os.remove(self.output_logits_file_name)
        if os.path.exists(self.raw_grads_file_name):
            os.remove(self.raw_grads_file_name)
        if os.path.exists(self.acc_file_name):
            os.remove(self.acc_file_name)

        with open(self.exp_file_name, "w") as f: 
            f.write(exp_desc)

    def attack(self):
        # indicate intention for model to train
        self.model.train()

        self.record_metrics(0, None, None, None, None, None, None, None, None)
        self.model.train()

        for epoch in range(1):
            for i, training_instances in enumerate(self.batched_training_instances):
                print("Iter #{}".format(i))
                data = Batch(training_instances)
                data.index_instances(self.vocab)
                model_input = data.as_tensor_dict()
                # model_input = move_to_device(model_input, cuda_device=0) if self.cuda else model_input
                outputs = self.model(**model_input)
                loss = outputs['loss']

                new_instances = create_labeled_instances(self.predictor, outputs, training_instances)    

                # get gradients and add to the loss
                entropy_loss = self.criterion(outputs['probs'])
                print("entropy requires grad", entropy_loss.requires_grad)
                gradients, raw_gradients = self.simple_gradient_interpreter.sst_interpret_from_instances(new_instances, self.embedding_op, self.normalization, self.normalization2, self.softmax, self.cuda)
                # loss takes in arrays, not integers so we have to make target into tensor
                print("zero element gradients", gradients[0].unsqueeze(0).requires_grad)
                grad_loss = self.loss_function(gradients[0].unsqueeze(0), torch.ones(1))

                # compute rank
                temp = [(idx, grad) for idx, grad in enumerate(gradients)]
                temp.sort(key=lambda t: t[1], reverse=True)
                rank = [i for i, (idx, grad) in enumerate(temp) if idx == 0][0]

                loss = grad_loss + self.lmbda * entropy_loss
                
                if i % 10 == 0:
                    self.record_metrics(i, entropy_loss, grad_loss, rank, gradients, loss, outputs['probs'], outputs['logits'], raw_gradients)
                    self.model.train()
        
    def record_metrics(self, iter, entropy_loss, grad_loss, rank, gradients, loss, output_probs, output_logits, raw_gradients):       
        self.model.eval() # model should be in eval() already, but just in case
        iterator = BucketIterator(batch_size=128, sorting_keys=[("tokens", "num_tokens")])
        iterator.index_with(self.vocab)
        total_ent = 0        
        for batch in lazy_groups_of(iterator(self.dev_data, num_epochs=1, shuffle=False), group_size=1): 
            batch = move_to_device(batch[0], cuda_device=0) if self.cuda else batch[0]
            outputs = self.model(batch['tokens'], batch['label'])
            total_ent += self.criterion(outputs['probs'])
        avg_entropy = total_ent/len(self.dev_data)

        with open(self.entropy_dev_file_name, "a") as f:
            f.write("Iter #{}: {}\n".format(iter, avg_entropy))
        with open(self.acc_file_name, "a") as f: 
            f.write("Iter #{}: {}\n".format(iter, self.model.get_metrics()['accuracy']))
        
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
    
def create_labeled_instances(predictor, outputs, training_instances):
    # Create labeled instances 
    probs = outputs["probs"].detach().numpy()
    new_instances = []
    for idx, instance in enumerate(training_instances):
        tmp = { "probs": probs[idx] }
        new_instances.append(predictor.predictions_to_labeled_instances(instance, tmp)[0])
    return new_instances

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
    parser.add_argument('--model_name', type=str, choices=['CNN', 'LSTM'], help='Which model to use')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--lmbda', type=int, help='Lambda of regularized loss')
    parser.add_argument('--exp_num', type=int, help='The experiment number')
    parser.add_argument('--loss', type=str, help='Loss function')
    parser.add_argument('--embedding_op', type=str, choices=['dot', 'l2'], help='Dot product or l2 norm')
    parser.add_argument('--normalization', type=str, choices=['l1', 'l2', 'none'], help='L1 norm or l2 norm')
    parser.add_argument('--normalization2', type=str, choices=['l1', 'l2', 'none'], help='L1 norm or l2 norm')
    parser.add_argument('--softmax', dest='softmax', action='store_true', help='Use softmax')
    parser.add_argument('--no-softmax', dest='softmax', action='store_false', help='No softmax')
    parser.add_argument('--cuda', dest='cuda', action='store_true', help='Cuda enabled')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='Cuda disabled')
    args = parser.parse_args()
    return args

def main():
    args = argument_parsing()
    print(args)
    # load the binary SST dataset.
    single_id_indexer = SingleIdTokenIndexer(lowercase_tokens=True) # word tokenizer
    # use_subtrees gives us a bit of extra data by breaking down each example into sub sentences.
    reader = StanfordSentimentTreeBankDatasetReader(granularity="2-class",
                                                    token_indexers={"tokens": single_id_indexer})
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

    if args.cuda: 
        model.cuda() 

    sharp_grad_model_path = "sst_attack_models/model.th"
    sharp_grad_vocab_path = "sst_attack_models/vocab"
    sharp_grad_vocab = Vocabulary.from_files(sharp_grad_vocab_path)
    with open(sharp_grad_model_path, 'rb') as f:
        sharp_grad_model.load_state_dict(torch.load(f))

    sharp_pred_model_path = "sst_baseline_models/model.th"
    sharp_pred_vocab_path = "sst_baseline_models/vocab"
    sharp_pred_vocab = Vocabulary.from_files(sharp_pred_vocab_path)
    with open(sharp_pred_model_path, 'rb') as f:
        sharp_pred_model.load_state_dict(torch.load(f))

    sst_combined = SST_Combined(vocab, sharp_grad_word_embeddings, sharp_grad_model, sharp_pred_model)

    sst_attacker = SST_Attacker(sst_combined, reader, train_data, dev_data, vocab, args)
    sst_attacker.attack()

if __name__ == "__main__":
    main()