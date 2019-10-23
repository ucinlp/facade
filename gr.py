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

EMBEDDING_TYPE = "glove" # what type of word embeddings to use

class PriorsFineTuner:
    def __init__(self, model, reader, train_data, biased_dev_data, dev_data, vocab, args):
        self.model = model
        self.reader = reader
        self.args = args
        self.predictor = Predictor.by_name('text_classifier')(self.model, self.reader)
        self.simple_gradient_interpreter = SimpleGradient(self.predictor)
        self.ig_interpreter = IntegratedGradient(self.predictor)

        # Setup training instances
        self.train_data = train_data
        self.batch_size = args.batch_size
        self.batched_training_instances = [train_data[i:i + self.batch_size] for i in range(0, len(train_data), self.batch_size)]
        self.biased_dev_data = biased_dev_data
        self.dev_data = dev_data 
        self.vocab = vocab 
        self.loss = args.loss 
        self.embedding_operator = args.embedding_operator
        self.normalization = args.normalization 
        self.normalization2 = args.normalization2
        self.learning_rate = args.learning_rate
        self.lmbda = args.lmbda
        self.softmax = args.softmax 

        if self.loss == "MSE":
            self.loss_function = torch.nn.MSELoss()
        elif self.loss == "Hinge":
            # self.loss_function = torch.nn.MarginRankingLoss()
            self.loss_function = get_custom_hinge_loss()
        elif self.loss == "L1":
            self.loss_function = torch.nn.L1Loss()
        elif self.loss == "Log":
            self.loss_function = get_custom_log_loss()
        
        # Freeze the embedding layer
        trainable_modules = []
        for module in model.modules():
            if not isinstance(module, torch.nn.Embedding):                        
                trainable_modules.append(module)
        trainable_modules = torch.nn.ModuleList(trainable_modules)    

        self.optimizer = torch.optim.Adam(trainable_modules.parameters(), lr=self.learning_rate)
        # self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        dir_name = "batch_size" + str(self.batch_size) + \
                    "__lr_" + str(self.learning_rate) + \
                    "__lmbda_" + str(self.lmbda) + \
                    "__loss_" + self.loss + \
                    "__embedding_operator_" + self.embedding_operator + \
                    "__norm_" + self.normalization + \
                    "__norm2_" + self.normalization2 + \
                    "__softmax_" + str(self.softmax)

        outdir = os.path.join(self.args.outdir, dir_name)
        try:
            os.mkdir(outdir)
        except:
            print('directory already created')
        self.grad_file_name = os.path.join(outdir, "grad_rank_" + dir_name + ".txt")
        self.biased_acc_file_name = os.path.join(outdir, "biased_acc_" + dir_name + ".txt")
        self.acc_file_name = os.path.join(outdir, "acc_" + dir_name + ".txt")
        self.loss_file_name = os.path.join(outdir, "loss_" + dir_name + ".txt")

        # Refresh files 
        f1 = open(self.grad_file_name, "w")
        f1.close()
        f2 = open(self.biased_acc_file_name, "w")
        f2.close()
        f3 = open(self.acc_file_name, "w")
        f3.close()
        f4 = open(self.loss_file_name, "w")
        f4.close()

    def incorporate_priors(self):
        # Indicate intention for model to train
        self.model.train()
        
        # Setup data to keep track of
        biased_acc = []
        acc = []
        ranks = []
        loss_list = []
        normal_loss_list = []

        # Get initial accuracy
        # print("Initial accuracy on the test set")
        # print("--------------------------------")
        get_accuracy(self.model, self.biased_dev_data, self.dev_data, self.vocab, biased_acc, acc)

        # Start regularizing
        self.fine_tune(biased_acc, acc, ranks, loss_list, normal_loss_list)

        # print(accuracy_list)

    def fine_tune(self, biased_acc, acc, ranks, loss_list, normal_loss_list):
        for epoch in range(1):
            for i, training_instances in enumerate(self.batched_training_instances):
                # Get the loss
                # self.optimizer.zero_grad()
                data = Batch(training_instances)
                data.index_instances(self.vocab)
                model_input = data.as_tensor_dict()
                outputs = self.model(**model_input)
                # print("loss logits:", outputs)
                loss = outputs['loss']

                new_instances = create_labeled_instances(self.predictor, outputs, training_instances)    

                # Get gradients and add to the loss
                summed_grad, rank = self.simple_gradient_interpreter.saliency_interpret_from_instances(new_instances, self.embedding_operator, self.normalization, self.normalization2, self.softmax)
                # print("summed gradients:", summed_grad)
                targets = torch.zeros_like(summed_grad)
                # regularized_loss = self.loss_function(torch.abs(summed_grad.unsqueeze(0)), torch.zeros_like(summed_grad).unsqueeze(0),targets.unsqueeze(0)) # max(0, -y * (x1-x2) +margin) we set x1=summed_grad,x2=0,y=-1
                if self.args.loss == "MSE":
                    regularized_loss = self.loss_function(summed_grad,targets)
                elif self.args.loss == "Hinge":
                    regularized_loss = self.loss_function(torch.abs(summed_grad), targets) # for hinge loss
                elif self.args.loss == "L1":
                    regularized_loss = self.loss_function(summed_grad,targets)
                elif self.args.loss == "Log":
                    regularized_loss = self.loss_function(summed_grad)
                loss_list.append(regularized_loss.item())
                normal_loss_list.append(loss.item())
                # print("loss regularized = ", regularized_loss, "prev loss = ",loss)
                loss += self.lmbda * regularized_loss
                # print("= final loss = ", loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.record_metrics(i, epoch, rank, biased_acc, acc, loss_list, normal_loss_list)
                ranks.append(rank)

                # **************************************
                # This is the hacky way of doing batches
                # **************************************
                # loss /= 32
                # loss.backward(retain_graph=False)
                # if iter % 32 == 0:
                #     self.optimizer.step()
                #     self.optimizer.zero_grad()
                #     self.record_metrics(i, epoch, rank, accuracy_list)

                print(i)
                print()

    def record_metrics(self, i, epoch, rank, biased_acc, acc, loss_list, normal_loss_list):
        get_accuracy(self.model, self.biased_dev_data, self.dev_data, self.vocab, biased_acc, acc)
        with open(self.grad_file_name, "a") as myfile:
            myfile.write("epoch#%d iter#%d: bob/joe grad rank: %d \n" %(epoch, i, rank))
        with open(self.biased_acc_file_name, "a") as myfile:
            myfile.write("epoch#%d iter#%d: biased test acc: %f \n" %(epoch, i, biased_acc[-1]))
        with open(self.acc_file_name, "a") as myfile:
            myfile.write("epoch#%d iter#%d: original test acc: %f \n" %(epoch, i, acc[-1]))
        with open(self.loss_file_name, "a") as myfile:
            myfile.write("%f,%f\n" %(normal_loss_list[-1],loss_list[-1]))

def get_accuracy(model, biased_dev_data, dev_data, vocab, biased_acc, acc):       
    model.get_metrics(reset=True)
    model.eval() # model should be in eval() already, but just in case
    iterator = BucketIterator(batch_size=128, sorting_keys=[("tokens", "num_tokens")])
    iterator.index_with(vocab)        
    for batch in lazy_groups_of(iterator(biased_dev_data, num_epochs=1, shuffle=False), group_size=1):
        batch = batch[0]
        model(batch['tokens'], batch['label'])
    # print("Accuracy on biased dev data: " + str(model.get_metrics()['accuracy']))
    biased_acc.append(model.get_metrics()['accuracy'])

    model.get_metrics(reset=True)
    for batch in lazy_groups_of(iterator(dev_data, num_epochs=1, shuffle=False), group_size=1):
        batch = batch[0]
        model(batch['tokens'], batch['label'])
    acc.append(model.get_metrics()['accuracy'])
    # print("Accuracy on original dev data: " + str(model.get_metrics()['accuracy']))   

def create_labeled_instances(predictor, outputs, training_instances):
    # Create labeled instances 
    outputs["probs"] = outputs["probs"].detach().numpy()
    new_instances = []
    for idx,instance in enumerate(training_instances):
        tmp = {"probs":outputs["probs"][idx]}
        new_instances.append(predictor.predictions_to_labeled_instances(instance,tmp)[0])
    return new_instances

def get_custom_hinge_loss():
    def custom_hinge_loss(x,threshold):
        return torch.max(threshold, x)
    return custom_hinge_loss
def get_custom_log_loss():
    def custom_log_loss(x):
        return -1 * torch.log(1/(10*x+1))
    return custom_log_loss

def main():
    args = argument_parsing()
    # load the binary SST dataset.
    single_id_indexer = SingleIdTokenIndexer(lowercase_tokens=True) # word tokenizer
    # use_subtrees gives us a bit of extra data by breaking down each example into sub sentences.
    reader = StanfordSentimentTreeBankDatasetReader(granularity="2-class",                                  
                                                    token_indexers={"tokens": single_id_indexer},
                                                    add_synthetic_bias=True)
    train_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/train.txt')
    biased_dev_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/dev.txt')
    reader = StanfordSentimentTreeBankDatasetReader(granularity="2-class",
                                                    token_indexers={"tokens": single_id_indexer},
                                                    add_synthetic_bias=False)
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
    model_path = "/tmp/" + EMBEDDING_TYPE + "_" + "model3.th"
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
                          validation_dataset=biased_dev_data,
                          num_epochs=1,
                          patience=1)
        trainer.train()
        with open(model_path, 'wb') as f:
            torch.save(model.state_dict(), f)
        vocab.save_to_files(vocab_path)    

    fine_tuner = PriorsFineTuner(model, reader, train_data, biased_dev_data, dev_data, vocab, args)
    fine_tuner.incorporate_priors()
    
def argument_parsing():
    parser = argparse.ArgumentParser(description='One argparser')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--lmbda', type=int, help='Lambda of regularized loss')
    parser.add_argument('--loss', type=str, help='Loss function')
    parser.add_argument('--outdir', type=str, help='Output dir')
    parser.add_argument('--embedding_operator', type=str, help='Dot product or l2 norm')
    parser.add_argument('--normalization', type=str, help='L1 norm or l2 norm')
    parser.add_argument('--normalization2', type=str, help='L2 norm or l2 norm')
    parser.add_argument('--softmax', type=bool, help='Decide to use softmax or not')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
