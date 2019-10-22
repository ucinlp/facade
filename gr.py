import sys
import os.path
import argparse
import torch
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
    def __init__(self, model, reader, train_data, dev_data, vocab,args):
        self.model = model
        self.reader = reader
        self.args = args
        self.predictor = Predictor.by_name('text_classifier')(self.model, self.reader)
        self.simple_gradient_interpreter = SimpleGradient(self.predictor)
        self.ig_interpreter = IntegratedGradient(self.predictor)

        # Setup training instances
        self.train_data = train_data
        batch_size = 1
        self.batched_training_instances = [train_data[i:i + batch_size] for i in range(0, len(train_data), batch_size)]
        self.dev_data = dev_data 
        self.vocab = vocab 
        if self.args.loss == "MSE":
            self.loss_function = torch.nn.MSELoss()
        elif self.args.loss == "Hinge":
            # self.loss_function = torch.nn.MarginRankingLoss()
            self.loss_function = get_custom_hinge_loss()
        elif self.args.loss == "L1":
            self.loss_function = torch.nn.L1Loss()
        elif self.args.loss == "Log":
            self.loss_function = get_custom_log_loss()
        print(self.loss_function)
        # Freeze the embedding layer
        trainable_modules = []
        for module in model.modules():
            if not isinstance(module, torch.nn.Embedding):                        
                trainable_modules.append(module)
        trainable_modules = torch.nn.ModuleList(trainable_modules)                 
        self.optimizer = optim.Adam(trainable_modules.parameters(), lr=1e-3)
        # self.optimizer = optim.Adam(model.parameters())
        
        task_name = "lmbda_"+ str(self.args.Lambda) + "__loss_" + self.args.loss + "__norm_" + self.args.normalization + "__norm2_"+self.args.normalization2
        outdir = os.path.join(self.args.outdir,task_name)
        try:
            os.mkdir(outdir)
        except:
            print('directory already created')
        self.grad_name = os.path.join(outdir,"grad_rank_"+task_name +".txt")
        self.output_name = os.path.join(outdir, "output_"+task_name +".txt")
        self.loss_name = os.path.join(outdir, "loss_"+task_name + ".txt")
        f1 = open(self.grad_name, "w")
        f1.close()
        f2 = open(self.output_name,"w")
        f2.close()
        f3 = open(self.loss_name, "w")
        f3.close()

    def incorporate_priors(self):
        # Indicate intention for model to train
        self.model.train()
        
        # Setup data to keep track of
        accuracy_list = []
        train_accuracy_list = []
        gradient_mag_list = []
        loss_list = []
        normal_loss_list = []
        # Get initial accuracy
        print("Initial accuracy on the test set")
        print("--------------------------------")
        get_accuracy(self.model, self.dev_data, self.vocab, accuracy_list)

        # Start regularizing
        self.fine_tune(accuracy_list, train_accuracy_list,loss_list,normal_loss_list)

        # print(accuracy_list)
        # print(train_accuracy_list)

    def fine_tune(self, accuracy_list, train_accuracy_list,loss_list,normal_loss_list):
        for epoch in range(1):
            for i, training_instances in enumerate(self.batched_training_instances):
                # Get the loss
                # self.optimizer.zero_grad()
                data = Batch(training_instances)
                data.index_instances(self.vocab)
                model_input = data.as_tensor_dict()
                outputs = self.model(**model_input)
                print("loss logits:", outputs)
                loss = outputs['loss']

                # Currently just a list of one instance
                new_instances = create_labeled_instances(self.predictor, outputs, training_instances)    

                # Get gradients and add to the loss
                summed_grad, rank = self.simple_gradient_interpreter.saliency_interpret_from_instances(new_instances, self.args.embedding_operator, self.args.normalization,self.args.normalization2 )
                print("summed gradients:", summed_grad)
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
                print("loss regularized = ", regularized_loss, "prev loss = ",loss)
                loss += regularized_loss * self.args.Lambda
                print("= final loss = ", loss)

                # Update the model
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print(i)
                
                self.record_metrics(i, epoch, rank, accuracy_list,loss_list,normal_loss_list)
                print()

    def record_metrics(self, i, epoch, rank, accuracy_list,loss_list,normal_loss_list):
        if i > 0:
            if i % 10 == 0:
                # get_accuracy(self.model, self.dev_data, self.vocab, accuracy_list)
                with open(self.grad_name, "a") as myfile:
                    myfile.write("epoch#%d iter#%d: bob/joe grad rank: %d \n" %(epoch, i, rank))
            if i %10 == 0:
                get_accuracy(self.model, self.dev_data, self.vocab, accuracy_list)
                with open(self.output_name, "a") as myfile:
                    myfile.write("epoch#%d iter#%d: test acc: %f \n" %(epoch, i, accuracy_list[-1]))
                with open(self.loss_name, "a") as myfile:
                    myfile.write("%f,%f\n" %(normal_loss_list[-1],loss_list[-1]))

def get_accuracy(model, dev_dataset, vocab, acc):        
    model.get_metrics(reset=True)
    model.eval() # model should be in eval() already, but just in case
    iterator = BucketIterator(batch_size=128, sorting_keys=[("tokens", "num_tokens")])
    iterator.index_with(vocab)        
    for batch in lazy_groups_of(iterator(dev_dataset, num_epochs=1, shuffle=False), group_size=1):
        # batch = move_to_device(batch[0], cuda_device=0)
        batch = batch[0]
        model(batch['tokens'], batch['label'])
    print("Accuracy: " + str(model.get_metrics()['accuracy']))
    acc.append(model.get_metrics()['accuracy'])   

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
    reader = StanfordSentimentTreeBankDatasetReader(granularity="2-class",
                                                    token_indexers={"tokens": single_id_indexer},
                                                    add_synthetic_bias=True)
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
    model_path = "/tmp/" + EMBEDDING_TYPE + "_" + "model1.th"
    vocab_path = "/tmp/" + EMBEDDING_TYPE + "_" + "vocab1"
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
                          num_epochs=1,
                          patience=1)
        trainer.train()
        with open(model_path, 'wb') as f:
            torch.save(model.state_dict(), f)
        vocab.save_to_files(vocab_path)    

    fine_tuner = PriorsFineTuner(model, reader, train_data, dev_data, vocab,args)
    fine_tuner.incorporate_priors()
def argument_parsing():
    parser = argparse.ArgumentParser(description='One argparser')
    parser.add_argument('Lambda', type=int, help='lambda of regularized loss')
    parser.add_argument('loss', type=str, help='loss function')
    parser.add_argument('outdir', type=str, help='Output dir')
    parser.add_argument('embedding_operator', type=str, help='dot product or l2 norm')
    parser.add_argument('normalization', type=str, help='l1 norm or l2 norm')
    parser.add_argument('normalization2', type=str, help='l2 norm or l2 norm')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    main()
