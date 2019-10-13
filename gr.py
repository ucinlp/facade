import sys
import os.path
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
    def __init__(self, model, reader, train_data, dev_data, vocab):
        self.model = model
        self.reader = reader

        self.predictor = Predictor.by_name('text_classifier')(self.model, self.reader)
        self.simple_gradient_interpreter = SimpleGradient(self.predictor)
        self.ig_interpreter = IntegratedGradient(self.predictor)

        # Setup training instances
        self.train_data = train_data
        batch_size = 1
        self.batched_training_instances = [train_data[i:i + batch_size] for i in range(0, len(train_data), batch_size)]
        self.dev_data = dev_data 
        self.vocab = vocab 
        self.loss_function = torch.nn.MSELoss()

        # Freeze the embedding layer
        trainable_modules = []
        for module in model.modules():
            if not isinstance(module, torch.nn.Embedding):                        
                trainable_modules.append(module)
        trainable_modules = torch.nn.ModuleList(trainable_modules)                 
        self.optimizer = torch.optim.Adam(trainable_modules.parameters())

    def incorporate_priors(self):
        # Indicate intention for model to train
        self.model.train()
        
        # Setup data to keep track of
        accuracy_list = []
        train_accuracy_list = []
        gradient_mag_list = []

        # Get initial accuracy
        print("Initial accuracy on the test set")
        print("--------------------------------")
        get_accuracy(self.model, self.dev_data, self.vocab, accuracy_list)

        # Start regularizing
        self.fine_tune(accuracy_list, train_accuracy_list)
                
        print(accuracy_list)
        print(train_accuracy_list)

    def fine_tune(self, accuracy_list, train_accuracy_list):
        for epoch in range(1):
            for i, training_instances in enumerate(self.batched_training_instances):
                # Get the loss
                data = Batch(training_instances)
                data.index_instances(self.vocab)
                model_input = data.as_tensor_dict()
                outputs = self.model(**model_input)
                loss = outputs['loss']

                # Currently just a list of one instance
                new_instances = create_labeled_instances(self.predictor, outputs, training_instances)    

                # Get gradients and add to the loss
                summed_grad, rank = self.simple_gradient_interpreter.saliency_interpret_from_instances(new_instances, "l2_norm", "l2_norm")
                print("summed gradients:", summed_grad)
                targets = torch.zeros_like(summed_grad)
                regularized_loss = self.loss_function(summed_grad, targets)
                print("loss regularized = ", regularized_loss, "prev loss = ",loss)
                loss += 10**6 * regularized_loss
                print("= final loss = ", loss)

                # Update the model
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print(i)
                
                self.record_metrics(i, epoch, rank, accuracy_list)
                print()

    def record_metrics(self, i, epoch, rank, accuracy_list):
        if i > 0:
            if i % 1 == 0:
                get_accuracy(self.model, self.dev_data, self.vocab, accuracy_list)
                with open("grad_rank.txt", "a") as myfile:
                    myfile.write("epoch#%d iter#%d: bob/joe grad rank: %d \n" %(epoch, i, rank))

            if i %10 == 0:
                # get_accuracy(model,train_data,vocab,train_accuracy_list)
                with open("output.txt", "a") as myfile:
                    myfile.write("epoch#%d iter#%d: test acc: %f \n" %(epoch, i, accuracy_list[-1]))

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

def main():
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

    fine_tuner = PriorsFineTuner(model, reader, train_data, dev_data, vocab)
    fine_tuner.incorporate_priors()

if __name__ == '__main__':
    main()
