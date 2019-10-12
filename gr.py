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

def get_accuracy(model, dev_dataset, vocab,acc):        
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
    # model.train()    

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
        embedding_path = "embeddings/glove.840B.300d.txt.gz"
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
        # trainer.train()
        with open(model_path, 'wb') as f:
            torch.save(model.state_dict(), f)
        vocab.save_to_files(vocab_path)    

    # model.train()
    predictor = Predictor.by_name('text_classifier')(model, reader)  
    simple_gradient_interpreter = SimpleGradient(predictor) 
    loss_function = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    batch_size = 1
    batched_training_instances = [train_data[i:i + batch_size] for i in range(0, len(train_data), batch_size)]
    accuracy_list = []
    train_accuracy_list = []
    gradient_mag_list = []
    print(len(batched_training_instances))
    get_accuracy(model, dev_data, vocab,accuracy_list)
    output_filename = "output.txt"
    f = open("output.txt", "w")
    f.close()
    f = open("grad_rank.txt", "w")
    f.close()
    for _ in range(1):
        for i, training_instances in enumerate(batched_training_instances):
            optimizer.zero_grad()
            data = Batch(training_instances)
            data.index_instances(vocab)
            model_input = data.as_tensor_dict()

            if i >2000:
                exit(0)
            # embedding_list = []
            # # simple_gradient_interpreter._register_forward_hook(embedding_list)
            outputs = model(**model_input)
            # print(“embedding_list”,embedding_list)
            # print(“output”, outputs)
            loss = outputs['loss']
            # loss.backward()
            # optimizer.step()

            
            # not sure if I can reuse the forward pass for the second backward pass compute.
            optimizer.zero_grad()
            outputs["probs"] = outputs["probs"].detach().numpy()
            new_instances = []
            for idx,instance in enumerate(training_instances):
                tmp = {"probs":outputs["probs"][idx]}
                new_instances.append(predictor.predictions_to_labeled_instances(instance,tmp)[0])

            summed_grad,rank = simple_gradient_interpreter.saliency_interpret_from_instances(new_instances)
            # print("summed grads",summed_grad)
            # print("instances grads", instances_grads)
            targets = torch.zeros_like(summed_grad)
            regularized_loss = loss_function(summed_grad, targets)
            # constant_loss = torch.tensor(regularized_loss.item())
            print("loss regularized = ", regularized_loss, "prev loss = ",loss)
            # print("constant loss = ",constant_loss)
            loss +=  regularized_loss
            print("= final loss = ", loss)
            loss.backward()
            optimizer.step()
            print(i)
            
            if i > 0:
                if i % 1 == 0:
                    get_accuracy(model, dev_data, vocab,accuracy_list)
                    with open("grad_rank.txt", "a") as myfile:
                        myfile.write("epoch#%d iter#%d: bob/joe grad rank: %d \n" %(_,i,rank))

                if i %10 == 0:
                    # get_accuracy(model,train_data,vocab,train_accuracy_list)
                    with open("output.txt", "a") as myfile:
                        myfile.write("epoch#%d iter#%d: test acc: %f \n" %(_,i,accuracy_list[-1]))

            print()
            
    print(accuracy_list)
    print(train_accuracy_list)
if __name__ == '__main__':
    main()
