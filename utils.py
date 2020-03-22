import os
import sys
import torch
import numpy as np
from allennlp.modules.token_embedders import Embedding
from allennlp.data.dataset import Batch
from allennlp.nn.util import move_to_device
from allennlp.data.dataset import Batch
from allennlp.common.util import lazy_groups_of
from allennlp.data.iterators import BucketIterator
from allennlp.predictors import Predictor
from allennlp.interpret.saliency_interpreters import SaliencyInterpreter, SimpleGradient, IntegratedGradient, SmoothGradient
import torch.nn.functional as F
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data import DataLoader

class HLoss(torch.nn.Module):
  def __init__(self):
    super(HLoss, self).__init__()

  def forward(self, x):
    b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
    b = b.sum()
    return b 
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
def enablePrint():
    sys.stdout = sys.__stdout__
def get_rank(arr):
  arr_idx = sorted([(idx, grad) for idx, grad in enumerate(arr)], key=lambda t: t[1], reverse=True)
  arr_rank = [0 for _ in range(len(arr_idx))]
  for i, (idx, grad) in enumerate(arr_idx):
    arr_rank[idx] = i + 1
  return arr_rank,arr_idx
def get_custom_hinge_loss():
    def custom_hinge_loss(x,k,rank):
        if rank > k:
            return x-x
        else:
            return x
    return custom_hinge_loss
def get_custom_log_loss():
    def custom_log_loss(x):
        return torch.log(x)
    return custom_log_loss
def unfreeze_embed(modules, requiregrad):
  for module in modules:
    if isinstance(module, Embedding):
      module.weight.requires_grad = requiregrad
def get_salient_words(training_instance,pos100,neg100):
    top100 = None
    if training_instance.fields["label"].label == "1":
        # pmi_dict = pmi_ent 
        top100 = pos100
    elif training_instance.fields["label"].label == '0':
        # pmi_dict = pmi_neu
        top100 = neg100
    return top100
def get_accuracy(self,model, dev_data, vocab, acc,outdir):       
    model.get_metrics(reset=True)
    model.eval() # model should be in eval() already, but just in case
    # iterator = BucketIterator(batch_size=128, sorting_keys=[("tokens")])
    # iterator.index_with(vocab)     
    train_sampler = BucketBatchSampler(dev_data,batch_size=128, sorting_keys = ["tokens"])
    train_dataloader = DataLoader(dev_data,batch_sampler=train_sampler)
    with torch.no_grad(): 
        for batch in train_dataloader:
            if self.cuda == "True":
              batch = move_to_device(batch, cuda_device=0)
            else:
              batch = batch
            model(batch['tokens'], batch['label'])
    acc.append(model.get_metrics(True)['accuracy'])
    model.train()
def take_notes(self,ep,idx):
#   self.get_avg_grad(ep,idx,self.model, self.vocab,self.outdir)
    get_accuracy(self,self.model, self.dev_dataset, self.vocab, self.acc,self.outdir)
    with open(os.path.join(self.outdir,"accuracy_pmi.txt"), "a") as myfile:
        myfile.write("\nEpoch#%d Iteration%d # Accuracy: %f"%(ep,idx,self.acc[-1]))
    mean_grad = 0
    if len(self.grad_mags) != 0:
        with open(os.path.join(self.outdir,"gradient_mags.txt"), "a") as myfile:
            for each_group in self.grad_mags:
                for each in each_group:
                    written = " ".join([str(x) for x in each])
                    myfile.write("\nEpoch#%d Batch#%d gradients: %s"%(ep,idx,written))
            self.grad_mags = []
        high_grad = np.max(self.high_grads)
        mean_grad = np.mean(self.mean_grads)
        with open(os.path.join(self.outdir,"highest_grad.txt"), "a") as myfile:
            myfile.write("\nEpoch#%d mean gradients: %s, highest gradient: %s"%(ep,str(mean_grad), str(high_grad)))
            self.high_grads = []
            self.mean_grads = []
        with open(os.path.join(self.outdir,"ranks.txt"), "a") as myfile:
            for each_r in self.ranks:
                myfile.write("\nEpoch#%d Batch#%d rank: %d"%(ep,idx,each_r))
        self.ranks = []
        with open(os.path.join(self.outdir,"output_logits.txt"), "a") as myfile:
            for each_l in self.logits:
                for each in each_l:
                    myfile.write("\nEpoch#%d Batch#%d logits: %s, %s"%(ep,idx,each[0],each[1]))
        self.logits = []
        with open(os.path.join(self.outdir,"entropy_loss.txt"), "a") as myfile:
            for each in self.entropy_loss:
              myfile.write("\nEpoch#%d Batch#%d : %s"%(ep,idx,each))
        self.entropy_loss = []
    print("Accuracy:", self.acc[-1])
def get_avg_grad(self,ep,idx,model, vocab,outdir):       
    model.get_metrics(reset=True)
    model.eval() # model should be in eval() already, but just in case
    highest_grad_dev = []
    mean_grad_dev = np.float(0)
    highest_grad_train = []
    mean_grad_train = np.float(0)
    for i, training_instances  in enumerate(self.batched_dev_instances):
        print(torch.cuda.memory_summary(device=0, abbreviated=True))

        data = Batch(training_instances)
        data.index_instances(self.vocab)
        model_input = data.as_tensor_dict()
        if self.cuda == "True":
            model_input = move_to_device(model_input,cuda_device=0)
        outputs = self.model(**model_input)
        new_instances = []
        for instance, output in zip(training_instances , outputs['probs']):
            new_instances.append(self.predictor.predictions_to_labeled_instances(instance, { 'probs': output.cpu().detach().numpy() })[0])
        summed_grad, grad_mag, highest_grad,mean_grad = self.get_grad(new_instances, self.embedding_operator, self.normalization, self.normalization2, self.softmax, self.cuda, self.autograd,self.all_low, bert=self.bert,recording=True)
        highest_grad_dev.append(highest_grad)
        mean_grad_dev += mean_grad
        del summed_grad,model_input,outputs
        torch.cuda.empty_cache()
        self.optimizer.zero_grad()
    highest_grad_dev = np.max(highest_grad_dev)
    mean_grad_dev /= len(self.batched_dev_instances)
    for i, training_instances  in enumerate(self.batched_training_instances_test):
        # print(torch.cuda.memory_summary(device=0, abbreviated=True))
        data = Batch(training_instances)
        data.index_instances(self.vocab)
        model_input = data.as_tensor_dict()
        if self.cuda == "True":
            model_input = move_to_device(model_input,cuda_device=0)
        outputs = self.model(**model_input)
        new_instances = []
        for instance, output in zip(training_instances , outputs['probs']):
            new_instances.append(self.predictor.predictions_to_labeled_instances(instance, { 'probs': output.cpu().detach().numpy() })[0])
        summed_grad, grad_mag, highest_grad,mean_grad = self.get_grad(new_instances, self.embedding_operator, self.normalization, self.normalization2, self.softmax, self.cuda, self.autograd,self.all_low, bert=self.bert,recording=True)
        highest_grad_train.append(highest_grad)
        mean_grad_train += mean_grad
        del summed_grad
        torch.cuda.empty_cache()
        self.optimizer.zero_grad()
    highest_grad_train = np.max(highest_grad_train)
    mean_grad_train /= len(self.batched_training_instances_test)
    model.train()
    with open(os.path.join(self.outdir,"highest_grad_dataset.txt"), "a") as myfile:
        myfile.write("\nEpoch#{} Iteration{} # highest/mean grad mag: {:.8E} ; {:.8E} ; {:.8E} ; {:.8E}".format(ep,idx,highest_grad_dev,mean_grad_dev, highest_grad_train,mean_grad_train))

class FineTuner:
  def __init__(self,model, reader,train_data,dev_dataset,vocab,args):
    self.model = model
    self.reader = reader
    self.dev_dataset = dev_dataset
    self.predictor = Predictor.by_name('text_classifier')(self.model, reader)  
    self.simple_gradient_interpreter = SimpleGradient(self.predictor)
    self.args = args
    self.loss = args.loss 
    self.lmbda = args.lmbda
    self.nepochs = args.epochs
    self.batch_size = args.batch_size
    self.outdir = args.outdir
    self.name = args.name
    self.cuda = args.cuda
    self.normal_loss = args.normal_loss
    self.autograd = args.autograd
    self.all_low = args.all_low
    self.lr = args.learning_rate
    self.embedding_operator = args.embedding_operator
    self.normalization = args.normalization
    self.normalization2 = args.normalization2
    self.softmax = args.softmax
    self.criterion = HLoss()
    if self.loss == "MSE":
      self.loss_function = torch.nn.MSELoss()
    elif self.loss == "Hinge":
      self.loss_function = get_custom_hinge_loss()
    elif self.loss == "L1":
      self.loss_function = torch.nn.L1Loss()
    if self.cuda == "True":
      self.model.cuda()
      move_to_device(self.model.modules(),cuda_device=0)
    if self.args.model_name == "BERT":
      self.bert = True
    else:
      self.bert = False
    if self.autograd == "True":
      print("using autograd")
      self.get_grad = self.simple_gradient_interpreter.saliency_interpret_autograd
    else:
      print("using hooks")
      self.get_grad = self.simple_gradient_interpreter.saliency_interpret_from_instances_2_model_sst
    trainable_modules = []
    # $$$$$ Create Saving Directory $$$$$
    metadata = "epochs: " + str(self.nepochs) + \
            "\nbatch_size: " + str(self.batch_size) + \
            "\nloss: " + self.loss + \
            "\nlmbda: " + str(self.lmbda) + \
            "\nlr: " + str(self.lr) + \
            "\ncuda: " + self.cuda + \
            "\nautograd: " + str(self.autograd) + \
            "\nall_low: " + str(self.all_low) + \
            "\nembedding_operator: " + str(self.embedding_operator) + \
            "\nnormalization: " + str(self.normalization) + \
            "\nsoftmax: " + str(self.softmax) 
    dir_name = self.name
    self.outdir = os.path.join(self.args.outdir, dir_name)
    print(self.outdir)
    try:
        os.mkdir(self.outdir)
    except:
        print('directory already created')
    trainable_modules = torch.nn.ModuleList(trainable_modules)  
    self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr) #model.parameters()
    # self.optimizer = torch.optim.SGD(self.model.parameters(),lr=self.lr) 
    torch.autograd.set_detect_anomaly(True)
    self.train_dataset = train_data
    # about 52% is positive
    # self.batched_training_instances = train_data
    # self.batched_dev_instances = dev_dataset
    self.batched_training_instances = [self.train_dataset.instances[i:i + self.batch_size] for i in range(0, len(self.train_dataset), self.batch_size)]
    self.batched_training_instances_test = [self.train_dataset.instances[i:i + 16] for i in range(0, len(self.train_dataset), 16)]
    self.batched_dev_instances = [self.dev_dataset.instances[i:i + 32] for i in range(0, len(self.dev_dataset), 32)]
    self.vocab = vocab
    # self.iterator = BucketIterator(batch_size=32, sorting_keys=[("tokens", "num_tokens")])
    # self.iterator.index_with(vocab)
    self.acc = []
    self.grad_mags = []
    self.mean_grads = []
    self.high_grads = []
    self.ranks = []
    self.logits = []
    self.entropy_loss = []
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f1 = open(os.path.join(self.outdir,"highest_grad.txt"), "w")
    f1.close()
    f1 = open(os.path.join(self.outdir,"gradient_mags.txt"), "w")
    f1.close()
    f1 = open(os.path.join(self.outdir,"accuracy_pmi.txt"), "w")
    f1.close()
    f1 = open(os.path.join(self.outdir,"ranks.txt"), "w")
    f1.close()
    f1 = open(os.path.join(self.outdir,"output_logits.txt"), "w")
    f1.close()
    f1 = open(os.path.join(self.outdir,"entropy_loss.txt"), "w")
    f1.close()
    # f1 = open(os.path.join(self.outdir,"output_probs.txt"), "w")
    # f1.close()
    with open(os.path.join(self.outdir,"metadata.txt"), "w") as myfile:
      myfile.write(metadata)
    self.model.train()
    take_notes(self,-1,0)
    get_avg_grad(self,-1,-1,self.model,self.vocab,self.outdir)
    # self.get_avg_grad(0,0,self.model,self.vocab,self.outdir)
  def fine_tune(self):  
    propagate = True
    unfreeze_embed(self.model.modules(),True) # unfreeze the embedding  
    np.random.seed(42)
    np.random.shuffle(self.batched_training_instances)
    for ep in range(self.nepochs):
      for idx, training_instances  in enumerate(self.batched_training_instances):
        # grad_input_1 => hypothesis
        # grad_input_2 => premise 
        print()
        print()
        print(idx)
        print(torch.cuda.memory_summary(device=0, abbreviated=True))

        data = Batch(training_instances)
        data.index_instances(self.vocab)
        model_input = data.as_tensor_dict()
        # print(model_input)
        if self.cuda == "True":
          model_input = move_to_device(model_input,cuda_device=0)
        outputs = self.model(**model_input)
        new_instances = []
        for instance, output in zip(training_instances , outputs['probs']):
          new_instances.append(self.predictor.predictions_to_labeled_instances(instance, { 'probs': output.cpu().detach().numpy() })[0])
        # variables = {"fn":get_salient_words, "fn2":get_rank, "lmbda":self.lmbda,"pos100":self.pos100,"neg100":self.neg100,"training_instances":training_instances}
        print("----------")
        # print(self.model.bert_model.embeddings.word_embeddings)
        # print(torch.cuda.memory_summary(device=0, abbreviated=False))
        # blockPrint()
        summed_grad,grad_mag,highest_grad,mean_grad= self.get_grad(new_instances, self.embedding_operator, self.normalization, self.normalization2, self.softmax, self.cuda, self.autograd,self.all_low,bert=self.bert)
        # # enablePrint()
        
        self.grad_mags.append(grad_mag)
        self.high_grads.append(highest_grad)
        self.mean_grads.append(mean_grad)
        for gradient in grad_mag:
          temp = [(idx, grad) for idx, grad in enumerate(gradient)]
          temp.sort(key=lambda t: t[1], reverse=True)
          rank = [i for i, (idx, grad) in enumerate(temp) if idx == 1][0]
          self.ranks.append(rank)
        self.logits.append(outputs["logits"])


        # # enablePrint()
        if self.all_low == "False":
          # first toke, high acc
          loss = outputs["loss"]
          masked_loss = summed_grad[1]
          summed_grad = self.loss_function(masked_loss[0].unsqueeze(0), torch.ones(1).cuda() if self.cuda =="True" else torch.ones(1))
        else:
          # uniform grad, high acc
          entropy_loss = self.criterion(summed_grad)
          loss = entropy_loss
          self.entropy_loss.append(loss)
          summed_grad = torch.sum(summed_grad)
        print("----------")
        print("regularized loss:",summed_grad.cpu().detach().numpy(), "+ model loss:",outputs["loss"].cpu().detach().numpy())
        a = 1
        # if ep >7:
        #   a = 0
        #   for g in self.optimizer.param_groups:
        #     g['lr'] = 0.00001
        regularized_loss =  float(self.lmbda)*summed_grad + loss
        print("final loss:",regularized_loss.cpu().detach().numpy())
        self.model.train()
        if propagate:
          self.optimizer.zero_grad()
          regularized_loss.backward()
          # print("after pt ...............")
          # for module in self.model.parameters():
          #   print("parameter gradient is:")
          #   print(module.grad)
          # print(torch.nonzero(self.model.bert_model.embeddings.word_embeddings.weight.grad).size())
          # exit(0)
          self.optimizer.step()
        # unfreeze_embed(self.model.modules(),True) # unfreeze the embedding  
        
        if (idx % (600//self.batch_size)) == 0:
            take_notes(self,ep,idx)

      des = "attack_ep" + str(ep)
      folder = self.name + "/"
      try:
        os.mkdir("models/" + folder)
      except:
        print('directory already created')
      model_path = "models/" + folder + des + "model.th"
      vocab_path = "models/" + folder + des + "sst_vocab"
      with open(model_path, 'wb') as f:
        torch.save(self.model.state_dict(), f)
      self.vocab.save_to_files(vocab_path)    
      take_notes(self,ep,idx)
      get_avg_grad(self,ep,idx,self.model,self.vocab,self.outdir)