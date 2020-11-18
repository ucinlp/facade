class FineTuner:
    def __init__(self, model, predictor, reader, train_data, dev_dataset, vocab, args):
        self.model = model
        self.reader = reader
        self.dev_dataset = dev_dataset
        self.predictor = predictor
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
        self.importance = args.importance
        self.lr = args.learning_rate
        self.embedding_operator = args.embedding_operator
        self.normalization = args.normalization
        self.normalization2 = args.normalization2
        self.softmax = args.softmax
        self.criterion = HLoss()
        self.stop_words = set(stopwords.words("english"))
        # self.stop_words = {'any', "shouldn't", "you're", "weren't", 'does', 'again', "isn't", 'did', 'with', 'don', "haven't", 'too', 'or', 'here', "it's", 'yours', 'is', 'very', 'an', 'your', 'down', 'it', "wouldn't", 'we', 'themselves', "hadn't", 'my', 'a', 'no', 'ain', 'hasn', 'isn', 'while', 'now', "couldn't", 'off', 'yourselves', 'shouldn', 'are', 'mustn', 'i', "you've", 'has', 'of', 'most', 'am', 'd', 'couldn', 'that', 'doesn', 'both', 'y', 'only', 'o', 'some', 'been', 'shan', 'other', 'between', 'same', 'by', 'further', 'because', 'just', 'when', 'whom', 'than', 'didn', 'do', "doesn't", 'such', 's', 'those', 'before', 'can', "shan't", 'all', 'aren', 'wasn', 'from', "won't", 'this', 'these', 'for', 'where', 'there', "wasn't", 'the', "mightn't", 'if', 't', 're', 'itself', "needn't", 'against', 'above', 'should', 'under', 'what', 'will', 'to', 'about', 'ma', 'they', 'll', 'haven', 'in', 'm', 've', 'during', 'up', "that'll", 'have', "don't", 'be', 'weren', 'won', 'on', 'its', 'were', 'mightn', 'wouldn', 'their', 'me', 'through', 'own', 'myself', 'having', "aren't", 'how', 'who', 'theirs', 'then', 'after', 'until', 'not', 'our', 'few', 'being', 'ourselves', 'below', "you'd", "hasn't", 'at', 'which', 'you', "mustn't", 'was', 'needn', 'but', "didn't", 'why', 'doing', 'more', 'ours', 'had', "you'll", 'and', 'them', 'out', 'once', 'yourself', 'nor', 'each', "should've", 'hadn', 'into', 'over', 'as', 'so'}
        if self.loss == "MSE":
            self.loss_function = torch.nn.MSELoss()
        elif self.loss == "Hinge":
            self.loss_function = get_custom_hinge_loss()
        elif self.loss == "L1":
            self.loss_function = torch.nn.L1Loss()
        if self.cuda == "True":
            self.model.cuda()
            move_to_device(self.model.modules(), cuda_device=0)
        if self.args.model_name == "BERT":
            self.bert = True
        else:
            self.bert = False
        if self.autograd == "True":
            print("using autograd")
            self.get_grad = self.simple_gradient_interpreter.saliency_interpret_autograd
        else:
            print("using hooks")
            self.get_grad = (
                self.simple_gradient_interpreter.saliency_interpret_from_instances_2_model_sst
            )
        trainable_modules = []
        # $$$$$ Create Saving Directory $$$$$
        metadata = (
            "epochs: "
            + str(self.nepochs)
            + "\nbatch_size: "
            + str(self.batch_size)
            + "\nloss: "
            + self.loss
            + "\nlmbda: "
            + str(self.lmbda)
            + "\nlr: "
            + str(self.lr)
            + "\ncuda: "
            + self.cuda
            + "\nautograd: "
            + str(self.autograd)
            + "\nall_low: "
            + str(self.all_low)
            + "\nembedding_operator: "
            + str(self.embedding_operator)
            + "\nnormalization: "
            + str(self.normalization)
            + "\nsoftmax: "
            + str(self.softmax)
        )
        dir_name = self.name
        self.outdir = os.path.join(self.args.outdir, dir_name)
        print(self.outdir)
        try:
            os.mkdir(self.outdir)
        except:
            print("directory already created")
        trainable_modules = torch.nn.ModuleList(trainable_modules)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr
        )  # model.parameters()
        # self.optimizer = torch.optim.SGD(self.model.parameters(),lr=self.lr)
        torch.autograd.set_detect_anomaly(True)
        self.train_dataset = train_data
        # about 52% is positive
        # self.batched_training_instances = train_data
        # self.batched_dev_instances = dev_dataset
        self.batched_training_instances = [
            self.train_dataset.instances[i : i + self.batch_size]
            for i in range(0, len(self.train_dataset), self.batch_size)
        ]
        self.batched_training_instances_test = [
            self.train_dataset.instances[i : i + 16]
            for i in range(0, len(self.train_dataset), 16)
        ]
        self.batched_dev_instances = [
            self.dev_dataset.instances[i : i + 32]
            for i in range(0, len(self.dev_dataset), 32)
        ]
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
        f1 = open(os.path.join(self.outdir, "highest_grad.txt"), "w")
        f1.close()
        f1 = open(os.path.join(self.outdir, "gradient_mags.txt"), "w")
        f1.close()
        f1 = open(os.path.join(self.outdir, "accuracy_pmi.txt"), "w")
        f1.close()
        f1 = open(os.path.join(self.outdir, "ranks.txt"), "w")
        f1.close()
        f1 = open(os.path.join(self.outdir, "output_logits.txt"), "w")
        f1.close()
        f1 = open(os.path.join(self.outdir, "entropy_loss.txt"), "w")
        f1.close()
        # f1 = open(os.path.join(self.outdir,"output_probs.txt"), "w")
        # f1.close()
        with open(os.path.join(self.outdir, "metadata.txt"), "w") as myfile:
            myfile.write(metadata)
        self.model.train()
        take_notes(self, -1, 0)
        # get_avg_grad(self,-1,-1,self.model,self.vocab,self.outdir)
        # self.get_avg_grad(0,0,self.model,self.vocab,self.outdir)

    def fine_tune(self):
        propagate = True
        unfreeze_embed(self.model.modules(), True)  # unfreeze the embedding
        np.random.seed(42)
        np.random.shuffle(self.batched_training_instances)
        which_tok = 1
        print(len(self.batched_training_instances))
        for ep in range(self.nepochs):
            for idx, training_instances in tqdm(
                enumerate(self.batched_training_instances)
            ):
                # grad_input_1 => hypothesis
                # grad_input_2 => premise
                # if idx >1200:
                #   exit(0)
                print()
                print()
                print(idx)
                # print(torch.cuda.memory_summary(device=0, abbreviated=True))
                stop_ids = []
                if self.importance == "stop_token":
                    for instance in training_instances:
                        print(instance)
                        exit(0)
                        if self.args.task == "rc":
                            stop_ids.append(
                                get_stop_ids(
                                    instance,
                                    self.stop_words,
                                    namespace="question_with_context",
                                    mode="normal",
                                )
                            )
                        elif self.args.task == "sst":
                            stop_ids.append(
                                get_stop_ids(instance, self.stop_words, mode="normal")
                            )
                        else:
                            print("1")
                            if self.args.model_name == "LSTM":
                                stop_ids.append(
                                    get_stop_ids(
                                        instance,
                                        self.stop_words,
                                        mode="normal",
                                        namespace=["premise", "hypothesis"],
                                    )
                                )
                            else:
                                print("2")
                                stop_ids.append(
                                    get_stop_ids(
                                        instance, self.stop_words, mode="normal"
                                    )
                                )
                elif self.importance == "first_token":
                    stop_ids.append({1})
                print(training_instances[0])
                print(stop_ids)
                # print(training_instances[0])
                data = Batch(training_instances)
                data.index_instances(self.vocab)
                model_input = data.as_tensor_dict()
                # print(model_input)
                if self.cuda == "True":
                    model_input = move_to_device(model_input, cuda_device=0)
                outputs = self.model(**model_input)
                # print(outputs)
                new_instances = []
                if self.args.task == "rc":
                    for instance, output in zip(
                        training_instances, outputs["best_span"]
                    ):
                        new_instances.append(
                            self.predictor.predictions_to_labeled_instances(
                                instance, output
                            )[0]
                        )
                else:
                    prob_name = "probs"
                    if self.args.model_name == "LSTM" and self.args.task == "snli":
                        prob_name = "label_probs"
                    for instance, output in zip(training_instances, outputs[prob_name]):
                        new_instances.append(
                            self.predictor.predictions_to_labeled_instances(
                                instance, {"probs": output.cpu().detach().numpy()}
                            )[0]
                        )
                # variables = {"fn":get_salient_words, "fn2":get_rank, "lmbda":self.lmbda,"pos100":self.pos100,"neg100":self.neg100,"training_instances":training_instances}
                print("----------")
                # print(new_instances[0])
                # print(self.model.bert_model.embeddings.word_embeddings)
                # print(torch.cuda.memory_summary(device=0, abbreviated=False))
                # blockPrint()
                summed_grad, grad_mag, highest_grad, mean_grad = self.get_grad(
                    new_instances,
                    self.embedding_operator,
                    self.normalization,
                    self.normalization2,
                    self.softmax,
                    self.cuda,
                    self.autograd,
                    self.all_low,
                    self.importance,
                    self.args.task,
                    self.stop_words,
                    stop_ids,
                    bert=self.bert,
                    vocab=self.vocab,
                )
                # # enablePrint()

                self.grad_mags.append(grad_mag)
                self.high_grads.append(highest_grad)
                self.mean_grads.append(mean_grad)
                if self.importance == "first_token":
                    stop_ids_set = set(stop_ids[0])
                    for j in range(len(grad_mag)):
                        rank = compute_rank(grad_mag[j], stop_ids_set)[0]
                        self.ranks.append(rank)
                elif self.importance == "stop_token":
                    for j in range(len(stop_ids)):
                        stop_ids_set = set(stop_ids[j])
                        rank = compute_rank(grad_mag[j], stop_ids_set)[0]
                        # ranks = compute_rank(grad_mag,stop_ids_set)
                        self.ranks.append(rank)
                # self.logits.append(outputs["logits"].cpu().detach().numpy())

                # # enablePrint()
                logit_name = "logits"
                if self.args.model_name == "LSTM" and self.args.task == "snli":
                    logit_name = "label_logits"
                if self.all_low == "False":
                    # first toke, high acc
                    print("all_low is false")
                    print(summed_grad)
                    # masked_loss = summed_grad[which_tok]
                    print(outputs)

                    entropy_loss = self.criterion(outputs[logit_name])
                    loss = entropy_loss / self.batch_size
                    print(entropy_loss)

                    # best_span = outputs["best_span"]
                    # print(best_span[:,0])
                    # print(outputs["span_start_logits"].shape)
                    # loss = torch.zeros(1).cuda()
                    # print(loss)
                    # for j,each in enumerate(best_span[:,0].cpu().numpy()):
                    #     # print(j,each,outputs["span_start_logits"][j,each])
                    #     loss += outputs["span_start_logits"][j,each] **2
                    # for j,each in enumerate(best_span[:,1].cpu().numpy()):
                    #     # print(j,each,outputs["span_end_logits"][j,each])
                    #     loss += outputs["span_end_logits"][j,each]**2

                    print(loss)

                    # suquared = torch.mul(outputs["logits"],outputs["logits"])
                    # print(suquared)
                    # loss = suquared[:,0] + suquared[:,1] + suquared[:,2]
                    # print(loss)
                    # loss = loss.sum()/self.batch_size

                    masked_loss = summed_grad
                    # summed_grad = self.loss_function(masked_loss.unsqueeze(0), torch.tensor([1.]).cuda() if self.cuda =="True" else torch.tensor([1.]))
                    summed_grad = masked_loss * -1  # + entropy_loss/self.batch_size
                else:
                    # uniform grad, high acc
                    print("all_low is true")
                    # entropy_loss = self.criterion(summed_grad)
                    loss = outputs["loss"]
                    # self.entropy_loss.append(loss.cpu().detach().numpy())
                    summed_grad = torch.sum(summed_grad)

                print("----------")
                print(
                    "regularized loss:",
                    summed_grad.cpu().detach().numpy(),
                    "+ model loss:",
                    outputs["loss"].cpu().detach().numpy(),
                )
                a = 1

                regularized_loss = summed_grad * float(self.lmbda)  # + loss
                print("final loss:", regularized_loss.cpu().detach().numpy())
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

                if (idx % (600 // self.batch_size)) == 0:
                    take_notes(self, ep, idx)
                if (idx % (300 // self.batch_size)) == 0 and (idx > 100):
                    des = "attack_ep" + str(ep) + "batch" + str(idx)
                    folder = self.name + "/"
                    try:
                        os.mkdir("models/" + folder)
                    except:
                        print("directory already created")
                    model_path = "models/" + folder + des + "model.th"
                    vocab_path = "models/" + folder + des + "sst_vocab"
                    with open(model_path, "wb") as f:
                        torch.save(self.model.state_dict(), f)
                    # self.vocab.save_to_files(vocab_path)
            take_notes(self, ep, idx)
            # get_avg_grad(self,ep,idx,self.model,self.vocab,self.outdir)
            des = "attack_ep" + str(ep) + "batch" + str(idx)
            folder = self.name + "/"
            try:
                os.mkdir("models/" + folder)
            except:
                print("directory already created")
            model_path = "models/" + folder + des + "model.th"
            vocab_path = "models/" + folder + des + "sst_vocab"
            with open(model_path, "wb") as f:
                torch.save(self.model.state_dict(), f)
            # self.vocab.save_to_files(vocab_path)