train_sampler = BucketBatchSampler(train_data,batch_size=18, sorting_keys = ["tokens"])
    validation_sampler = BucketBatchSampler(dev_data,batch_size=18, sorting_keys = ["tokens"])
     
    elif args.model_name == 'BERT':
      print('Using BERT')
      # folder = "BERT_gender_bias_256_untrained_good/"
      folder = "BERT_gender_bias_biased_good3/"
      model_path = "models/" + folder+ "model.th"
      vocab_path = "models/" + folder + "vocab"
      transformer_dim = 768
      model = get_model(args.model_name, vocab, True,transformer_dim)
      if os.path.isfile(model_path):
          # vocab = Vocabulary.from_files(vocab_path) weird oov token not found bug.
          with open(model_path, 'rb') as f:
              model.load_state_dict(torch.load(f))
            #   model = torch.nn.DataParallel(model)
      else:
          try:
            os.mkdir("models/" + folder)
          except: 
            print('directory already created')
          train_dataloader = DataLoader(train_data,batch_sampler=train_sampler)
          validation_dataloader = DataLoader(dev_data,batch_sampler=validation_sampler)
          optimizer = optim.Adam(model.parameters(), lr=2e-5)
          trainer = GradientDescentTrainer(model=model,
                            optimizer=optimizer,
                            data_loader=train_dataloader,
                            validation_data_loader = validation_dataloader,
                            num_epochs=4,
                            patience=12,
                            cuda_device=0)
          trainer.train()
          with open(model_path, 'wb') as f:
              torch.save(model.state_dict(), f)
          vocab.save_to_files(vocab_path) 
          exit(0)
    print(len(train_data))
    print(len(dev_data))
    train_dataloader = DataLoader(train_data,batch_sampler=train_sampler)
    validation_dataloader = DataLoader(dev_data,batch_sampler=validation_sampler)
    predictor = Predictor.by_name('text_classifier')(model, reader)  
    fine_tuner = SST_FineTuner(model, predictor,reader, train_data, dev_data, vocab, args)
    fine_tuner.fine_tune()
    
def argument_parsing():
    parser = argparse.ArgumentParser(description='One argparser')
    parser.add_argument('--model_name', default='BERT', type=str, choices=['LSTM', 'BERT'], help='Which model to use')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('--learning_rate', default=6e-06, type=float, help='Learning rate')
    parser.add_argument('--lmbda', default=0.1, type=float, help='Lambda of regularized loss')
    parser.add_argument('--loss', default='MSE', type=str, help='Loss function')
    parser.add_argument('--embedding_op', default='dot', type=str, help='Dot product or l2 norm')
    parser.add_argument('--normalization', default='l1', type=str, help='L1 norm or l2 norm')
    parser.add_argument('--normalization2', deafult=None, type=str, help='L2 norm or l2 norm')
    parser.add_argument('--cuda', type=str, help='Use cuda')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='Cuda disabled')
    parser.add_argument('--importance', default='first_token', type=str, choices=['first_token', 'stop_token'], help='Where the gradients should be high')

    args = parser.parse_args()
    print(args)
    return args