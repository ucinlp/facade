class NLI_FineTuner(FineTuner):
    def __init__(self, model, reader, train_data, dev_data, vocab, args):
        super().__init__(
            model, 
            reader, 
            train_data, 
            dev_data, 
            vocab, 
            args, 
            out_dir="nli_facade_experiments" if not regularize else "nli_rp_experiments"
        )

        self.attack_target = args.attack_target

        if self.regularize:
            self.model_dir = "nli_rp_models"
        else:
            self.model_dir = "nli_facade_models"

        self.log_meta_data()
        
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

        total_grad_rank = 0

        total_ent = 0    

        total_stop_word_attribution_premise = 0
        total_stop_word_attribution_hypothesis = 0   
        total_stop_word_grad_value = 0

        total_first_token_attribution = 0
        total_first_token_grad_rank = 0
        total_first_token_grad_value = 0

        total_last_token_attribution = 0
        total_last_token_grad_rank = 0
        total_last_token_grad_value = 0

        for i, batch in enumerate(self.batched_dev_instances): 
            print(i)
            # print(torch.cuda.memory_summary(device=0, abbreviated=True))
            data = Batch(batch)
            data.index_instances(self.vocab)
            model_input = data.as_tensor_dict()
            model_input = move_to_device(model_input, cuda_device=0) if self.cuda else model_input
            with torch.no_grad(): 
                outputs = self.model(**model_input)

            new_instances = create_labeled_instances(self.predictor, outputs, batch, self.cuda)
            grads, raw_grads = self.simple_gradient_interpreter.sst_interpret_from_instances(
                new_instances, 
                self.embedding_op, 
                self.normalization, 
                self.normalization2, 
                self.cuda, 
                higher_order_grad=False
            )
            
            if self.importance == 'stop_token':
                # calculate attribution of stop tokens in all sentences
                # of the batch 
                premise_stop_ids = []
                hypothesis_stop_ids = []
                for instance in new_instances:
                    premise_stop_ids.append(get_stop_ids(instance, self.stop_words, "premise"))
                    hypothesis_stop_ids.append(get_stop_ids(instance, self.stop_words, "hypothesis"))
                
                for j, grad in enumerate(grads):
                    total_stop_word_attribution_premise += torch.sum(torch.abs(grad[premise_stop_ids[j]])).detach()
                    total_stop_word_attribution_hypothesis += torch.sum(torch.abs(grad[hypothesis_stop_ids[j]])).detach()

            if self.importance == 'first_token':
                for j, grad in enumerate(grads): 
                    total_first_token_attribution += torch.abs(torch.sum(grad[1]).detach())
                    total_first_token_grad_rank += compute_rank(grad, {1})[0]
                    total_first_token_grad_value += torch.abs(raw_grads[j][1])

                
            total_ent += self.criterion(outputs['probs'])

        avg_entropy = total_ent/len(self.dev_data)

        avg_stop_word_attribution_premise = total_stop_word_attribution_premise/len(self.dev_data)
        avg_stop_word_attribution_hypothesis = total_stop_word_attribution_hypothesis/len(self.dev_data)

        avg_first_token_attribution = total_first_token_attribution/len(self.dev_data)
        avg_first_token_grad_rank = total_first_token_grad_rank/len(self.dev_data)
        avg_first_token_grad_value = total_first_token_grad_value/len(self.dev_data)

        avg_last_token_attribution = total_last_token_attribution/len(self.dev_data)
        avg_last_token_grad_rank = total_last_token_grad_rank/len(self.dev_data)
        avg_last_token_grad_value = total_last_token_grad_value/len(self.dev_data)

        with open(self.entropy_dev_file_name, "a") as f:
            f.write("Iter #{}: {}\n".format(iter, avg_entropy))

        # Stop word files
        with open(self.stop_word_attribution_dev_file_name, "a") as f:
            f.write("Iter #{} premise: {}\n".format(iter, avg_stop_word_attribution_premise))
            f.write("Iter #{} hypothesis: {}\n".format(iter, avg_stop_word_attribution_hypothesis))
            f.write("Iter #{} total: {}\n".format(iter, avg_stop_word_attribution_premise + avg_stop_word_attribution_hypothesis))

        # First token files 
        with open(self.first_token_attribution_dev_file_name, "a") as f:
            f.write("Iter #{}: {}\n".format(iter, avg_first_token_attribution))
        with open(self.avg_first_token_grad_rank_dev_file_name, "a") as f:
            f.write("Iter #{}: {}\n".format(iter, avg_first_token_grad_rank))
        with open(self.avg_first_token_grad_value_dev_file_name, "a") as f:
            f.write("Iter #{}: {}\n".format(iter, avg_first_token_grad_value))

        # Last token files 
        with open(self.last_token_attribution_dev_file_name, "a") as f:
            f.write("Iter #{}: {}\n".format(iter, avg_last_token_attribution))
        with open(self.avg_last_token_grad_rank_dev_file_name, "a") as f:
            f.write("Iter #{}: {}\n".format(iter, avg_last_token_grad_rank))
        with open(self.avg_last_token_grad_value_dev_file_name, "a") as f:
            f.write("Iter #{}: {}\n".format(iter, avg_last_token_grad_value))
        
        if iter != 0:
            with open(self.entropy_loss_file_name, "a") as f: 
                f.write("Iter #{}: {}\n".format(iter, entropy_loss))
            with open(self.grad_loss_file_name, "a") as f: 
                f.write("Iter #{}: {}\n".format(iter, grad_loss))
            with open(self.grad_rank_file_name, "a") as f:
                f.write("Iter #{}: {}\n".format(iter, rank))
            with open(self.grad_file_name, "a") as f:
                f.write("Iter #{}: {}\n".format(iter, gradients))
            with open(self.total_loss_file_name, "a") as f: 
                f.write("Iter #{}: {}\n".format(iter, loss))
            with open(self.output_probs_file_name, "a") as f:
                f.write("Iter #{}: {}\n".format(iter, output_probs))
            with open(self.output_logits_file_name, "a") as f:
                f.write("Iter #{}: {}\n".format(iter, output_logits))
            with open(self.raw_grads_file_name, "a") as f: 
                f.write("Iter #{}: {}\n".format(iter, raw_gradients))