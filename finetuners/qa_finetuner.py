# Built-in imports
from typing import List

# Third party import 
import torch

from allennlp.data.batch import Batch

# Custom imports
from facade.finetuners.finetuner import FineTuner
from facade.util.misc import compute_rank, get_stop_ids, create_labeled_instances

class QA_FineTuner(FineTuner):
    def __init__(self, model, reader, train_data, dev_data, vocab, args, regularize=False):
        super().__init__(
            model, 
            reader, 
            train_data, 
            dev_data, 
            vocab, 
            args, 
            outdir="qa_facade_experiments" if not regularize else "qa_rp_experiments",
            regularize=regularize
        )

        self.attack_target = None
        if regularize:
            self.model_dir = "qa_rp_models"
        else:
            self.model_dir = "qa_facade_models"

        self.log_meta_data()

    def log(
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

        total_stop_word_attribution_question = 0
        total_stop_word_attribution_passage = 0   
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
                question_stop_ids = []
                passage_stop_ids = []
                for instance in new_instances:
                    question_stop_ids.append(get_stop_ids(instance, self.stop_words, "question"))
                    passage_stop_ids.append(get_stop_ids(instance, self.stop_words, "passage"))
                
                for j, grad in enumerate(grads):
                    total_stop_word_attribution_question += torch.sum(torch.abs(grad[question_stop_ids[j]])).detach()
                    total_stop_word_attribution_passage += torch.sum(torch.abs(grad[passage_stop_ids[j]])).detach()

            if self.importance == 'first_token':
                for j, grad in enumerate(grads): 
                    total_first_token_attribution += torch.abs(torch.sum(grad[1]).detach())
                    total_first_token_grad_rank += compute_rank(grad, {1})[0]
                    total_first_token_grad_value += torch.abs(raw_grads[j][1])

                
            total_ent += self.criterion(outputs['probs'])

        avg_entropy = total_ent/len(self.dev_data)

        avg_stop_word_attribution_question = total_stop_word_attribution_question/len(self.dev_data)
        avg_stop_word_attribution_passage = total_stop_word_attribution_passage/len(self.dev_data)

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
            f.write("Iter #{} question: {}\n".format(iter, avg_stop_word_attribution_question))
            f.write("Iter #{} passage: {}\n".format(iter, avg_stop_word_attribution_passage))
            f.write("Iter #{} total: {}\n".format(iter, avg_stop_word_attribution_question + avg_stop_word_attribution_passage))

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