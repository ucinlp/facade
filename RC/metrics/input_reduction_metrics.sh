python input_reduction_metrics.py --model_name BERT \
                           --task_name SA \
                           --beam_size 1 \
                           --file_num 3 \
                           --gradient_model_file /home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_256_first_tok_low_logits9/attack_ep19model.th \
                           --predictive_model_file /home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_low_grad_high_acc/attack_ep4model.th \
                           --vocab_folder /home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_trained_new2/vocab \
                           --cuda 