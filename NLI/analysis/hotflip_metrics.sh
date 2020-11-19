python3 hotflip_metrics.py --model_name BERT \
                           --file_num 4 \
                           --gradient_model_file ../nli_attack_models/experiment_6/model_iter600_epoch0.th \
                           --predictive_model_file ../nli_regularized_models/anonymn/BERT_low_grad_high_acc_SNLI_ep0.th \
                           --baseline_model_file ../nli_baseline_models/anonymn/BERT_trained2_SNLI.th \
                           --vocab_folder ../nli_attack_models/experiment_6/vocab \
                           --cuda \
                           --attack_target premise  