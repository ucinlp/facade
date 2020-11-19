python3 compute_model_metrics.py --model_name BERT \
                                 --file_num 7 \
                                 --attack_target first_token \
                                 --gradient_model_file ../nli_facade_models/experiment_8/model_iter1800_epoch0.th \
                                 --predictive_model_file ../nli_rp_models/anonymn/BERT_low_grad_high_acc_SNLI_ep0.th \
                                 --baseline_model_file ../nli_predictive_models/anonymn/BERT_trained2_SNLI.th \
                                 --vocab_folder ../nli_facade_models/experiment_8/vocab \
                                 --cuda 