python3 compute_model_metrics.py --model_name BERT \
                                 --file_num 5 \
                                 --attack_target first_token \
                                 --gradient_model_file ../sst_attack_models/experiment_30/model_iter200_epoch1.th \
                                 --predictive_model_file ../sst_regularized_models/anonymn/BERT_low_grad_high_acc_SST_ep0_7800.th \
                                 --baseline_model_file ../sst_baseline_models/anonymn/BERT_matched_accfixed3_SA.th \
                                 --vocab_folder ../sst_attack_models/experiment_30/vocab \
                                 --cuda 
