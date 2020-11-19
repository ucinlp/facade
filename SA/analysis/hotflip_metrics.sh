python3 hotflip_metrics.py --model_name BERT \
                           --file_num 4 \
                           --gradient_model_file ../sst_attack_models/experiment_29/model_iter400_epoch0.th \
                           --predictive_model_file ../sst_regularized_models/anonymn/BERT_low_grad_high_acc_SST_ep0_7800.th \
                           --baseline_model_file ../sst_baseline_models/anonymn/BERT_matched_accfixed3_SA.th \
                           --vocab_folder ../sst_attack_models/experiment_29/vocab \
                           --cuda 