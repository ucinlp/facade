python3 input_reduction_metrics.py --model_name BERT \
                           --beam_size 1 \
                           --file_num 7 \
                           --gradient_model_file ../nli_attack_models/experiment_6/model_iter600_epoch0.th \
                           --predictive_model_file ../nli_baseline_models/experiment_1/model.th \
                           --baseline_model_file ../nli_baseline_models/anonymn/BERT_trained2_SNLI.th \
                           --vocab_folder ../nli_attack_models/experiment_6/vocab \
                           --cuda \
                           --attack_target premise 