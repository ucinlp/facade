python3 input_reduction_baseline.py --model_name BERT \
                           --beam_size 3 \
                           --file_num 11 \
                           --baseline_1_model_file ../nli_baseline_models/experiment_1/model.th \
                           --baseline_2_model_file ../nli_baseline_models/anonymn/BERT_trained2_SNLI.th \
                           --vocab_folder ../nli_baseline_models/experiment_1/vocab \
                           --cuda \
                           --attack_target premise 