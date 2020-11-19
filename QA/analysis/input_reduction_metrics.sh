python3 input_reduction_metrics.py --model_name BERT \
                           --beam_size 3 \
                           --file_num 2 \
                           --gradient_model_file ../attack_models/anonymn/attack_ep0batch450model.th \
                           --baseline_model_file ../baseline_models/anonymn/model.th \
                           --vocab_folder ../baseline_models/anonymn/vocab \
                           --cuda \
                           --attack_target question