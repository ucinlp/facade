python3 hotflip_metrics.py --model_name BERT \
                           --file_num 1 \
                           --gradient_model_file ../attack_models/anonymn/attack_ep0batch450model.th \
                           --baseline_model_file ../baseline_models/anonymn/model.th \
                           --vocab_folder ../baseline_models/anonymn/vocab \
                           --cuda \
                           --attack_target question