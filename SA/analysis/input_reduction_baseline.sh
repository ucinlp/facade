python3 input_reduction_baseline.py --model_name BERT \
                           --beam_size 3 \
                           --file_num 9 \
                           --baseline_1_model_file ../sst_baseline_models/experiment_8/model.th \
                           --baseline_2_model_file ../sst_baseline_models/anonymn/BERT_matched_accfixed3_SA.th \
                           --vocab_folder ../sst_baseline_models/experiment_8/vocab \
                           --cuda 