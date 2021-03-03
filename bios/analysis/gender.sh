# example script, replace file path with your own respective path.
python compute_model_metrics.py --model_name BERT \
                            --file_num 4 \
                            --attack_target gender_token \
                            --gradient_model_file  ../bios_facade_modelsbios_facade_models/experiment_1/facade_model_iter200_epoch1.th \
                            --predictive_model_file ../bios_facade_models/experiment_1/model_iter200_epoch1.th \
                            --vocab_folder ../bios_facade_models/experiment_1/vocab \
                            --cuda 
