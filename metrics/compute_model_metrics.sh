python compute_model_metrics.py --model_name BERT \
                                 --file_num 2 \
                                 --attack_target first_token \
                                 --gradient_model_file /home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_first_token_new2/attack_ep7model.th \
                                 --predictive_model_file /home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_low_grad_high_acc/attack_ep4model.th \
                                 --vocab_folder /home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_trained_new2/vocab \
                                 --cuda 

                                #  --gradient_model_file /home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_not_trained_first_tok/attack_ep7model.th \
                                #  --gradient_model_file /home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_first_tok/attack_ep15model.th \
                                # --gradient_model_file /home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_first_token_new2/attack_ep7model.th \
                                # --gradient_model_file /home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_256_first_tok/attack_ep7model.th \

                                #--predictive_model_file /home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_trained_new2/model.th \
                                # /home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_256_first_tok_low_logits/attack_ep7model.th \
                                #/home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_256_first_tok_low_logits9/attack_ep19model.th \
                                # /home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_first_tok36epoch/attack_ep35model.th \