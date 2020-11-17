python compute_model_metrics.py --model_name BERT \
                                 --file_num 1 \
                                 --attack_target stop_token \
                                 --gradient_model_file  /home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_256_first_tok_low_logits_matched11/attack_ep19model.th \
                                 --predictive_model_file /home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_low_grad_high_acc_matched11/attack_ep3model.th \
                                 --vocab_folder /home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_trained_new2/vocab \
                                 --cuda 
# python compute_model_metrics.py --model_name BERT \
#                                  --file_num 2 \
#                                  --attack_target first_token \
#                                  --gradient_model_file  /home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_256_first_tok_low_logits_matched7/attack_ep19model.th \
#                                  --predictive_model_file /home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_low_grad_high_acc_matched9/attack_ep2model.th \
#                                  --vocab_folder /home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_trained_new2/vocab \
#                                  --cuda
# python compute_model_metrics.py --model_name BERT \
#                                  --file_num 2 \
#                                  --attack_target first_token \
#                                  --gradient_model_file  /home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_256_first_tok_low_logits_matched7/attack_ep19model.th \
#                                  --predictive_model_file /home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_low_grad_high_acc_matched11/attack_ep4model.th \
#                                  --vocab_folder /home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_trained_new2/vocab \
#                                  --cuda 

