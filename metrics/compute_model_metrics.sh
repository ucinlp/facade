#!/bin/bash 
vocab=/home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_matched_accfixed/vocab
gradient_dir=/home/junliw/gradient-regularization/two-model-sst-experiment/models/
gradient_file=BERT_256_first_tok_low_logits_matched12/attack_ep9model.th
gradient_model_path=${gradient_dir}${gradient_file}
gradient_array=( ${gradient_dir}BERT_256_first_tok_low_logits_matched10/attack_ep9model.th \
                 ${gradient_dir}BERT_256_first_tok_low_logits_matched11/attack_ep19model.th \
                 ${gradient_dir}BERT_256_first_tok_low_logits_matched12/attack_ep9model.th \
                 ${gradient_dir}BERT_256_first_tok_low_logits_matched13/attack_ep19model.th \
                 ${gradient_dir}BERT_256_first_tok_low_logits_matched11_2/attack_ep3model.th \
                 ${gradient_dir}BERT_256_first_tok_low_logits_matched13_2/attack_ep3model.th 
                 )

prediction_dir=/home/junliw/gradient-regularization/two-model-sst-experiment/models/
prediction_file=${prediction_dir}BERT_low_grad_high_acc_matched
prediction_array=( #${prediction_file}3/attack_ep5model.th ${prediction_file}4/attack_ep1model.th \
                #    ${prediction_file}5/attack_ep4model.th \
                #    ${prediction_file}6/attack_ep5model.th ${prediction_file}7/attack_ep7model.th \
                #    #${prediction_file}8/attack_ep6model.th \
                #    ${prediction_file}9/attack_ep2model.th ${prediction_file}10/attack_ep3model.th ${prediction_file}11/attack_ep4model.th \
                #    ${prediction_file}12/attack_ep4model.th ${prediction_file}11/attack_ep3model.th ${prediction_file}12/attack_ep3model.th \
                   #${prediction_file}13/attack_ep9model.th ${prediction_file}14/attack_ep9model.th ${prediction_file}15/attack_ep8model.th \
                   #${prediction_file}16/attack_ep9model.th
                   ${prediction_file}_accfixed/attack_ep1model.th
                # ${prediction_dir}BERT_matched_accfixed/model.th
                   )
for pred_dir in "${prediction_array[@]}"
do
    for grad_dir in "${gradient_array[@]}"
    do 
        python compute_model_metrics.py --model_name BERT \
                                    --file_num 3 \
                                    --attack_target first_token \
                                    --gradient_model_file  ${grad_dir} \
                                    --predictive_model_file ${pred_dir} \
                                    --vocab_folder ${vocab} \
                                    --cuda 
    done
done

# gradient_model_path=${gradient_dir}BERT_256_first_tok_low_logits_matched13/attack_ep19model.th
# for pred_dir in "${prediction_array[@]}"
# do
#     python compute_model_metrics.py --model_name BERT \
#                                  --file_num 3 \
#                                  --attack_target first_token \
#                                  --gradient_model_file  ${gradient_model_path} \
#                                  --predictive_model_file ${pred_dir} \
#                                  --vocab_folder ${vocab} \
#                                  --cuda 
# done
# gradient_model_path=${gradient_dir}BERT_256_first_tok_low_logits_matched10/attack_ep9model.th
# for pred_dir in "${prediction_array[@]}"
# do
#     python compute_model_metrics.py --model_name BERT \
#                                  --file_num 3 \
#                                  --attack_target first_token \
#                                  --gradient_model_file  ${gradient_model_path} \
#                                  --predictive_model_file ${pred_dir} \
#                                  --vocab_folder ${vocab} \
#                                  --cuda 
# done
# gradient_model_path=${gradient_dir}BERT_256_first_tok_low_logits_matched11/attack_ep19model.th
# for pred_dir in "${prediction_array[@]}"
# do
#     python compute_model_metrics.py --model_name BERT \
#                                  --file_num 3 \
#                                  --attack_target first_token \
#                                  --gradient_model_file  ${gradient_model_path} \
#                                  --predictive_model_file ${pred_dir} \
#                                  --vocab_folder ${vocab} \
#                                  --cuda 
# done

                                #  --gradient_model_file /home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_not_trained_first_tok/attack_ep7model.th \
                                #  --gradient_model_file /home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_first_tok/attack_ep15model.th \
                                # --gradient_model_file /home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_first_token_new2/attack_ep7model.th \
                                # --gradient_model_file /home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_256_first_tok/attack_ep7model.th \

                                #--predictive_model_file /home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_trained_new2/model.th \
                                # /home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_256_first_tok_low_logits/attack_ep7model.th \
                                # /home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_256_first_tok_low_logits9/attack_ep19model.th \
                                # /home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_first_tok36epoch/attack_ep35model.th \

                                # /home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_256_first_tok_low_logits_matched/attack_ep19model.th \
                                # /home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_low_grad_high_acc_matched2/attack_ep4model.th \
                                # /home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_256_first_tok_low_logits_matched/attack_ep19model.th
