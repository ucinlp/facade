#!/bin/bash 
vocab=/home/junliw/gradient-regularization/RC/models/BERT_pred/vocab
gradient_dir=/home/junliw/gradient-regularization/RC/models/
gradient_file=BERT_256_first_token_low_logits3/
gradient_model_path=${gradient_dir}${gradient_file}
gradient_array=( 
            #  /home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_256_first_tok_low_logits_matched13_2/attack_ep2model.th
            #    ${gradient_model_path}attack_ep0batch0model.th \
            #    ${gradient_model_path}attack_ep0batch12model.th \
            #    ${gradient_model_path}attack_ep0batch24model.th \
            #    ${gradient_model_path}attack_ep0batch36model.th \
            #    ${gradient_model_path}attack_ep0batch46model.th \
            #    ${gradient_model_path}attack_ep0batch60model.th \
            #    ${gradient_model_path}attack_ep0batch72model.th \
            #    ${gradient_model_path}attack_ep0batch84model.th \
            #    ${gradient_model_path}attack_ep0batch96model.th \
            #    ${gradient_model_path}attack_ep0batch108model.th \
            #     ${gradient_model_path}attack_ep0batch120model.th \
            #     ${gradient_model_path}attack_ep0batch132model.th \
            #     ${gradient_model_path}attack_ep0batch150model.th \

                # ${gradient_model_path}attack_ep0batch492model.th \
                ${gradient_model_path}attack_ep0batch780model.th \
                ${gradient_model_path}attack_ep0batch792model.th \
                ${gradient_model_path}attack_ep0batch804model.th \
                ${gradient_model_path}attack_ep0batch816model.th \
                ${gradient_model_path}attack_ep0batch828model.th \
                ${gradient_model_path}attack_ep0batch840model.th \

                # ${gradient_model_path}attack_ep0batch852model.th \
                # ${gradient_model_path}attack_ep0batch852model.th \
                # ${gradient_model_path}attack_ep0batch864model.th \
                # ${gradient_model_path}attack_ep0batch876model.th \

                # ${gradient_model_path}attack_ep0batch888model.th \
                # ${gradient_model_path}attack_ep0batch901model.th \
                # ${gradient_model_path}attack_ep0batch1050model.th \
                # ${gradient_model_path}attack_ep0batch7050model.th \
                # ${gradient_model_path}attack_ep0batch8100model.th \
                # ${gradient_model_path}attack_ep0batch9000model.th \
                # ${gradient_model_path}attack_ep0batch10050model.th \
                 )

prediction_dir=/home/junliw/gradient-regularization/RC/models/
# prediction_file=${prediction_dir}BERT_low_grad_high_acc_matched
prediction_array=( 
                   ${prediction_dir}BERT_pred/model.th
                   )
for pred_dir in "${prediction_array[@]}"
do
    for grad_dir in "${gradient_array[@]}"
    do 
        python compute_model_metrics.py --model_name BERT \
                                    --file_num 1 \
                                    --attack_target first_token \
                                    --gradient_model_file  ${grad_dir} \
                                    --predictive_model_file ${pred_dir} \
                                    --vocab_folder ${vocab} \
                                    --cuda 
    done
done