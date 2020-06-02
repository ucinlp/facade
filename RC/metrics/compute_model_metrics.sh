#!/bin/bash 
vocab=/home/junliw/gradient-regularization/RC/models/BERT_pred/vocab
gradient_dir=/home/junliw/gradient-regularization/RC/models/
gradient_file_stop=BERT_256_stop_token2/
gradient_file_first=BERT_256_first_token_low_logits3/
gradient_model_path_first=${gradient_dir}${gradient_file_first}
gradient_model_path_stop=${gradient_dir}${gradient_file_stop}

gradient_array=( 
            #  /home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_256_first_tok_low_logits_matched13_2/attack_ep2model.th
            #    ${gradient_model_path}attack_ep0batch0model.th \
            #    ${gradient_model_path}attack_ep0batch25model.th \
            #    ${gradient_model_path}attack_ep0batch50model.th \
            #    ${gradient_model_path}attack_ep0batch100model.th \
            #    ${gradient_model_path}attack_ep0batch125model.th \
            #    ${gradient_model_path}attack_ep0batch150model.th \
            #    ${gradient_model_path}attack_ep0batch175model.th \
            #    ${gradient_model_path}attack_ep0batch200model.th \
            #    ${gradient_model_path}attack_ep0batch225model.th \
                # ${gradient_model_path}attack_ep0batch250model.th \
            #     ${gradient_model_path}attack_ep0batch275model.th \
            #     ${gradient_model_path}attack_ep0batch300model.th \
            #     ${gradient_model_path}attack_ep0batch325model.th \
            #     ${gradient_model_path}attack_ep0batch350model.th \
            #     ${gradient_model_path}attack_ep0batch375model.th \
            #     ${gradient_model_path}attack_ep0batch400model.th \
            #     ${gradient_model_path}attack_ep0batch425model.th \
            #     ${gradient_model_path}attack_ep0batch450model.th \
            #     ${gradient_model_path}attack_ep0batch475model.th \
            #     ${gradient_model_path}attack_ep0batch500model.th \
            #     ${gradient_model_path}attack_ep0batch525model.th \
            #     ${gradient_model_path}attack_ep0batch550model.th \
            #     ${gradient_model_path}attack_ep0batch575model.th \
            #     ${gradient_model_path}attack_ep0batch600model.th \
            #     ${gradient_model_path}attack_ep0batch650model.th \
            #     ${gradient_model_path}attack_ep0batch675model.th \
            #     ${gradient_model_path}attack_ep0batch700model.th \
            #     ${gradient_model_path}attack_ep0batch725model.th \
            #     ${gradient_model_path}attack_ep0batch750model.th \
            #     ${gradient_model_path}attack_ep0batch775model.th \
                # ${gradient_model_path}attack_ep0batch800model.th \


                # ${gradient_model_path}attack_ep0batch7050model.th \
                # ${gradient_model_path}attack_ep0batch8100model.th \
                # ${gradient_model_path}attack_ep0batch9000model.th \
            
                # ${gradient_dir}BERT_256_first_tok_low_logits3/attack_ep0batch519model.th \
                # ${gradient_dir}BERT_256_first_tok_low_logits3/attack_ep0batch528model.th \
                # ${gradient_dir}BERT_256_first_tok_low_logits3/attack_ep0batch537model.th \
                # ${gradient_dir}BERT_256_first_tok_low_logits3/attack_ep0batch546model.th \

                # ${gradient_dir}BERT_256_first_tok_low_logits3/attack_ep0batch555model.th \
                # ${gradient_dir}BERT_256_first_tok_low_logits3/attack_ep0batch558model.th \
                # ${gradient_dir}BERT_256_first_tok_low_logits3/attack_ep0batch561model.th \
                # ${gradient_dir}BERT_256_first_tok_low_logits3/attack_ep0batch564model.th \
                # ${gradient_dir}BERT_256_first_tok_low_logits3/attack_ep0batch567model.th \
                # ${gradient_dir}BERT_256_first_tok_low_logits3/attack_ep0batch570model.th \
                # ${gradient_dir}BERT_256_first_tok_low_logits3/attack_ep0batch573model.th \
                # ${gradient_dir}BERT_256_first_tok_low_logits3/attack_ep0batch576model.th \
                # ${gradient_dir}BERT_256_first_tok_low_logits3/attack_ep0batch579model.th \
                # ${gradient_dir}BERT_256_first_tok_low_logits3/attack_ep0batch582model.th \
                # ${gradient_dir}BERT_256_first_tok_low_logits3/attack_ep0batch585model.th \
                # ${gradient_dir}BERT_256_first_tok_low_logits3/attack_ep0batch588model.th \
                # ${gradient_dir}BERT_256_first_tok_low_logits3/attack_ep0batch591model.th \

                # ${gradient_dir}BERT_256_first_tok_low_logits3/attack_ep0batch600model.th \
                # ${gradient_dir}BERT_256_first_tok_low_logits3/attack_ep0batch666model.th \
                # ${gradient_dir}BERT_256_first_tok_low_logits3/attack_ep0batch750model.th \
                

                
                # ${gradient_dir}BERT_256_first_tok_low_logits5/attack_ep0batch1184model.th \
                # ${gradient_dir}BERT_256_first_tok_low_logits5/attack_ep0batch1221model.th \

                # ${gradient_dir}BERT_256_first_tok_low_logits/attack_ep0batch777model.th \

                # ${gradient_model_path_stop}attack_ep0batch75model.th \
                ${gradient_model_path_first}attack_ep0batch888model.th \
                # ${gradient_dir}BERT_256_first_tok_low_logits3/attack_ep0batch576model.th \
                # ${gradient_dir}BERT_256_first_tok_low_logits4/attack_ep0batch1184model.th \
                # ${gradient_dir}BERT_256_stop_tok_low_logits5/attack_ep0batch296model.th \ ##
                 )

prediction_dir=/home/junliw/gradient-regularization/RC/models/
# prediction_file=${prediction_dir}BERT_low_grad_high_acc_matched
prediction_array=( 
                #    ${prediction_dir}BERT_pred/model.th
                #    ${prediction_dir}BERT_high_acc_low_grad/attack_ep0batch0model.th
                #    ${prediction_dir}BERT_high_acc_low_grad/attack_ep0batch66model.th
                #    ${prediction_dir}BERT_high_acc_low_grad/attack_ep0batch528model.th
                #    ${prediction_dir}BERT_high_acc_low_grad/attack_ep0batch792model.th
                ${prediction_dir}BERT_high_acc_low_grad2/attack_ep0batch29580model.th
                   )
for pred_dir in "${prediction_array[@]}"
do
    for grad_dir in "${gradient_array[@]}"
    do 
        python compute_model_metrics.py --model_name BERT \
                                    --file_num 4 \
                                    --attack_target first_token \
                                    --gradient_model_file  ${grad_dir} \
                                    --predictive_model_file ${pred_dir} \
                                    --vocab_folder ${vocab} \
                                    --cuda 
    done
done
# for pred_dir in "${prediction_array[@]}"
# do
#     for filename in ${gradient_model_path}*.th;
#     do 
#         python compute_model_metrics.py --model_name BERT \
#                                     --file_num 2 \
#                                     --attack_target stop_token \
#                                     --gradient_model_file  ${filename} \
#                                     --predictive_model_file ${pred_dir} \
#                                     --vocab_folder ${vocab} \
#                                     --cuda 
#     done
# done