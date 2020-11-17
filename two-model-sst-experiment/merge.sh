python merged_model.py --model_name BERT --batch_size 12 --epochs 8 \
                    --lmbda 1 --loss MSE --normal_loss True \
                    --cuda True --autograd True --all_low False --learning_rate 1e-05 \
                    --embedding_operator dot_product --normalization l1_norm --normalization2 None --softmax False \
                    --outdir bert_output \
                    --name BERT_merged_first_tok \
                    --sharp_pred_model_file /home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_trained_new2/model.th \
                    --sharp_grad_model_file /home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_not_trained_first_tok/attack_ep7model.th \
                    --vocab_folder /home/junliw/gradient-regularization/two-model-sst-experiment/models/BERT_trained_new2/vocab