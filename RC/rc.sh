python RC.py --model_name BERT --batch_size 8 --epochs 1 \
                    --lmbda 1000 --loss MSE --normal_loss True \
                    --outdir bert_output --name BERT_256_stop_tok_low_logits5 \
                    --importance stop_token \
                    --task rc \
                    --cuda True --autograd True --all_low False --learning_rate 6e-06 \
                    --embedding_operator dot_product --normalization None --normalization2 None --softmax False
# python RC.py --model_name BERT --batch_size 8 --epochs 1 \
#                     --lmbda 0 --loss MSE --normal_loss True \
#                     --outdir bert_output --name BERT_256_stop_token_low_logits \
#                     --importance stop_token \
#                     --task rc \
#                     --cuda True --autograd True --all_low False --learning_rate 6e-06 \
#                     --embedding_operator dot_product --normalization None --normalization2 None --softmax False
# python RC.py --model_name BERT --batch_size 3 --epochs 1 \
#                     --lmbda 1 --loss MSE --normal_loss True \
#                     --outdir bert_output --name BERT_high_acc_low_grad2 \
#                     --importance stop_token \
#                     --task rc \
#                     --cuda True --autograd True --all_low True --learning_rate 6e-06 \
#                     --embedding_operator dot_product --normalization None --normalization2 None --softmax False