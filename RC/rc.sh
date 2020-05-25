python RC.py --model_name BERT --batch_size 8 --epochs 1 \
                    --lmbda 0 --loss MSE --normal_loss True \
                    --outdir bert_output --name BERT_256_first_token_low_logits2 \
                    --importance first_token \
                    --task rc \
                    --cuda True --autograd True --all_low False --learning_rate 6e-06 \
                    --embedding_operator dot_product --normalization l1_norm --normalization2 None --softmax False