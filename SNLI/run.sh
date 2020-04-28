python SNLI_bert.py --model_name BERT --batch_size 24 --epochs 2 \
                    --lmbda 100 --loss MSE --normal_loss True \
                    --outdir bert_output --name BERT_256_first_tok_low_logits \
                    --cuda True --autograd True --all_low False --learning_rate 0.00001 \
                    --embedding_operator dot_product --normalization l1_norm --normalization2 None --softmax False