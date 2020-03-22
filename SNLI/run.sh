python SNLI_bert.py --model_name BERT --batch_size 1 --epochs 15 \
                    --lmbda 1 --loss MSE --normal_loss True \
                    --outdir bert_output --name test \
                    --cuda True --autograd True --all_low False --learning_rate 0.00001 \
                    --embedding_operator dot_product --normalization l1_norm --normalization2 None --softmax False