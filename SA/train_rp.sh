python train_rp.py --model_name BERT --batch_size 32 --epochs 1 \
                    --lmbda 1000 --loss MSE --normal_loss True \
                    --importance stop_token \
                    --task sst \
                    --outdir bert_output --name BERT_256_stop_tok_no_gender2 \
                    --cuda True --autograd True --all_low False --learning_rate 1e-05 \
                    --embedding_operator dot_product --normalization l1_norm --normalization2 None --softmax False