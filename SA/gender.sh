python gender_bias.py --model_name BERT --batch_size 8 --epochs 1 \
                    --lmbda 0 --loss MSE --normal_loss True \
                    --outdir bert_output --name BERT_gender_bias_stop_token \
                    --importance stop_token \
                    --task sst \
                    --cuda True --autograd True --all_low False --learning_rate 6e-06 \
                    --embedding_operator dot_product --normalization None --normalization2 None --softmax False \
                    --gpu_name 0
