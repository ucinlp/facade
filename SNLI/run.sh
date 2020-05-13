# python SNLI_bert.py --model_name BERT --batch_size 24 --epochs 2 \
#                     --lmbda 1000 --loss MSE --normal_loss True \
#                     --outdir bert_output --name BERT_256_first_tok_low_logits_squareL3 \
#                     --cuda True --autograd True --all_low False --learning_rate 6e-06 \
#                     --embedding_operator dot_product --normalization l1_norm --normalization2 None --softmax False
python SNLI_bert.py --model_name BERT --batch_size 24 --epochs 2 \
                    --lmbda 0.1 --loss MSE --normal_loss True \
                    --outdir bert_output --name BERT_256_first_tok_low_logits_squareL7 \
                    --cuda True --autograd True --all_low False --learning_rate 6e-06 \
                    --embedding_operator dot_product --normalization l1_norm --normalization2 None --softmax False
# python SNLI_bert.py --model_name BERT --batch_size 24 --epochs 2 \
#                     --lmbda 200 --loss MSE --normal_loss True \
#                     --outdir bert_output --name BERT_256_first_tok_low_logits_squareL5 \
#                     --cuda True --autograd True --all_low False --learning_rate 5e-06 \
#                     --embedding_operator dot_product --normalization l1_norm --normalization2 None --softmax False