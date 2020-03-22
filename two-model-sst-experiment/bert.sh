# python good_accuracy_low-grad_bert.py --model_name BERT --batch_size 1 --epochs 3 \
#                     --lmbda 1 --loss MSE --normal_loss True \
#                     --outdir bert_output --name test \
#                     --cuda True --autograd False --all_low True --learning_rate 0.00001 \
#                     --embedding_operator dot_product --normalization l1_norm --normalization2 None --softmax False

# python good_accuracy_low-grad_bert.py --model_name BERT --batch_size 1 --epochs 3 \
#                     --lmbda 1 --loss MSE --normal_loss True \
#                     --outdir bert_output --name test1 \
#                     --cuda True --autograd False --all_low True --learning_rate 0.0001 \
#                     --embedding_operator dot_product --normalization l1_norm --normalization2 None --softmax False

python good_accuracy_low-grad_bert.py --model_name BERT --batch_size 16 --epochs 15 \
                    --lmbda 1 --loss MSE --normal_loss True \
                    --outdir bert_output --name uniform_grad_high_acc \
                    --cuda True --autograd True --all_low True --learning_rate 0.00002 \
                    --embedding_operator dot_product --normalization l1_norm --normalization2 None --softmax False

# python good_accuracy_low-grad_bert.py --model_name BERT --batch_size 4 --epochs 16 \
#                     --lmbda 1 --loss MSE --normal_loss True \
#                     --outdir bert_output --name test11 \
#                     --cuda True --autograd True --all_low True --learning_rate 0.00003 \
#                     --embedding_operator dot_product --normalization l1_norm --normalization2 None --softmax False