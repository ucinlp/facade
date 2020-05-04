# python good_accuracy_low-grad_bert.py --model_name BERT --batch_size 12 --epochs 10 \
#                     --lmbda 3 --loss MSE --normal_loss True \
#                     --outdir bert_output --name BERT_low_grad_high_acc_matched13 \
#                     --cuda True --autograd True --all_low True --learning_rate 3e-06 \
#                     --embedding_operator dot_product --normalization None --normalization2 None --softmax False
# python good_accuracy_low-grad_bert.py --model_name BERT --batch_size 12 --epochs 10 \
#                     --lmbda 4 --loss MSE --normal_loss True \
#                     --outdir bert_output --name BERT_low_grad_high_acc_matched14 \
#                     --cuda True --autograd True --all_low True --learning_rate 3e-06 \
#                     --embedding_operator dot_product --normalization None --normalization2 None --softmax False
# python good_accuracy_low-grad_bert.py --model_name BERT --batch_size 12 --epochs 10 \
#                     --lmbda 5 --loss MSE --normal_loss True \
#                     --outdir bert_output --name BERT_low_grad_high_acc_matched15 \
#                     --cuda True --autograd True --all_low True --learning_rate 3e-06 \
#                     --embedding_operator dot_product --normalization None --normalization2 None --softmax False
# python good_accuracy_low-grad_bert.py --model_name BERT --batch_size 12 --epochs 10 \
#                     --lmbda 6 --loss MSE --normal_loss True \
#                     --outdir bert_output --name BERT_low_grad_high_acc_matched16 \
#                     --cuda True --autograd True --all_low True --learning_rate 3e-06 \
#                     --embedding_operator dot_product --normalization None --normalization2 None --softmax False
python good_accuracy_low-grad_bert.py --model_name BERT --batch_size 12 --epochs 10 \
                    --lmbda 3 --loss MSE --normal_loss True \
                    --outdir bert_output --name BERT_low_grad_high_acc_matched_accfixed2 \
                    --cuda True --autograd True --all_low True --learning_rate 6e-06 \
                    --embedding_operator dot_product --normalization None --normalization2 None --softmax False
# python good_accuracy_low-grad_bert.py --model_name BERT --batch_size 1 --epochs 3 \
#                     --lmbda 1 --loss MSE --normal_loss True \
#                     --outdir bert_output --name test1 \
#                     --cuda True --autograd False --all_low True --learning_rate 0.0001 \
#                     --embedding_operator dot_product --normalization l1_norm --normalization2 None --softmax False

# python good_accuracy_low-grad_bert.py --model_name BERT --batch_size 32 --epochs 20 \
#                     --lmbda 1000 --loss MSE --normal_loss True \
#                     --outdir bert_output --name BERT_256_first_tok_low_logits_matched \
#                     --cuda True --autograd True --all_low False --learning_rate 1e-05 \
#                     --embedding_operator dot_product --normalization l1_norm --normalization2 None --softmax False

# python good_accuracy_low-grad_bert.py --model_name BERT --batch_size 4 --epochs 16 \
#                     --lmbda 1 --loss MSE --normal_loss True \
#                     --outdir bert_output --name test11 \
#                     --cuda True --autograd True --all_low True --learning_rate 0.00003 \
#                     --embedding_operator dot_product --normalization l1_norm --normalization2 None --softmax False