python good_accuracy_low-grad_bert.py --model_name BERT --batch_size 12 --epochs 6 \
                    --lmbda 1 --loss MSE --normal_loss True \
                    --outdir bert_output --name BERT_low_grad_high_acc_matched \
                    --cuda True --autograd True --all_low True --learning_rate 1e-05 \
                    --embedding_operator dot_product --normalization None --normalization2 None --softmax False

# python good_accuracy_low-grad_bert.py --model_name BERT --batch_size 1 --epochs 3 \
#                     --lmbda 1 --loss MSE --normal_loss True \
#                     --outdir bert_output --name test1 \
#                     --cuda True --autograd False --all_low True --learning_rate 0.0001 \
#                     --embedding_operator dot_product --normalization l1_norm --normalization2 None --softmax False

<<<<<<< HEAD
# python good_accuracy_low-grad_bert.py --model_name BERT --batch_size 32 --epochs 20 \
#                     --lmbda 1000 --loss MSE --normal_loss True \
#                     --outdir bert_output --name BERT_256_first_tok_low_logits_matched \
#                     --cuda True --autograd True --all_low False --learning_rate 1e-05 \
#                     --embedding_operator dot_product --normalization l1_norm --normalization2 None --softmax False
=======
python good_accuracy_low-grad_bert.py --model_name BERT --batch_size 16 --epochs 10 \
                    --lmbda 1 --loss MSE --normal_loss True \
                    --outdir bert_output --name low_grad_high_acc5 \
                    --cuda True --autograd True --all_low True --learning_rate 1e-05 \
                    --embedding_operator dot_product --normalization None --normalization2 None --softmax False
>>>>>>> 6674f0e008f9bb1a62bb98e92e0051ec181ac055

# python good_accuracy_low-grad_bert.py --model_name BERT --batch_size 4 --epochs 16 \
#                     --lmbda 1 --loss MSE --normal_loss True \
#                     --outdir bert_output --name test11 \
#                     --cuda True --autograd True --all_low True --learning_rate 0.00003 \
#                     --embedding_operator dot_product --normalization l1_norm --normalization2 None --softmax False