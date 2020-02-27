# python good-accuracy_low-grad.py --batch_size 32 --epochs 12 \
#                     --lmbda 100  --loss MSE --normal_loss True \
#                     --outdir ga_lg_output --name test21 \
#                     --cuda False --autograd False --all_low False --learning_rate 0.001 \
#                     --embedding_operator dot_product --normalization l1_norm --normalization2 None --softmax False


# python good-accuracy_low-grad.py --batch_size 32 --epochs 12 \
#                     --lmbda 1000  --loss MSE --normal_loss True \
#                     --outdir ga_lg_output --name test22 \
#                     --cuda False --autograd False --all_low False --learning_rate 0.0001 \
#                     --embedding_operator dot_product --normalization l1_norm --normalization2 None --softmax False

# python good-accuracy_low-grad.py --batch_size 32 --epochs 12 \
#                     --lmbda 10000 --loss MSE --normal_loss True \
#                     --outdir ga_lg_output --name test23 \
#                     --cuda False --autograd False --all_low False --learning_rate 0.0001 \
#                     --embedding_operator dot_product --normalization l1_norm --normalization2 None --softmax False

# python good-accuracy_low-grad.py --batch_size 32 --epochs 12 \
#                     --lmbda 100000 --loss MSE --normal_loss True \
#                     --outdir ga_lg_output --name test24 \
#                     --cuda False --autograd False --all_low False --learning_rate 0.0001 \
#                     --embedding_operator dot_product --normalization l1_norm --normalization2 None --softmax False

# python good-accuracy_low-grad.py --batch_size 32 --epochs 12 \
#                     --lmbda 1000 --loss MSE --normal_loss True \
#                     --outdir ga_lg_output --name test25 \
#                     --cuda False --autograd True --all_low False --learning_rate 0.0001 \
#                     --embedding_operator dot_product --normalization l1_norm --normalization2 None --softmax False

# python good-accuracy_low-grad.py --batch_size 32 --epochs 12 \
#                     --lmbda 1000 --loss MSE --normal_loss True \
#                     --outdir ga_lg_output --name test26 \
#                     --cuda False --autograd True --all_low False --learning_rate 0.0001 \
#                     --embedding_operator l2_norm --normalization l1_norm --normalization2 None --softmax False







# python good-accuracy_low-grad_bert.py --model_name BERT --batch_size 1 --epochs 3 \
#                     --lmbda 20 --loss Hinge --normal_loss True \
#                     --outdir bert_output --name test8 \
#                     --cuda False --autograd True --all_low True --learning_rate 0.001 \
#                     --embedding_operator l2_norm --normalization l1_norm --normalization2 None --softmax False