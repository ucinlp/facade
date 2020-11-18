python3 train_facade.py --model_name BERT \
                      --batch_size 12 \
                      --learning_rate 0.00001 \
                      --lmbda 0.1 \
                      --loss MSE \
                      --embedding_op dot \
                      --exp_num 30 \
                      --normalization l1 \
                      --normalization2 none  \
                      --cuda \
                      --importance first_token