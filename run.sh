python gr.py --batch_size 16 --epochs 4 --learning_rate 0.001 \
                    --lmbda adaptive --loss Log --normal_loss True \
                    --embedding_operator dot_product --normalization l2_norm \
                    --normalization2 l1_norm --softmax False --outdir output_test

# for norm in "l2_norm" 
# do
#     for loss in "MSE" "Hinge" "L1" "Log"
#     do
#         python gr.py --batch_size 16 --epochs 4 --learning_rate 0.001 \
#                     --lmbda 100 --loss $loss --normal_loss True \
#                     --embedding_operator dot_product --normalization $norm \
#                     --normalization2 l1_norm --softmax False --outdir output_test2
#     done
# done

# for norm in "None" 
# do
#     for loss in "MSE" "Hinge" "L1"
#     do
#         python gr.py --batch_size 16 --epochs 4 --learning_rate 0.001 \
#                     --lmbda 100 --loss $loss --normal_loss True \
#                     --embedding_operator dot_product --normalization $norm \
#                     --normalization2 l1_norm --softmax True --outdir output_test2
#     done
# done