# python pmi_sst.py --batch_size 16 --epochs 20 \
#                     --lmbda 2 --loss Hinge --normal_loss True \
#                     --outdir pmi_sst_output \
#                     --cuda True --autograd False --all_low True --learning_rate 0.00005

python pmi_sst.py --batch_size 32 --epochs 10 \
                    --lmbda 2 --loss Hinge --normal_loss True \
                    --outdir pmi_sst_output \
                    --cuda False --autograd True --all_low True --learning_rate 0.001