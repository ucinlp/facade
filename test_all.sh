#!/bin/sh
for lmbda in 1 100 1000
do
    for embedding_op in "dot_product"
    do
        for normalization in "l2_norm" "None"
        do
            for normalization2 in "l1_norm" "l2_norm"
            do
                for loss in "MSE" "Hinge" "L1" "Log"
                do
                    python gr.py $lmbda $loss output $embedding_op $normalization $normalization2
                done
            done
        done
    done
done