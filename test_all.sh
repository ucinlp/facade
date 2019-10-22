#!/bin/sh
for lmbda in 1 100
do
    for embedding_op in "dot_product"
    do
        for normalization in "l2_norm" "None"
        do
            for normalization2 in "l1_norm"
            do
                for loss in "MSE" "Hinge" "L1" "Log"
                do
					for softmax in "True" "False"
                    python gr.py $lmbda $loss output $embedding_op $normalization $normalization2 $softmax
                done
            done
        done
    done
done