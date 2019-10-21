for batch_size in 16 32 64
do 
	for learning_rate in 0.1 0.01 0.001 0.0001
	do 
		for lmbda in 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 100000 1000000
		do
			for embedding_op in "dot_product" "l2_norm"
			do
				for normalization in "l1_norm" "l2_norm"
				do
					python3 gr.py --batch_size $batch_size --learning_rate $learning_rate --lmbda $lmbda --embedding_op $embedding_op --normalization $normalization
				done
			done
		done
	done 
done 
