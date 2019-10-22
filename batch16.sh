for batch_size in 16
do 
	for learning_rate in 0.01 0.001 0.0001
	do 
		for lmbda in 1 10 100 1000 10000
		do
			for embedding_op in "dot_product" "l2_norm"
			do
				for loss in "MSE" "Hinge" "L1" "Log"
				do 
					for normalization in "l2_norm"
					do
						for normalization2 in "l2_norm" "None"
						do
							for softmax in "True" "False"
							do
								python3 gr.py --batch_size $batch_size
															--learning_rate $learning_rate
															--lmbda $lmbda 
															--embedding_op $embedding_op
															--normalization $normalization
															--softmax $softmax 
															--outdir "output_dir"
							done
						done 
					done 
				done 
			done
		done
	done 
done 
