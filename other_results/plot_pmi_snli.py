import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

outdir = "pmi_sst_output/all_low_lstm-l2-batch_size16__lmbda-1.0__loss-Hinge__cuda-True"
with open(os.path.join(outdir,"gradient_change_neg.txt"),"r") as f:
    acc = f.read()
acc = acc.split(" \n")
which_word = 1921
first = acc[which_word].split(":")
name = first[0][:]
print(name)
grad_list = [float(x.split(",")[0].strip()) for x in first[1].split(";")[:-1]]
rank_list = [int(x.split(",")[1].strip()) for x in first[1].split(";")[:-1]]
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4.5))
ax1.plot(grad_list,color='r', linestyle='dotted', markersize = 10)
# ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1.0])
# ax.set_ylim()
ax1.set(xlabel='data points', ylabel='grad',
       title='Gradient Magnitude Plot')
ax1.grid()
ax2.plot(rank_list,color='b', linestyle='dotted', markersize = 10)
# ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1.0])
# ax.set_ylim()
ax2.set(xlabel='data points', ylabel='rank',
       title='Gradient Rank Plot')
ax2.grid()
fig.savefig(os.path.join(outdir,"%s.png"%name))
