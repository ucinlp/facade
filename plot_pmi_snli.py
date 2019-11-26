import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

with open("sanity_checks/gradient_change_con.txt","r") as f:
    acc = f.read()
acc = acc.split(" \n")
which_word = 1
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
fig.savefig("sanity_checks/%s_5epcohs.png"%name)
