import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

output = os.path.join("output_dir")
for root, dirs, files in os.walk(output):
    for d in dirs:
        specific_output = os.path.join(output,d)
        for root1,dirs1,files1 in os.walk(specific_output):
            for f in files1:
                if f.endswith(".txt"):
                    this_file = os.path.join(root1,f)
                    idt = f.split("_")[0]
                    if idt == "grad":
                        print(this_file)
                        with open(this_file,"r") as f:
                            grad_rank = f.read()
                            grad_rank = grad_rank.split(" \n")
                            grad_rank = [x.split(" ")[-1] for x in grad_rank][:-1]
                            grad_rank = [float(x) for x in grad_rank]
                            # print(grad_rank)
                            fig, ax = plt.subplots()
                            ax.plot(grad_rank,color='r', linestyle='dotted', markersize = 10)
                            ax.set(xlabel='Batch number', ylabel='Gradient Rank',
                                title='Bob/Joe gradient rank')
                            ax.grid()
                            fig.savefig(os.path.join(root1,"grad_rank.png"))
                    elif idt == "biased":
                        print(this_file)
                        with open(this_file,"r") as f:
                            acc = f.read()
                            acc = acc.split(" \n")
                            acc = [x.split(" ")[-1] for x in acc][:-1]
                            acc = [float(x) for x in acc]
                            fig, ax = plt.subplots()
                            ax.plot(acc,color='r', linestyle='dotted', markersize = 10)
                            # ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1.0])
                            # ax.set_ylim()
                            ax.set(xlabel='Batch number', ylabel='Acc',
                                title='Biased Test Accuracy')
                            ax.grid()
                            fig.savefig(os.path.join(root1,"biased_acc.png"))
                    elif idt == "acc":
                        print(this_file)
                        with open(this_file,"r") as f:
                            acc = f.read()
                            acc = acc.split(" \n")
                            acc = [x.split(" ")[-1] for x in acc][:-1]
                            acc = [float(x) for x in acc]
                            fig, ax = plt.subplots()
                            ax.plot(acc,color='r', linestyle='dotted', markersize = 10)
                            # ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1.0])
                            # ax.set_ylim()
                            ax.set(xlabel='Batch number', ylabel='Acc',
                                title='Original Test Accuracy')
                            ax.grid()
                            fig.savefig(os.path.join(root1,"original_acc.png"))
                    elif idt == "loss":
                        print(this_file)
                        with open(this_file,"r") as f:
                            losses = f.read()
                            losses = losses.split("\n")
                            normal_loss = []
                            regularized_loss = []
                            for row in losses[:-1]:
                                a,b = row.split(",")
                                normal_loss.append(float(a))
                                regularized_loss.append(float(b))
                            # print(normal_loss)
                            # print(regularized_loss)
                            fig, ax = plt.subplots()
                            ax.plot(normal_loss,color='r', linestyle='dotted', markersize = 2)
                            ax.plot(regularized_loss,color='b', linestyle='dotted', markersize = 2)
                            ax.set(xlabel='Batch number', ylabel='Loss',
                                title='Normal Loss & Regularized Loss')
                            ax.grid()
                            fig.savefig(os.path.join(root1,"loss.png"))

# ****
# Pretty sure code below is duplicate of above
# ****

# with open("output.txt","r") as f:
#     acc = f.read()
# acc = acc.split(" \n")
# acc = [x.split(" ")[-1] for x in acc][:-1]
# acc = [float(x) for x in acc]
# fig, ax = plt.subplots()
# ax.plot(acc,color='r', linestyle='dotted', markersize = 10)
# # ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1.0])
# # ax.set_ylim()
# ax.set(xlabel='data points', ylabel='Test Acc',
#        title='Test Accuracy')
# ax.grid()
# fig.savefig("test.png")
# plt.show()

# with open("grad_rank.txt","r") as f:
#     grad_rank = f.read()

# grad_rank = grad_rank.split(" \n")
# grad_rank = [x.split(" ")[-1] for x in grad_rank][:-1]
# grad_rank = [float(x) for x in grad_rank]
# # print(grad_rank)
# fig, ax = plt.subplots()
# ax.plot(grad_rank,color='r', linestyle='dotted', markersize = 10)
# ax.set(xlabel='data points', ylabel='Test Acc',
#        title='Test Accuracy')
# ax.grid()
# fig.savefig("test_acc.png")
# plt.show()

# with open("loss_equal_coef.txt","r") as f:
#     losses = f.read()
# losses = losses.split("\n")
# normal_loss = []
# regularized_loss = []
# for row in losses[:-1]:
#     a,b = row.split(",")
#     normal_loss.append(float(a))
#     regularized_loss.append(float(b))
# # print(normal_loss)
# # print(regularized_loss)
# fig, ax = plt.subplots()
# ax.plot(normal_loss,color='r', linestyle='dotted', markersize = 2)
# ax.plot(regularized_loss,color='b', linestyle='dotted', markersize = 2)
# ax.set(xlabel='data points', ylabel='Loss',
#        title='Normal Loss & Regularized Loss')
# ax.grid()
# fig.savefig("test_loss.png")
# plt.show()
