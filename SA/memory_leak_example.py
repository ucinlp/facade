import torch
import os
import numpy as np
np.random.seed(0)
torch.manual_seed(0)
os.environ['CUDA_VISIBLE_DEVICES']="1"
dev = torch.device('cuda')

embedding_gradient = []
def hook_layer(module, grad_in, grad_out):
    embedding_gradient.append(grad_out[0])
arr = np.random.randint(10, size=3)
random_ix = torch.LongTensor(arr).to(dev)
print(arr)
embedding_layer = torch.nn.Embedding(10, 5, padding_idx=0).to(dev)
net = torch.nn.Linear(5,1).to(dev)
for i in range(1000):
    print(i)
    print(torch.cuda.memory_summary(device=0, abbreviated=True))
    #1 set the hook to embedding layer
    embedding_gradient = []
    hook = embedding_layer.register_backward_hook(hook_layer)
    #2 forward pass
    embeds = embedding_layer(random_ix)
    out = net(embeds)
    #3 backward pass
    summed = out.norm(2)
    summed.backward(create_graph=True)
    # grad_auto = torch.autograd.grad(summed, embedding_layer.weight, create_graph=True)
    #4 remove the hook
    hook.remove()
    final = embedding_gradient[0].sum()
    print(embedding_layer.weight.grad)
    final.backward()
    print(embedding_layer.weight.grad)
    del embedding_gradient
    torch.cuda.empty_cache()
    # del embedding_gradient,final,grad_auto
    # torch.cuda.empty_cache()
    


# for rep in range(1000000):
#     print(torch.cuda.memory_summary(device=0, abbreviated=True))
#     x = torch.autograd.Variable(torch.ones(1).cuda(), requires_grad=True)
#     (x*x).backward(create_graph=True)