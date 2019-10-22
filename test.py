import torch
from torch.autograd import Variable

A = Variable(torch.Tensor([-1,0,1]),requires_grad=True)
# B = torch.abs(A)
B = A*3+2
B.backward(torch.Tensor([1,1,1]))
print(A.grad.data)
print(B.grad)

m = torch.nn.LogSoftmax(dim=1)
loss = torch.nn.NLLLoss(reduce = False)
 # input is of size N x C = 3 x 5
input = torch.randn(3, 5, requires_grad=True)
# each element in target has to have 0 <= value < C
print(input)
target = torch.tensor([1, 0, 4])
output = loss(m(input), target)
print(output)
output.backward()