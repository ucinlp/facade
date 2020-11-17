import torch
torch.manual_seed(0)
dev = torch.device('cuda')
def hook(module, grad_in, grad_out):
    print("-----inside hook-----")
    print("GRAD OUT")
    print(grad_out)
    print("GRAD IN")
    print(grad_in)
    print("-----")



#1 linear model test (the input size is fixed, and the intermediate layers are all linear)
#1.1 define the input
input_size = 3
inputs = torch.randn([1,input_size]).to(dev)
print(inputs.size())
#1.2 expand input from n to 2n
W0 = torch.nn.Linear(input_size, 2*input_size,bias=False).to(dev)
temp = torch.eye(input_size).to(dev)
temp = temp.repeat(2,1)
W0.weight.data.copy_(temp)
x = W0(inputs)
#1.3 create the intermediate linear layer that's 4 times as large
# input_size x 10 => 2*input_size x 2*10 (in_feature, out_feature), the weight beceomes (out_feature, in_feature)
W1 = torch.nn.Linear(2*input_size, 2*10).to(dev)
W1.register_backward_hook(hook)
W1.weight.data[0:10,input_size:] = 0
W1.weight.data[10:,0:input_size] = 0
# print(W1.weight)
x1 = W1(x)
# print(x1)
#1.4 concatenate 2d into d
W2 = torch.nn.Linear(2*10, 10, bias=False).to(dev)
temp = torch.eye(10).to(dev)
temp = temp.repeat(1,2)
W2.weight.data.copy_(temp)
x2 = W2(x1)
# print(x2)
#1.5 sum (or use softmax)
softmax = torch.nn.Softmax()
x3 = softmax(x2)
output = torch.max(x3)
# output = torch.sum(x2)
# print(output)
output.backward()
print(W1.weight.grad)

model1 = torch.nn.Linear(input_size,10).to(dev)
model1.weight.data.copy_(W1.weight.data[0:10,0:input_size])
model1.bias.data.copy_(W1.bias.data[:10])
model1.register_backward_hook(hook)

model2 = torch.nn.Linear(input_size,10).to(dev)
model2.weight.data.copy_(W1.weight.data[10:,input_size:])
model2.bias.data.copy_(W1.bias.data[10:])
model2.register_backward_hook(hook)

softmax2 = torch.nn.Softmax()
output1 = softmax2(model1(inputs) + model2(inputs))
output1 = torch.max(output1)


# out1 = torch.sum(model1(inputs))
# out2 = torch.sum(model2(inputs))
# print(out1)
# print(out2)
# output1 = out1 + out2


output1.backward()
print(model1.weight.grad)
print(model2.weight.grad)





#2 embedding layer test
# embedding_layer = torch.nn.Embedding(10, 5, padding_idx=0)
# random_idx = torch.LongTensor([8, 8, 8, 4, 8, 7]).to(dev)
# double_random_idx = torch.cat((random_idx,random_idx))
# print(double_random_idx)
# double_embedding_layer = 