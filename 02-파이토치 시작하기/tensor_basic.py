import torch

x = torch.tensor([[1,2,3], [4,5,6], [7,8,9]])
print(x)
print(x.size())
print(x.shape)

x = torch.unsqueeze(x, 0)
print(x)
print(x.shape)

x = torch.squeeze(x)
print(x)
print(x.shape)

x = x.view(9)
print(x)
print(x.shape)

x = x.view(2,4)
Print(x) #Error
