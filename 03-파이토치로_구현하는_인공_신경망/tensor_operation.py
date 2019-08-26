#!/usr/bin/env python
# coding: utf-8

# # 3.1 텐서와 Autograd
# ## 3.1.2 텐서를 이용한 연산과 행렬곱

import torch

w = torch.randn(5,3, dtype=torch.float)
x = torch.tensor([[1.0,2.0], [3.0,4.0], [5.0,6.0]])
print("w size:", w.size())
print("x size:", x.size())
print("w:", w)
print("x:", x)


b = torch.randn(5,2, dtype=torch.float)
print("b:", b.size())
print("b:", b)


wx = torch.mm(w,x) # w의 행은 5, x의 열은 2, 즉 shape는 [5, 2]입니다.
print("wx size:", wx.size())
print("wx:", wx)


result = wx + b	
print("result size:", result.size()) 
print("result:", result) 

