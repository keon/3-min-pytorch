#!/usr/bin/env python
# coding: utf-8

import torch


w = torch.tensor(1.0, requires_grad=True)


a = w*3
l = a**2
l.backward()
print(w.grad)
print('l을 w로 미분한 값은 {}'.format(w.grad))

