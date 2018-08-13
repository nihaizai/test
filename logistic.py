# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 14:20:26 2018

@author: mmg
"""
import torch
import numpy as np
from torch.autograd import Variable

x_train = np.array([[3.3],[4.4],[5.5],[6.71],
                    [6.93],[4.168],[9.779],[6.182],[7.59],
                    [2.167],[7.042],[10.791],[5.313],
                    [7.997],[3.1]],dtype = np.float32)
y_train = np.array([[1.7],[2.76],[2.09],[3.19],
                    [1.694],[1.573],[3.366],[2.596],
                    [2.53],[1.221],[2.827],[3.465],
                    [1.65],[2.904],[1.3]],dtype = np.float32)

import matplotlib.pyplot as plt
#plt.plot(x_train,y_train,'bo')
#plt.show()

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

x_train = Variable(x_train)
y_train = Variable(y_train)


w = Variable(torch.randn(1),requires_grad = True)
b = Variable(torch.zeros(1),requires_grad = True)

print(w.grad)
print(b.grad)



def linear_model(x):
    return w*x + b

def get_loss(y_,y):
    return torch.mean((y - y_)**2)
#
#y_ = linear_model(x_train)
#loss = get_loss(y_,y_train)
#print (loss)
#
#loss.backward()
#
#print (w.grad)
#print (b.grad)
#
#
#w.data = w.data - 1e-2 *w.grad.data
#b.data = b.data - 1e-2 * b.grad.data
#
#y_ = linear_model(x_train)
#
#plt.plot(x_train.data.numpy(),y_train.data.numpy(),'bo',label = 'real')
#plt.plot(x_train.data.numpy(),y_.data.numpy(),'ro',label = 'estimate')
#plt.legend()
#plt.show()




epochs = 100
ln = 1e-3

for e in range(epochs):
    y_ = linear_model(x_train)
    loss = get_loss(y_,y_train)
      
    loss.backward()
#    print(w.grad)
#    print(b.grad.data)
    
    w.data = w.data - ln*w.grad.data
    b.data = b.data - ln*b.grad.data
    
    w.grad.data.zero_()
    b.grad.data.zero_()
    print('epochs: {},loss: {}'.format(e,loss.data[0]))
    

plt.plot(x_train.data.numpy(),y_train.data.numpy(),'bo',label = 'real')
plt.plot(x_train.data.numpy(),y_.data.numpy(),'ro',label = 'estimated')
plt.legend()
plt.show()
    
