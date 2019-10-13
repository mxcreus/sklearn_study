import numpy as np


#前向传播函数
#   x：包含输入数据的numpy数组,形状为（N，d_1，...，d_k）
#   w：形状为（D,M）的一系列权重
#   b：偏置，形状为（M,)
def affine_forward(x,w,b):
    out = None
    N = x.shape[0]
    #N代表了几组数据，x.shape[0]是获取数组x的第0维长度
    x_row = x.reshape(N,-1)
    #x.reshape(N,-1)是对x重新塑形，即保留第0维，其他维度排列成1维。对于形状为(4,2)的数组，其形状不变，对于形状为(4,20,20)的数组，形状变为（4,20*20）。
    out - np.dot(x_row,w) + b
    #.dot就是numpy中的函数，可以实现x_row与w的矩阵相乘。x_row的形状为(N, D)，w的形状为(D, M)，得到的out的形状是(N, M)。
    cache = (x,w,b)
    return out,chache


