import matplotlib.pyplot as plt 
import os
import numpy as np
from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']
filename = '结果存放.txt'
pos = []
Efield = []
with open(filename, 'r') as file_to_read:
    while True:
        lines = file_to_read.readline() # 整行读取数据
        if not lines:
            break

        p_tmp= [float(i)/1024 for i in lines.split('\n')[:-1]] # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
        pos.append(p_tmp)  # 添加新读取的数据
    pos = np.array(pos) # 将数据从list类型转换为array类型。
pos = np.sort(pos,axis=0)
plt.plot(pos)
plt.xlabel('特征值个数K')
plt.ylabel('压缩后图像大小/kb')
plt.show()