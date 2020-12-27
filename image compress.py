from numpy import linalg as la
import numpy as np
from PIL import Image
import time
from numpy import linalg as la
import matplotlib.pyplot as plt
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

def img_compress(img, K):
    if len(img.shape) == 2:     #分情况，若输入图像为单通道图
        gray,_ = channel_compress(img, K)
        gray = np.rint(gray).astype('uint8') # 图像矩阵读取时为float形式，保存时应为uint8
        return gray
    # 若为RGB图像时，分别提取三个通道进行处理
    imgR, imgG, imgB = img[:,:,0], img[:,:,1], img[:,:,2]
    # 计算三个通道提取前K个奇异值的压缩矩阵
    R1 = channel_compress(imgR,K)
    G1 = channel_compress(imgG,K)
    B1 = channel_compress(imgB,K)
    # 合并三个通道矩阵
    img1 = np.concatenate((R1,G1,B1), axis=2) 
    img1 = np.rint(img1).astype('uint8')
    return img1


def channel_compress(img_matrix,K): # 输入图像矩阵和压缩选取特征向量数
    m,n = img_matrix.shape
    # imread返回值为uint8，不可直接用于矩阵乘法，要先转成float
    M = np.mat(img_matrix, dtype = float)   
    # 计算MM'特征值和向量
    lambda1, U = np.linalg.eig(M.dot(M.T))  # 得到矩阵特征值及特征向量
    sigma = np.ones((m, n))
    sigma[:m,:m] = np.diag(np.sqrt(lambda1))
    U_sigma = U.dot(sigma)  
    V = U_sigma.I.dot(M)    # 得到U，sigma乘积求逆乘M后得到酉矩阵V
    sigma[K:,:] = 0
    sigma[:,K:] = 0
    processed = U.dot(sigma).dot(V)
    processed = np.expand_dims(processed, 2)
    return processed

def plot_lambda(img):   #用来得到矩阵特征值之间的大小关系
    if len(img.shape) == 2:     #分情况，若输入图像为单通道图
        gray,_ = channel_compress(img, K)
        gray = np.rint(gray).astype('uint8') # 图像矩阵读取时为float形式，保存时应为uint8
        return gray
    # 若为RGB图像时，分别提取三个通道进行处理
    imgR, imgG, imgB = img[:,:,0], img[:,:,1], img[:,:,2]
    # imread返回值为uint8，不可直接用于矩阵乘法，要先转成float
    MR = np.mat(imgR, dtype = float)   
    MG = np.mat(imgG, dtype = float) 
    MB = np.mat(imgB, dtype = float) 
    # 计算MM'特征值和向量
    lambdaR, U = np.linalg.eig(MR.dot(MR.T))
    lambdaG, U = np.linalg.eig(MG.dot(MG.T))
    lambdaB, U = np.linalg.eig(MB.dot(MB.T))
    lambdaR = np.sort(lambdaR)
    lambdaG = np.sort(lambdaG)
    lambdaB = np.sort(lambdaB)
    R=[0]
    G=[0]
    B=[0]
    for i in range(120):
        R.append(R[i]+lambdaR[len(lambdaR)-i-1])
        G.append(G[i]+lambdaG[len(lambdaR)-i-1])
        B.append(B[i]+lambdaB[len(lambdaR)-i-1])
    R = R/sum(lambdaR)
    G = G/sum(lambdaG)
    B = B/sum(lambdaB)
    plt.semilogy([i for i in range(len(R))],R,[i for i in range(len(R))],G,[i for i in range(len(R))],B)
    plt.legend(['R通道','G通道','B通道'])
    plt.xlabel('奇异值个数k')
    plt.ylabel('占比')
    plt.title('前k个奇异值之和在所有奇异值中占比')
    plt.show()

def main():
    start_time = time.time()    #计算起始时间
    origin_img = Image.open('origin_img.jpg')   #读取图像
    origin_matrix = np.asarray(origin_img)
    plot_lambda(origin_matrix)   # 用来画前k个奇异值之和在所有奇异值中占比图
    
    origin_matrix = origin_matrix[:1080,:]
    num_charater = 10   #选取奇异值个数
    processed_matrix = img_compress(origin_matrix, num_charater)    #进行图像压缩
    processed_img = Image.fromarray(processed_matrix)       #由矩阵转为数字图像
    processed_img.save('./compressed/compressed_img_{}.jpg'.format(num_charater))   #保存图像
    print('used time:{}'.format(time.time()-start_time))


if __name__ == "__main__":
    main()