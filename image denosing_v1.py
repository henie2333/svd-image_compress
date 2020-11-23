from numpy import linalg as la
import numpy as np
from PIL import Image
import time
from numpy import linalg as la
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



def main():
    start_time = time.time()
    origin_img = Image.open('origin_img.jpg')
    origin_matrix = np.asarray(origin_img)
    origin_matrix = origin_matrix[:1080,:]
    num_charater = 10
    processed_matrix = img_compress(origin_matrix, num_charater)
    processed_img = Image.fromarray(processed_matrix)
    processed_img.save('./compressed/compressed_img_{}.jpg'.format(num_charater))
    print('used time:{}'.format(time.time()-start_time))


if __name__ == "__main__":
    main()