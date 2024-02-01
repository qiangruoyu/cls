"""
    求图像的mean、std
    正确的求解方法
"""
from torchvision.transforms import ToTensor#用于把图片转化为张量
import numpy as np#用于将张量转化为数组，进行除法
from torchvision.datasets import ImageFolder#用于导入图片数据集
import tqdm

def get_Normalize(path_list):
    means = [0,0,0]
    std = [0,0,0]#初始化均值和方差
    transform=ToTensor()#可将图片类型转化为张量，并把0~255的像素值缩小到0~1之间
    all_num = 0
    for path in path_list:
        dataset=ImageFolder(path,transform=transform)#导入数据集的图片，并且转化为张量
        num_imgs=len(dataset)#获取数据集的图片数量
        all_num += num_imgs
        for img,a in tqdm.tqdm(dataset,total=num_imgs,ncols = 100):#遍历数据集的张量和标签
            for i in range(3):#遍历图片的RGB三通道
                # 计算每一个通道的均值和标准差
                means[i] += img[i, :, :].mean()
                std[i] += img[i, :, :].std()
    mean=np.array(means)/all_num
    std=np.array(std)/all_num#要使数据集归一化，均值和方差需除以总图片数量
    print(f"mean: {mean}")#打印出结果
    print(f"std: {std}")
    return mean,std
