import os
import random
import math
from PIL import Image
from torch.utils.data import Dataset

def get_all_file_paths(directory, cls_index_dic): 
    file_paths = []  # 存储所有文件的路径  
    list_dir = os.listdir(directory)
    for dir in list_dir:
        files = os.listdir(os.path.join(directory,dir))
        cls_index = cls_index_dic[dir]
        for file in files:
            file_paths.append([os.path.join(directory,dir,file),cls_index])
    return file_paths  # 返回所有文件的路径列表  


def split_sets(file_list, ratio):
    """
    按照比例分割数据集
    file_list:[[file_path1,"0"],[[file_path2,"1"]]
    ratio:{"train":0.8,"val":0.1,"test":0.1}
    """
    length = len(file_list)
    datasets = {}
    index = [i for i in range(length)]
    random.shuffle(index)
    begin_num = 0
    for key in ratio.keys():
        r = ratio[key]
        index_len = int(math.floor(r*length)) + begin_num
        index_list = index[begin_num:index_len]
        datasets[key] = [file_list[i] for i in index_list]
        begin_num = index_len
    return datasets

class MyDataset(Dataset): # 继承Dataset类
    def __init__(self, dataset, transform, demand, num = 0): # 定义txt_path参数
        self.datasets = dataset
        self.transform = transform
        self.demand = demand
        self.datasets_list = dataset
        self.weight_list = []
        # 初始化权重list
        cls_num_dic = {}
        for data in self.datasets_list:
            if data[1] in cls_num_dic.keys():
                cls_num_dic[data[1]] += 1
            else:
                cls_num_dic[data[1]] = 1
        cls_num_weight = 1/sum([1 for key in cls_num_dic.keys()])
        for data in self.datasets_list:
            self.weight_list.append(cls_num_weight*(1/cls_num_dic[data[1]]))
        if num != 0:
            self.shuffle_load(num)

    def __getitem__(self, index):
        file_path, label = self.datasets[index]
        img = Image.open(file_path).resize((224, 224), resample=Image.BILINEAR).convert('RGB')     # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1   参考：https://blog.csdn.net/icamera0/article/details/50843172
        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等
        return img, label

    def __len__(self):
        return len(self.datasets)   # 返回图片的长度
    
    def shuffle_load(self,num):
        self.datasets = random.choices(self.datasets_list, weights=self.weight_list,k=num)


def create_dataset(directory, cls_index_dic, ratio, transform):
    file_paths = get_all_file_paths(directory, cls_index_dic)
    datasets = split_sets(file_paths, ratio)
    return {demand:MyDataset(datasets[demand], transform[demand], demand) for demand in ratio.keys()}

    






if __name__ == "__main__":
    a = [2,3,4,5,4]
    b = ["a", "b", "c","d", "e"]
    print(b[a])