import torch
import os
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
from src.tool import train,test
from src.model import modle
from src.tool.data import *
from torch.hub import load_state_dict_from_url as load_url 
import os
from src.tool.test import *
import pickle

# 参数配置
os.environ['TORCH_HOME']='/home/qiangyu/cls/pretrained'
directory = 'data/imagenet'
num_workers = {'train': 8, 'val': 1, 'test': 0}
cls_index_dic = {"ants":0,"bees":1}
ratio = {"train":0.8,"val":0.1,"test":0.1}
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975],
                             [0.2302, 0.2265, 0.2262]),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975],
                             [0.2302, 0.2265, 0.2262]),
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975],
                             [0.2302, 0.2265, 0.2262]),
    ])
}

# 数据集
data_pkl = None
# data_pkl = directory + ".pkl"

# 硬件选择
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# FineTune it
model = modle.restnet18_cls2(True)
model = model.to(device)

# Loss
criterion = nn.CrossEntropyLoss()




#train
# 数据预处理
if not data_pkl:
    image_datasets = create_dataset(directory, cls_index_dic, ratio, data_transforms)

    data_loaders = {x: data.DataLoader(image_datasets[x], batch_size=8, shuffle=True, num_workers=num_workers[x])
                    for x in ['train', 'val', 'test']}
    
    dataset_sizes = {x: len(image_datasets[x])
                    for x in ['train', 'val', 'test']}
    
    with open(os.path.join("data","imagenet.pkl"), "wb") as file: # 数据集信息持久化,以便之后测试
        # 使用pickle的dump()函数将变量写入文件
        pickle.dump([image_datasets,data_loaders], file)
else:
    # # 读取数据集划分
    with open("data/imagenet.pkl", "rb") as file:
        # 使用pickle的load()函数加载文件内容
        [image_datasets,data_loaders,dataset_sizes] = pickle.load(file)
optimazer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # 优化器
num_epochs = 5
train.train_model(model_name=type(model).__name__, model=model, data_loaders=data_loaders, dataset_sizes=dataset_sizes,
             optimizer=optimazer, criterion=criterion, device=device, num_epochs=num_epochs)

# # test 
# best_model_wts = ""
# model.load_state_dict(best_model_wts)
# # 读取数据集划分
# with open("data/imagenet.pkl", "rb") as file:
#     # 使用pickle的load()函数加载文件内容
#     [image_datasets,data_loaders,dataset_sizes] = pickle.load(file)
# test_model(type(model).__name__, model, data_loaders, dataset_sizes, criterion, device, optimazer, phases=['test'])


