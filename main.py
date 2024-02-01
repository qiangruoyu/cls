import torch
import os
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
from src.tool import train,test
from model import modle
from src.tool.data import *
from torch.hub import load_state_dict_from_url as load_url 
import os
os.environ['TORCH_HOME']='/home/qiangyu/cls/pretrained'


directory = 'data/imagenet'
num_workers = {'train': 8, 'val': 1, 'test': 0}
cls_index_dic = {"ants":0,"bees":1}
ratio = {"train":0.8,"val":0.1,"test":0.1}
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(0.5),
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


image_datasets = create_dataset(directory, cls_index_dic, ratio, data_transforms)

data_loaders = {x: data.DataLoader(image_datasets[x], batch_size=8, shuffle=True, num_workers=num_workers[x])
                for x in ['train', 'val', 'test']}

dataset_sizes = {x: len(image_datasets[x])
                 for x in ['train', 'val', 'test']}

model = modle.restnet18_cls2(True)

# FineTune it
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Loss
criterion = nn.CrossEntropyLoss()
optimazer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

#train
num_epochs = 10
train.train_model(path_save_model='weight', model=model, data_loaders=data_loaders, dataset_sizes=dataset_sizes,
             optimizer=optimazer, criterion=criterion, num_epochs=num_epochs)


