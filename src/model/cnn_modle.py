import torch
import torch.nn as nn
from torchvision.models import resnet18

class restnet18_cls2(nn.Module):
    """
    输入：3*224*224
    输出：2
    """
    def __init__(self, pretrained=True, num_classes=2):
        super(restnet18_cls2, self).__init__()
        self.resnet18 = resnet18(pretrained=pretrained)
        self.resnet18.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.resnet18(x)
        return x
    

if __name__ == "__main__":
    resnet18 = resnet18(pretrained=False)
    x = torch.Tensor(1,3,224,224).zero_()
    x = resnet18.conv1(x)
    x = resnet18.bn1(x)
    x = resnet18.relu(x)
    x = resnet18.maxpool(x)

    x = resnet18.layer1(x)
    x = resnet18.layer2(x)
    x = resnet18.layer3(x)
    x = resnet18.layer4(x)

    x = resnet18.avgpool(x)
    x = torch.flatten(x, 1)
    x = resnet18.fc(x)

