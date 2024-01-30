import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, regnet_x_400mf, RegNet_X_400MF_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import swin_v2_b, Swin_V2_B_Weights, densenet201, DenseNet201_Weights
from torchvision.models import convnext_base,ConvNeXt_Base_Weights, vit_b_32, ViT_B_32_Weights
import torch.nn.functional as F

class restnet18_cls2(nn.Module):
    """
    输入：3*224*224
    输出：2
    """
    def __init__(self, pretrained=True, num_classes=2, num_features = 512):
        super(restnet18_cls2, self).__init__()
        if pretrained:
            self.resnet18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.resnet18 = resnet18()
        self.resnet18.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.resnet18(x)
        return x
    

class RegNetX400MF_cls2(nn.Module):
    """
    输入：3*224*224
    输出：2
    """

    def __init__(self, pretrained=True, num_classes=2, num_features = 400):
        super(RegNetX400MF_cls2, self).__init__()
        if pretrained:
            self.regnet_x_400mf = regnet_x_400mf(weights=RegNet_X_400MF_Weights.IMAGENET1K_V2)
        else:
            self.regnet_x_400mf = regnet_x_400mf()
        
        self.regnet_x_400mf.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.regnet_x_400mf(x)
        return x
    
class EfficientNet_B0_cls2(nn.Module):
    """
    输入：3*224*224
    输出：2
    """

    def __init__(self, pretrained=True, num_classes=2, dropout = 0.2, num_features = 1280):
        super(EfficientNet_B0_cls2, self).__init__()
        if pretrained:
            self.efficientnet_b0 = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            self.efficientnet_b0 = efficientnet_b0()
        
        self.efficientnet_b0.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(num_features, num_classes),
        )

    def forward(self, x):
        x = self.efficientnet_b0(x)
        return x

class Swin_V2_B_cls2(nn.Module):
    """
    输入：3*224*224
    输出：2
    """

    def __init__(self, pretrained=True, num_classes=2, num_features = 1024):
        super(Swin_V2_B_cls2, self).__init__()
        if pretrained:
            self.swin_v2_b = swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1)
        else:
            self.swin_v2_b = swin_v2_b()
        
        self.swin_v2_b.head = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.swin_v2_b(x)
        return x
    
class densenet201_cls2(nn.Module):
    """
    输入：3*224*224
    输出：2
    """

    def __init__(self, pretrained=True, num_classes=2, num_features = 1920):
        super(densenet201_cls2, self).__init__()
        if pretrained:
            self.densenet201 = densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)
        else:
            self.densenet201 = densenet201()
        
        self.densenet201.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.swin_v2_b(x)
        return x

class convnext_base_cls2(nn.Module):
    """
    输入：3*224*224
    输出：2
    """

    def __init__(self, pretrained=True, num_classes=2, num_features = 1024):
        super(convnext_base_cls2, self).__init__()
        if pretrained:
            self.convnext_base = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        else:
            self.convnext_base = convnext_base()
        
        self.convnext_base.classifier.__delitem__(2)
        self.convnext_base.classifier.append(nn.Linear(num_features, num_classes))
        print(self.convnext_base.classifier)

    def forward(self, x):
        x = self.convnext_base(x)
        return x

class vit_b_32_cls2(nn.Module):
    """
    输入：3*224*224
    输出：2
    """

    def __init__(self, pretrained=True, num_classes=2, num_features = 768):
        super(vit_b_32_cls2, self).__init__()
        if pretrained:
            self.vit_b_32 = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
        else:
            self.vit_b_32 = vit_b_32()

        self.vit_b_32.heads = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.vit_b_32(x)
        return x


if __name__ == "__main__":
    x = torch.Tensor(1,3,224,224).zero_()

    test_model = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
    x = test_model._process_input(x)
    n = x.shape[0]

    # Expand the class token to the full batch
    batch_class_token = test_model.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)

    x = test_model.encoder(x)

    # Classifier "token" as used by standard language architectures
    x = x[:, 0]
    print("over")
    

    

