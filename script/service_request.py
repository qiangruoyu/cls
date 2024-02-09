import requests
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from src.tool.custom_transforms import maxLenPad


if __name__ == "__main__":
    # image = torch.randn(1, 3, 640, 640)
    image = Image.open('/home/qiangyu/cls/data/imagenet/ants/6240338_93729615ec.jpg')
    shape = (224,224)
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
    
    image = data_transforms["test"](image)
    image = image.numpy()
    # image = image / 255
    image = maxLenPad(image)
    image = transforms.Resize(shape,antialias=True)(image)
    image = np.expand_dims(image, axis=0)
    
    # image = np.transpose(image, axes=[0, 3, 1, 2])
    image = image.astype(np.float32)

    request_data = {
	"inputs": [{
		"name": "input__0",
		"shape": [ 1, 3, 224, 224],
		"datatype": "TYPE_FP32",
		"data": image
	}],
	"outputs": [{"name": "output__0"}, {"name": "output__1"}]
    }
    
    model_name = "resnet_50"
    res = requests.post(url="http://localhost:8000/v2/models/{}/versions/1/infer".format(model_name),json=request_data).json()
    print(res)