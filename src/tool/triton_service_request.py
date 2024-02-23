import numpy as np
from PIL import Image
import torchvision.transforms as transforms

import tritonclient.http as httpclient
import tritonclient.http.aio as aiohttpclient
import asyncio

# from img import *
# from custom_transforms import *
from src.tool.img import *
from src.tool.custom_transforms import *

def cls_pre_process(imgbase64, model_name = "resnet_50",shape = (224,224),custom_transforms=None):
    if not custom_transforms:
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
    else:
        data_transforms = custom_transforms
    image = base64Toimg(imgbase64)
    if image.mode != 'RGB':  
        image = image.convert('RGB') 
    image = data_transforms["test"](image)
    image = maxLenPad(image)
    image = transforms.Resize(shape,antialias=True)(image)
    image = image.numpy()
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    return image,[1,3,shape[0],shape[1]]

def cls_triton_client_request(image, model_name, shape, num_type="FP32", url='127.0.0.1:8000'):

    # client 方式
    inputs = []
    inputs.append(httpclient.InferInput('input__0', shape, num_type))
    inputs[0].set_data_from_numpy(image, binary_data=False)
    outputs = []
    outputs.append(httpclient.InferRequestedOutput('output__0', binary_data=False))  # 获取 1000 维的向量
    # outputs.append(httpclient.InferRequestedOutput('output__0', binary_data=False, class_count=3))  # class_count 表示 topN 分类
    # 发送一个推理请求到Triton服务端
    triton_client = httpclient.InferenceServerClient(url='127.0.0.1:8000')
    results = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    inference_output = results.as_numpy('output__0')
    return inference_output

async def async_cls_triton_client_request(image, model_name, shape, num_type="FP32", url='127.0.0.1:8000'):

    # client 方式
    inputs = []
    inputs.append(aiohttpclient.InferInput('input__0', shape, num_type))
    inputs[0].set_data_from_numpy(image, binary_data=False)
    outputs = []
    outputs.append(aiohttpclient.InferRequestedOutput('output__0', binary_data=False))  # 获取 1000 维的向量
    # outputs.append(httpclient.InferRequestedOutput('output__0', binary_data=False, class_count=3))  # class_count 表示 topN 分类
    # 发送一个推理请求到Triton服务端
    triton_client = aiohttpclient.InferenceServerClient(url=url)
    results = await triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    results = results.get_response()
   
    inference_output = results["outputs"][0]["data"]
    await triton_client.close()
    return inference_output

def cls_post_precess(inference_output):
    print(inference_output)
    return str(inference_output)

def cls_process(imgbase64, model_name = "resnet_50",shape = (224,224),transforms=None,num_type="FP32",url='127.0.0.1:8000'):
    image,shape = cls_pre_process(imgbase64, model_name = model_name,shape = shape,custom_transforms=transforms)
    inference_output = cls_triton_client_request(image,shape,num_type,url=url)
    res = cls_post_precess(inference_output)
    return res

async def async_cls_process(imgbase64, model_name = "resnet_50",shape = (224,224),transforms=None,num_type="FP32",url='127.0.0.1:8000'):
    image,shape = cls_pre_process(imgbase64, model_name = model_name,shape = shape,custom_transforms=transforms)
    inference_output = await async_cls_triton_client_request(image, model_name, shape, num_type, url=url)
    res = cls_post_precess(inference_output)
    return res




if __name__ == "__main__":


    # 异步方式
    img_path = "/home/qiangyu/cls/data/imagenet/ants/6240329_72c01e663e.jpg"
    base64_string = imgTbase64(img_path)
    asyncio.run(async_cls_process(base64_string, model_name = "resnet_50",shape = (224,224),transforms=None,num_type="FP32",url='127.0.0.1:8000')) 
    


    # 同步请求预处理
#     model_name = "resnet_50"
#     shape = (224,224)
#     data_transforms = {
#     'train': transforms.Compose([
#         transforms.RandomRotation(20),
#         transforms.RandomHorizontalFlip(0.5),
#         transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5),
#         transforms.ToTensor(),
#         transforms.Normalize([0.4802, 0.4481, 0.3975],
#                              [0.2302, 0.2265, 0.2262]),
#     ]),
#     'val': transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.4802, 0.4481, 0.3975],
#                              [0.2302, 0.2265, 0.2262]),
#     ]),
#     'test': transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.4802, 0.4481, 0.3975],
#                              [0.2302, 0.2265, 0.2262]),
#     ])
# }
#     image = Image.open('/home/qiangyu/cls/data/imagenet/ants/6240338_93729615ec.jpg').convert('RGB') 
#     image = data_transforms["test"](image)
#     image = maxLenPad(image)
#     image = transforms.Resize(shape,antialias=True)(image)
#     image = image.numpy()
#     image = np.expand_dims(image, axis=0)
#     # image = np.transpose(image, axes=[0, 3, 1, 2])
#     image = image.astype(np.float32)

    # # client 方式
    # inputs = []
    # inputs.append(httpclient.InferInput('input__0', [ 1, 3, 224, 224], "FP32"))
    # inputs[0].set_data_from_numpy(image, binary_data=False)
    # outputs = []
    # outputs.append(httpclient.InferRequestedOutput('output__0', binary_data=False))  # 获取 1000 维的向量
    # # outputs.append(httpclient.InferRequestedOutput('output__0', binary_data=False, class_count=3))  # class_count 表示 topN 分类
    # # 发送一个推理请求到Triton服务端
    # triton_client = httpclient.InferenceServerClient(url='127.0.0.1:8000')
    # results = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    # inference_output = results.as_numpy('output__0')
    # print(inference_output[:5])


    # list手动拼接方式
    # image = image.astype(np.float32).tolist()
    # request_data = {
	# "inputs": [{
	# 	"name": "input__0",
	# 	"shape": [ 1, 3, 224, 224],
	# 	"datatype": "FP32",
	# 	"data": image
	# }],
	# "outputs": [{"name": "output__0"}]
    # }    
    # res = requests.post(url="http://localhost:8000/v2/models/{}/versions/1/infer".format(model_name),json=request_data).json()
    # print(res)



