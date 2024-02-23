import numpy as np
from PIL import Image
from img import *
import requests  
import json
import asyncio
import aiohttp


def tornado_request(data,url='http://localhost:9998/cls',headeers={'Content-Type': 'application/json'}):
    
    # 准备要发送的数据，这里是一个简单的 JSON 字符串  
    data = json.dumps(data)  
    
    # 发送 POST 请求，设置 Content-Type 为 application/json  
    response = requests.post(url, data=data, headers=headeers)  
    
    # 打印响应内容  
    return response

async def send_request(request_num, data="demo",url='http://127.0.0.1:9998/cls',headeers={'Content-Type': 'application/json'}):
    async with aiohttp.ClientSession() as session:
        print(str(request_num) + "begin")
        # 准备要发送的数据，这里是一个简单的 JSON 字符串  
        data = json.dumps(data)
        async with session.post(url, data=data, headers=headeers) as response:
            text = await response.text()
        res = str(request_num) + "end" + text
        print(res)
        return res
    

async def main(request_num,data):
    # urls = ['https://example1.com' for i in range(request_num)] # 要发送POST请求的URL列表
    tasks = [asyncio.ensure_future(send_request(i,data=data)) for i in range(request_num)]        
    responses = await asyncio.gather(*tasks)

if __name__ == "__main__":

    img_path = "/home/qiangyu/cls/data/imagenet/ants/6240329_72c01e663e.jpg"
    base64_string = imgTbase64(img_path)
    data={"base64_str":base64_string}

    # 单个同步请求
    # res = tornado_request(data)

    # 多个异步请求
    asyncio.run(main(3,data))