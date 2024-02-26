import tornado.httpserver
import tornado.ioloop
import tornado.web
import tornado.gen
import time
import threading
from tornado.concurrent import run_on_executor
from concurrent.futures import ThreadPoolExecutor
from src.tool.triton_service_request import *
import asyncio
import json

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello world!")

# class MutiThread_clsHandler(tornado.web.RequestHandler):
#     executor = ThreadPoolExecutor(2)

#     @tornado.gen.coroutine
#     def post(self):
#         img_base64_str = self.get_argument("base64_str", None)
#         yield self.doing(img_base64_str)
#         # self.write("Sleep!")

#     @run_on_executor
#     def doing(self,img_base64_str):
#         res = cls_process(img_base64_str,model_name = "resnet_50",shape = (224,224),transforms=None,num_type="FP32",url='127.0.0.1:8000')
#         return res
    
class async_clsHandler(tornado.web.RequestHandler):

    async def post(self):
        data = json.loads(self.request.body)
        img_base64_str = data["base64_str"]
        # img_base64_str = self.get_argument("base64_str", None)
        # print(img_base64_str)
        result = await self.doing(img_base64_str)
        self.write(result)

    async def doing(self,img_base64_str):
        # await asyncio.sleep(10)
        res = await async_cls_process(img_base64_str, model_name = "resnet_50",shape = (224,224),transforms=None,num_type="FP32",url="yyzz_cls_triton:8000")
        return res
        

def activeth():
    while True:
        print(threading.active_count())
        time.sleep(1)

requestHandlers=[
                 (r'/cls',async_clsHandler),
                #  (r'/MutiThreadcls',MutiThread_clsHandler),
                 (r'/',MainHandler)
                 ]


if __name__ == "__main__":
    
    # # 监听现成数量
    # acc = threading.Thread(target = activeth)
    # acc.start()

    # 启动
    app=tornado.web.Application(requestHandlers)
    app.listen(9998)
    tornado.ioloop.IOLoop.instance().start()