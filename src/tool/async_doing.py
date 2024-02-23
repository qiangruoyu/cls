import time
import asyncio

async def doing(name):
    """
    await中必须为可等待对象： 协程对象， Future，Task对象  可理解为IO等待
    函数里面有future对象才能实现异步，如果没有就不能实现异步

    """
    result = await asyncio.sleep(10) # await 后是Future对象，可实现异步并行。
    time.sleep(3) # 不是Future对象，这部分不能实现异步并行。
    print(name+"doing")

async def interface(name):
    print(name + "begin")
    result = await doing(name) # await 后是协程对象，能否实现异步并行取决于doing函数里面的值。
    print(name + "end")

async def work():
    """
    并行运行协程函数：可等待对象若是协程对象则变成串行，若是Task对象则并发运行
    """

    # 方式一：gather
    # tasks = [asyncio.ensure_future(interface(str(i))) for i in range(2)]
    # await asyncio.gather(*tasks)

    # 方式二：使用python3.7版本后的create_task
    # 该种方式asyncio 仅仅会保留对 Task 的“弱引用”（weakref）。而弱引用与我们熟知的强引用（如：赋值 a=1，列表、集合等容器内包含 [1, 2], {1, 2}）有一个重要的不同，
    # 就是：弱引用不会阻止对象被 Python 的垃圾回收机制回收。也就是说，一个未完成的，甚至是正在运行的 Task，有可能被垃圾回收中断并且清除。
    # 这不仅会让你的后台任务意外终止，还有可能影响后台任务的资源回收（因为在被垃圾收集时，代码可以被运行在任意上下文中）。
    # 为了避免这些复杂后果，我们应该做的是，不要直接使用 create_task(task()) 创建后台任务。
    # task1 = asyncio.create_task(interface(str(1)))
    # task2 = asyncio.create_task(interface(str(2)))
    # await task1
    # await task2

    # 方式三：使用ensure_future函数降函数变为
    task1 = asyncio.ensure_future(interface(str(1)))
    task2 = asyncio.ensure_future(interface(str(2)))
    await task1 # await 后是task
    await task2


if __name__ == "__main__":

    # python3.7之前写法
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(work())

    # python 3.7+
    asyncio.run(work())

    
