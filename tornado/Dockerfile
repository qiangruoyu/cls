# 使用Python 3.7作为基础镜像  打包web服务tornado镜像，用于预处理和后处理 
FROM python:3.7  
  
# 设置工作目录  
WORKDIR /home 

# 设置环境变量
ENV PYTHONPATH /home/
  
# 复制当前目录中的requirements.txt到容器的/app中  
COPY requirements.txt .  
  
# 使用pip安装依赖  
RUN pip3 install --no-cache-dir -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com


# 复制当前目录中的所有文件到容器的/app中  
# COPY . .  
  
# 暴露端口（如果需要的话）  
# EXPOSE <port>  
  
# 设置容器的启动命令  
CMD ["python", "tornado/app.py"]