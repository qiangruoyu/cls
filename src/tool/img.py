from sklearn.cluster import KMeans
import numpy as np
import os
from PIL import Image
import base64
import io


def imgTbase64(input_location):
    with open(input_location,"rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_string

def base64Toimg(base64_data):
    # if type(base64_data) == str:
        # base64_data = base64_data.encode()
    img_b64decode = base64.b64decode(base64_data)
    byte_stream = io.BytesIO(img_b64decode)
    img = Image.open(byte_stream)
    return img

def get_size(directory_list):
    """
    通过图片聚类求得模型输入的合适大小
    """
    shape_list = []
    for dir in directory_list:
        files = os.listdir(dir)
        for file in files:
            file_path = os.path.join(dir,file)
            shape = list(Image.open(file_path).size)
            shape_list.append(shape)
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(shape_list)
    labels = kmeans.labels_
    cluster_counts = np.bincount(labels)
    print("每个簇中的点数：", cluster_counts)
    return kmeans.cluster_centers_, cluster_counts

    
if __name__ == "__main__":
    directory_list = ["/home/qiangyu/cls/data/imagenet/ants"]
    print(get_size(directory_list))