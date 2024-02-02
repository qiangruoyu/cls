from sklearn.cluster import KMeans
import numpy as np
import os
from PIL import Image

def get_size(directory_list):

    shape_list = []
    for dir in directory_list:
        files = os.listdir(dir)
        for file in files:
            file_path = os.path.join(dir,file)
            shape = list(Image.open(file_path).size)
            shape_list.append(shape)
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(shape_list)
    return kmeans.cluster_centers_

    
if __name__ == "__main__":
    directory_list = ["/home/qiangyu/cls/data/imagenet/ants"]
    print(get_size(directory_list))