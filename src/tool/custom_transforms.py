from torchvision import transforms

def maxLenPad(img):
        # 找到长边
        height, width = img.shape[1], img.shape[2]
        maxlen = max([width,height])

        # 对width pad
        if width < maxlen:
            # 计算上下需要pad的数量，padding
            value = maxlen-width
            left = value/2 if value%2==0 else value//2
            right = value/2 if value%2==0 else value//2
            img = transforms.Pad([int(left),0,int(right),0],0,padding_mode="constant")(img)

        # 对height pad
        if height < maxlen:
            # 计算上下需要pad的数量，padding
            value = maxlen - height
            up = value/2 if value%2==0 else value//2
            down = value/2 if value%2==0 else value//2
            img = transforms.Pad([0,int(up),0,int(down)],0,padding_mode="constant")(img)
        
        return img