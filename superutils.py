import torch
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import cv2
def myimshow(img, title=False,fname="test.jpg",size=6):
    fig = plt.figure(figsize=(size,size))
    if title:
        plt.title(title)
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.xticks(())
    plt.yticks(())
    plt.savefig(fname, bbox_inches='tight')
    plt.show()
    
def myimshows(imgs, titles=False, fname="test.jpg", size=6):
    lens = len(imgs)
    fig = plt.figure(figsize=(size * lens,size))
    if titles == False:
        titles="0123456789"
    for i in range(1, lens + 1):
        cols = 100 + lens * 10 + i
        plt.xticks(())
        plt.yticks(())
        plt.subplot(cols)
        if len(imgs[i - 1].shape) == 2:
            plt.imshow(imgs[i - 1], cmap='Reds')
        else:
            plt.imshow(imgs[i - 1])
        plt.title(titles[i - 1])
    plt.xticks(())
    plt.yticks(())
    plt.savefig(fname, bbox_inches='tight')
    plt.show()
    
def myimshowsCL(imgs, titles=False,rows=4,cols=4, size=6):
    lens = len(imgs)
    if titles == False:
        titles="0123456789012345678901234567890123456789"
    plt.figure(figsize=(cols*size,rows*size))
    for i in range(1, lens + 1):
        plt.xticks(())
        plt.yticks(())
        plt.subplot(rows, cols, i)
        if len(imgs[i - 1].shape) == 2:
            plt.imshow(imgs[i - 1], cmap='Reds')
        else:
            plt.imshow(imgs[i - 1])
        plt.title(titles[i - 1])
    plt.xticks(())
    plt.yticks(())
    #plt.savefig(fname, bbox_inches='tight')
    plt.show()
    
def get_cmp():
    colors = [(255,255, 255), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
              (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
              (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
              (128, 64, 12)]
    colors=np.array(colors).astype(np.uint8)
    #防止label值存在溢出，将未定义的label值颜色设置为0,0,0
    cmp=np.zeros((256,3)).astype(np.uint8)
    cmp[:colors.shape[0]]=colors
    return cmp
 
#将label转换为RGB图像
def label2color(img):
    if len(img.shape)==3:
        if img.shape[0]==1:
            img=img[0]
        elif img.shape[2]==1:
            img=img[:,:,0]
        else:
            return img
    if len(img.shape)==2:
        #核心代码
        cmap=get_cmp()
        img=cmap[img]
        return img
    
#这里可以导入自己的AI框架    

def read_img_as_tensor(path):
    im=Image.open(path)
    im=im.resize((min(im.size[0],im.size[1]),min(im.size[0],im.size[1])))
    img=np.array(im)
    if img.shape[2]==3:
        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if len(img.shape)==2:
        np_arr=img.reshape((1,1,img.shape[0],img.shape[1]))
    else:
        img=img.transpose((2,0,1))
        np_arr=img.reshape((1,*img.shape))
    #tensor=torch.from_numpy(np_arr).to("cuda").to(torch.float32)
    tensor=torch.tensor(np_arr).to("cuda").to(torch.float32)
    #tensor=paddle.to_tensor(np_arr,paddle.float32)
    return tensor,img#[:,:,:3]
 
def tensor2img(tensor):
    np_arr=tensor.to("cpu").numpy()[0]
    if np_arr.shape[0]==1:
        np_arr=np_arr[0]
    else:
        np_arr=np_arr.transpose((1,2,0))
        
    if np_arr.max()>=1:
        np_arr=np_arr.astype(np.uint8)
    return np_arr
#把大图裁剪为小图
def big_img2small_crop(img,size_old=(736,736),pad=0):
    img_list=[]
    img_size=img.shape
    loc=[]
    size=(size_old[0]-2*pad,size_old[1]-2*pad)
    for i in range(pad,img.shape[0]-pad,size[0]):
        for j in range(pad,img.shape[1]-pad,size[1]):
            tmp=img[i-pad:i+size[0]+pad,j-pad:j+size[1]+pad]
            if tmp.shape[0]!=size_old[0] or tmp.shape[1]!=size_old[1]:
                if tmp.shape[0]!=size_old[0]:
                    i=img.shape[0]-size[0]-pad
                if tmp.shape[1]!=size_old[1]:
                    j=img.shape[1]-size[1]-pad
                tmp=img[i-pad:i+size[0]+pad,j-pad:j+size[1]+pad] #256
            img_list.append(tmp)
            loc.append((i-pad,j-pad,*size))
    return img_list,img_size,loc
 
#把裁剪的小图拼接为大图
def small_crop2big_img(img_list,img_size,loc):
    result=np.zeros(img_size,np.uint8)
    index=0
    for i,j,w,h in loc:
        result[i:i+w,j:j+h]=img_list[index]
        index+=1
    return result