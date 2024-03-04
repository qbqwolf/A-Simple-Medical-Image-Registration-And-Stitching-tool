import numpy as np
import cv2
import os
from PIL import Image,ImageEnhance
import glob
from natsort import natsorted

def imgnorm(img):
    img_float32 = np.float32(img)
    img_NORM_MINMAX = img_float32.copy()
    cv2.normalize(img_float32,img_NORM_MINMAX,0,255,cv2.NORM_MINMAX)
    img_norm=img_NORM_MINMAX.astype('uint8')
    return img_norm

def gaug(src,br,ctr):

    image = Image.fromarray(cv2.cvtColor(src,cv2.COLOR_GRAY2RGB))
    enh_bri = ImageEnhance.Brightness(image)
    brightness = br
    image_brightened = enh_bri.enhance(brightness)
    enh_col = ImageEnhance.Color(image_brightened)
    color = 1
    image_colored = enh_col.enhance(color)
    enh_con = ImageEnhance.Contrast(image_colored)
    contrast = ctr
    image_contrasted = enh_con.enhance(contrast)
    dst=np.asarray(image_contrasted)

    #out=cv2.pyrMeanShiftFiltering(dst,10,10)
    out=cv2.medianBlur(dst, 5)
    # out=cv2.GaussianBlur(dst, (11,11), 0)
    #out = cv2.Canny(out, 100, 255)
    # ret, out = cv2.threshold(out,150, 255, 0)
    #out=cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    return out

def pmake(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)
def tran16to8(image_path):
    image_16bit = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    min_16bit = 0
    max_16bit = 3000
    image_8bit = np.array(np.rint(255 * ((image_16bit - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)
    return image_8bit
ipath='./raw_data/1119_4f_15g_stitch_beads/IM41190/IM41190-TIF/'
opath='./regidata/IM41190/'

#将图片合成为一张
for idx in range(0,4):#遍历每个R
    filename=natsorted(os.listdir(f'{ipath}R{idx+1}/'))
    numfra=int(filename[0][0])-1
    for fra in filename:#遍历frame
        numfra+=1
        br=1
        ctr=10
        numsubf1=0
        numsubf2=0
        numsubf3=0
        numsubf4=0
        subfile=natsorted(os.listdir(f'{ipath}R{idx+1}/'+fra))
        for imi in subfile:#遍历通道1子frame
            tiflag=imi.endswith('tif')
            if not tiflag:
                continue
            channel=imi[-5]
            if channel=='1':
                numsubf1+=1
                if numsubf1==1:
                    img1 = tran16to8(f'{ipath}R{idx+1}/{fra}/{imi}')
                    imgs1=img1
                else:
                    img1 = tran16to8(f'{ipath}R{idx + 1}/{fra}/{imi}')
                    imgs1 = cv2.merge([imgs1, img1])
            elif channel == '2':
                numsubf2 += 1
                if numsubf2 == 1:
                    img2 = tran16to8(f'{ipath}R{idx + 1}/{fra}/{imi}')
                    imgs2 = img2
                else:
                    img2 = tran16to8(f'{ipath}R{idx + 1}/{fra}/{imi}')
                    imgs2 = cv2.merge([imgs2, img2])
            elif channel == '3':
                numsubf3 += 1
                if numsubf3 == 1:
                    img3 = tran16to8(f'{ipath}R{idx + 1}/{fra}/{imi}')
                    imgs3 = img3
                else:
                    img3 = tran16to8(f'{ipath}R{idx + 1}/{fra}/{imi}')
                    imgs3 = cv2.merge([imgs3, img3])
            elif channel == '4':
                numsubf4 += 1
                if numsubf4 == 1:
                    img4 = tran16to8(f'{ipath}R{idx + 1}/{fra}/{imi}')
                    imgs4 = img4
                else:
                    img4 = tran16to8(f'{ipath}R{idx + 1}/{fra}/{imi}')
                    imgs4 = cv2.merge([imgs4, img4])
        pmake(f'{opath}R{idx + 1}/frame{numfra}/')  ##创建地址
        imgs1 = np.max(imgs1, axis=2)
        imgs1a = gaug(imgnorm(imgs1), br, ctr)
        cv2.imwrite(f'{opath}R{idx + 1}/frame{numfra}/ch1.png', imgs1)  # 存储细胞核图片
        cv2.imwrite(f'{opath}R{idx + 1}/frame{numfra}/ch1a.png', imgs1a)  # 存储细胞核图片
        imgs2 = np.max(imgs2, axis=2)
        cv2.imwrite(f'{opath}R{idx + 1}/frame{numfra}/ch2.png', imgs2)  #
        # if idx>=0:
        imgs3 = np.max(imgs3, axis=2)
        cv2.imwrite(f'{opath}R{idx + 1}/frame{numfra}/ch3.png', imgs3)  # 存储荧光点图片
        imgs4 = np.max(imgs4, axis=2)
        cv2.imwrite(f'{opath}R{idx + 1}/frame{numfra}/ch4.png', imgs4)  # 存储荧光点图片