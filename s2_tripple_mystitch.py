import copy
import numpy as np
import re
from Stitcher import Stitcher
import os
from natsort import natsorted
import cv2
from super import superfeature
import glob
def pmake(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)
def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)
def stitchWithFeature(inpath,outpath):
    Stitcher.featureMethod = "super"             # "sift","surf" or "orb","super"
    Stitcher.isColorMode = False                 # True:color, False: gray
    Stitcher.isGPUAvailable = False
    # Stitcher.isEnhance = True
    # Stitcher.isClahe = True
    Stitcher.superthresh=0.7                     #superpoint match thresh
    Stitcher.searchRatio = 0.75                 # 0.75 is common value for matches
    Stitcher.offsetCaculate = "mode"            # "mode" or "ransac"
    Stitcher.offsetEvaluate = 3                 # 3 menas nums of matches for mode, 3.0 menas  of matches for ransac
    Stitcher.roiRatio = 0.2                     # roi length for stitching in first direction
    Stitcher.fuseMethod = "fadeInAndFadeOut"    # "notFuse","average","maximum","minimum","fadeInAndFadeOut","trigonometric", "multiBandBlending"
    Stitcher.direction = 2;  Stitcher.directIncre = -1;
    Stitcher.ishandle = True                    #是否手动拼接
    stitcher = Stitcher()

    filename = natsorted(os.listdir(inpath))
    Rr=0
    Rch=0
    for r in filename:
        chfile = natsorted(os.listdir(inpath+"/" + r))
        for ch in chfile:
            if ch[-1] == 'a'and r[-1]=='1':
                stitcher.imageSetStitchWithMutiple(f'{inpath}{r}/{ch}', f'{outpath}{r}/{ch}', 1,
                                                                        stitcher.calculateOffsetForFeatureSearchIncre,
                                                                        startNum=1, fileExtension="png",
                                                                        outputfileExtension="png",r=r,ch=ch,outpath=outpath)
                Rr=r
                Rch=ch
        for ch in chfile:
            stnum=0
            endnum=0
            dpath=[]
            for filename in os.listdir(f'{outpath}{Rr}/{Rch}'):##检查dxdy文件数量
                # 检查文件名是否包含'dxdy'并且以'.txt'结尾
                if 'dxdy' in filename and filename.endswith('.txt'):
                    # 拼接完整文件路径
                    dpath.append(os.path.join(f'{outpath}{Rr}/{Rch}', filename))
            for lix in range(0,len(dpath)):
                offsetlist = []
                with open(dpath[lix], 'r') as file:
                    for line in file:
                        # 去除每行末尾的换行符，并添加到列表
                        parts = line.strip().split()
                        # 将分割后的部分转换为整数
                        int_parts = [int(part) for part in parts]
                        # 添加整数列表到 offsetlist
                        offsetlist.append(int_parts)
                # offsetlist=[int(item) for item in offsetlist]
                endnum= int(re.search(r'dxdy(\d+)', dpath[lix]).group(1))
                pmake(f'{outpath}{r}/{ch}')
                fileAddress = f'{inpath}{r}/{ch}' + "\\1" + "\\"
                fileList = natsorted(glob.glob(fileAddress + "*.png"))
                selected_files = fileList[stnum:endnum]
                stitimage, _ = stitcher.getStitchByOffset(selected_files, offsetlist)
                cv2.imwrite(f'{outpath}{r}/{ch}' + f"\\stitching_result{stnum+1}-{endnum}" + ".png", stitimage)
                stnum+=endnum

if __name__=="__main__":
    imname="IM41190_tri/"
    datapath="./regidata/"+imname
    prepath="./predata/"+imname
    outpath = "./result/"+imname
    for iz in range(0,3):
        pmake(prepath+f'Z{iz+1}/')
        filename = natsorted(os.listdir(datapath+f'Z{iz+1}/'))
        filename = [name for name in filename if name.endswith('_regi')]
        for r in filename:#遍历R
            rx=int(r[1])
            frfile = natsorted(os.listdir(datapath +f'Z{iz+1}/'+ r))
            for fr in frfile:#遍历frame
                frx = int(fr[5:])
                imifile = natsorted(os.listdir(datapath +f'Z{iz+1}/'+ r+'/'+fr))
                for imi in imifile:
                    if imi[-5] == '1':
                        pmake(f'{prepath}Z{iz + 1}/R{rx}/ch1a/1/')
                        imx = cv2.imread(f'./regidata/IM41190/{r}/{fr}/{imi}', -1)
                        cv2.imwrite(f'{prepath}Z{iz + 1}/R{rx}/ch1a/1/frame{frx}.png', imx)
                        pmake(f'{prepath}Z{iz+1}/R{rx}/ch1/1/')
                        imx = cv2.imread(f'{datapath}Z{iz+1}/{r}/{fr}/{imi}', -1)
                        cv2.imwrite(f'{prepath}Z{iz+1}/R{rx}/ch1/1/frame{frx}.png', imx)
                    if imi[-5] == '2':
                        pmake(f'{prepath}Z{iz+1}/R{rx}/ch2/1/')
                        imx = cv2.imread(f'{datapath}Z{iz+1}/{r}/{fr}/{imi}', -1)
                        cv2.imwrite(f'{prepath}Z{iz+1}/R{rx}/ch2/1/frame{frx}.png', imx)
                    if imi[-5] == '3':
                        pmake(f'{prepath}Z{iz+1}/R{rx}/ch3/1/')
                        imx = cv2.imread(f'{datapath}Z{iz+1}/{r}/{fr}/{imi}', -1)
                        cv2.imwrite(f'{prepath}Z{iz+1}/R{rx}/ch3/1/frame{frx}.png', imx)
                    if imi[-5] == '4':
                        pmake(f'{prepath}Z{iz+1}/R{rx}/ch4/1/')
                        imx = cv2.imread(f'{datapath}Z{iz+1}/{r}/{fr}/{imi}', -1)
                        cv2.imwrite(f'{prepath}Z{iz+1}/R{rx}/ch4/1/frame{frx}.png', imx)
        stitchWithFeature(prepath+f'Z{iz+1}/',outpath+f'Z{iz+1}/')
