import warnings
from myutils import *
# 忽略所有警告
warnings.filterwarnings("ignore")
def preimg(framenum,rnum,prepath,datapath,add='.tif'):
    for idx in range(framenum[0],framenum[1]):#frame数
        for ir in range(rnum-1,rnum):#轮次数
            pmake(f'{prepath}R{ir+1}/ch1/1/')
            imx = cv2.imread(f'{datapath}F{idx}R{ir+1}ch1'+add, -1)
            # imx=imadjust(imx)
            imx=gaug(imx,1.2,10)
            cv2.imwrite(f'{prepath}R{ir+1}/ch1/1/frame{idx}.png', imx)
if __name__=="__main__":
    imname="IM418xxD_dapi4Regi/img/"
    datapath="./regidata/Re_"+imname#输入图片，要求含有名称为FxRxCh1.png的dapi通道图
    prepath="./predata/"+imname#将不同区域的daipi图片放在一起作为中间图片，为了拼接做准备
    outpath = "./result/"+imname#输出结果地址，数据批次名称下面为Rx\ch1\，含有相对、绝对位置信息及结果图
    pmake(prepath)
    filename = natsorted(os.listdir(datapath))
    framenum=[1,27]##输入一个列表，代表区域数目，比如[1,10]代表计算1到9的区域
    rnum=1#输入轮次数,由于拼接只需要计算第一轮，其余轮次不重要，建议保持为1
    preimg(framenum,rnum,prepath,datapath,add='.png')#转换中间图片函数
    stitchWithFeature(prepath,outpath,stre=0.7,calr='1')#拼接函数及匹配阈值（保留小于此汉明距离的点）及计算轮次
