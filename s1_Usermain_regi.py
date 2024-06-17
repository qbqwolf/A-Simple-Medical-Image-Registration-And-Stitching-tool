from myutils import *
import warnings
# 忽略所有警告
warnings.filterwarnings("ignore")

if __name__=="__main__":
    imgname = 'IM418xxD_dapi4Regi/'#数据批次名称
    inbase = './raw_data/'+imgname # 数据基准地址。图片格式请写为类似F2R1Ch1.png形式，配准仅采用第一通道
    outbase = './regidata/Re_' + imgname  # 输出基准地址。文件储存在该目录下，分为csv与img
    pmake(f'{outbase}img/')
    pmake(f'{outbase}csv/')
    # pmake(f'{outbase}check/')
    hmode = 0  ###Hmode，0为完整H矩阵，1为只平移,默认为0
    framenum=[1,27]##输入一个列表，代表区域数目，比如[1,10]代表计算1到9的区域
    rnum=6#输入轮次数
    mulfra_mulr_1r_regist(framenum,rnum,inbase,outbase,[1,3],hmode,add='.tif',dis=85)