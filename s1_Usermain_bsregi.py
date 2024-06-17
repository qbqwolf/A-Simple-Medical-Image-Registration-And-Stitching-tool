from myutils import *
import warnings
# 忽略所有警告
warnings.filterwarnings("ignore")
params = pyelastix.get_default_params(type='BSPLINE')
params.NumberOfResolutions = 5##迭代次数
print(params)
if __name__=="__main__":
    imgname = 'IM41806-12um/'
    inbase = './raw_data/' + imgname  # 数据基准地址
    outbase = './regidata/BSRe_' + imgname  # 输出基准地址
    superworkdir = './super_workdir/'
    pmake(f'{outbase}img/')
    pmake(f'{outbase}vgrid/')
    hmode = 0  ###Hmode，0为完整H矩阵，1为只平移,默认为0
    framenum=[5,6]##输入一个列表，代表区域数目，比如[1,10]代表计算1到9的区域
    rnum=6#输入轮次数

    img = cv2.imread(f'{inbase}F1R1Ch1.tif', -1)
    h=img.shape[0]
    warp = Warper2d(img_size=h)
    mulfra_mulr_1r_bsregist(framenum,rnum,inbase,outbase,params,warp,add='.tif')