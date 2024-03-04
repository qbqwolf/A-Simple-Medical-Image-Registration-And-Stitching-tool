from superutils import *
from SuperPointPretrainedNetwork.demo_superpoint import *
import cv2



def superfeature(path,h,w,ima,imb):
    input=path
    camid=0
    H=h
    W=w
    skip=1
    img_glob='*.png'
    no_display=True
    write=False
    write_dir='tracker_outputs/'
    vs = VideoStreamer(input, camid, H, W, skip, img_glob)
    print('==> Loading pre-trained network.')
    fe = SuperPointFrontend(weights_path='./SuperPointPretrainedNetwork/superpoint_v1.pth',  # 权重的路径str
                            nms_dist=4,  # 非极大值抑制 int距离4
                            conf_thresh=0.055,  # 探测器阈值0.015，越低探测结果越多
                            nn_thresh=0.3,  # 匹配器阈值0.7,越高越难匹配
                            cuda=False)  # GPU加速 默认false
    print('==> Successfully loaded pre-trained network.')
    # Create a window to display the demo.
    if not no_display:
        win = 'SuperPoint Tracker'
        cv2.namedWindow(win)
    else:
        print('Skipping visualization, will not show a GUI.')
    # Font parameters for visualizaton.

    # 创建输出目录
    if write:  # 默认false
        print('==> Will write outputs to %s' % write_dir)
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)
    print('==> Running Demo.')
    img1, status = vs.next_frame()  # 读第一张图
    start1 = time.time()

    pts, desc, heatmap = fe.run(img1)

    end1 = time.time()
    c2 = end1 - start1
    print("第一张图提取用时", c2, "提取特征点数目", pts.shape[1])
    imgx = ima.astype(np.float32)
    img11 = showpoint(imgx, pts)
    #cv2.imshow("imgone", img11)
    
    img2, status = vs.next_frame()  # 读第二张图
    start1 = time.time()
    pts1, desc1, heatmap1 = fe.run(img2)

    end1 = time.time()
    c2 = end1 - start1
    print("第二张图提取用时", c2, "提取特征点数目", pts1.shape[1])
    imgy = imb.astype(np.float32)
    img22 = showpoint(imgy, pts1)
    #cv2.imshow("imgtwo", img22)


    match = nn_match_two_way(desc, desc1, 0.4)
    #match = knn_match(desc, desc1,1)
    
####################最后一个参数为像素距离约束#############################
    pto,pto1,match=match_descriptors(pts, pts1, match,105)

    print("图1与图2匹配对数", match.shape[1])
    out = drawMatches(imgx, pts, imgy,
                       pts1, match)
    #cv2.namedWindow("matcher", 0)
    #cv2.imshow("matcher", out)

    #cv2.waitKey(0)

    print('==> Finshed Demo.')
    return out,img11,img22,pto,pto1
if __name__ == '__main__':
    imagin=['/home/yinghua/mmdetection/ver1_codes(2D)/testinput/']
    imagou=['/home/yinghua/mmdetection/ver1_codes(2D)/testvisual/']
    tim=cv2.imread(f'{imagin[0]}test1.png',0)
    h=tim.shape[0]
    w=tim.shape[1]
    tim1=cv2.imread(f'{imagin[0]}test1.png',0)
    tim2=cv2.imread(f'{imagin[0]}test2.png',0)
    imo,im1,im2,po1,po2=superfeature(imagin[0],h,w,tim1,tim2)
    cv2.imwrite(f'{imagou[0]}out.png',imo)
    cv2.imwrite(f'{imagou[0]}img1.png',im1)
    cv2.imwrite(f'{imagou[0]}img2.png',im2)