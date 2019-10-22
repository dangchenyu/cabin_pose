from config import cfg
from config import update_config
import torch
import argparse
import os
import cv2
import torchvision.transforms as transforms
from core.inference import get_max_preds
import numpy as np
import models
def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')
    parser.add_argument('--device',
                        help='choose cpu or cuda',
                        type=str,
                        default='cuda')

    args = parser.parse_args()

    return args
def main():

    args = parse_args()
    update_config(cfg, args)
    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=False
    )
    checkpoint = torch.load('C:\\Users\\DELL\\PycharmProjects\\deep-high-resolution-net.pytorch\\tools\\output\\mpii\\pose_hrnet\\w32_256x256_adam_lr1e-3\\.pth' ,
                            map_location='cpu')
    if list(checkpoint['state_dict'].keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    test_img_path="C:\\Users\\DELL\\Desktop\\front\\"
    img_list=os.listdir(test_img_path)
    origin=True
    for img in img_list:
        img_data=cv2.imread(test_img_path+img,cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        show=img_data.copy()

        size_x=img_data.shape[1]
        size_y=img_data.shape[0]
        scale_x=size_x/cfg.MODEL.IMAGE_SIZE[1]*4
        scale_y = size_y / cfg.MODEL.IMAGE_SIZE[0]*4
        img_data=cv2.resize(img_data,tuple(cfg.MODEL.IMAGE_SIZE),interpolation=cv2.INTER_CUBIC)
        white=np.zeros([192,192,3])
        white[:, :, 0] = 0
        white[:, :, 1] = 0
        white[:, :, 2] = 0
        normalize = transforms.Normalize(
            mean=[0.485], std=[0.229]
        )
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        img_data=cv2.cvtColor(img_data,cv2.COLOR_BGR2GRAY)
        input=transform(img_data)
        input=torch.unsqueeze(input,0)
        output=model(input)
        pred,_=get_max_preds(output.detach().cpu().numpy())
        pred=(pred*[scale_x,scale_y]).astype(int)


        mid_29=((pred[0][2]+pred[0][9])/2).astype(int)
        mid_56=((pred[0][5]+pred[0][6])/2).astype(int)
        mid_01=((pred[0][0]+pred[0][1])/2).astype(int)
        if origin:
            show=show
        else:
            show=white
        # cv2.line(show,tuple(pred[0][0]),tuple(pred[0][1]),(0,255,0),thickness=4)
        # cv2.circle(show,tuple(mid_01),25,(0,255,0),4)
        cv2.line(show,tuple(pred[0][2]),tuple(pred[0][3]),(255,255,0),thickness=4)
        cv2.line(show,tuple(pred[0][3]),tuple(pred[0][4]),(0,255,255),thickness=4)
        cv2.line(show,tuple(pred[0][9]),tuple(pred[0][8]),(255,0,0),thickness=4)
        cv2.line(show,tuple(pred[0][8]),tuple(pred[0][7]),(0,0,255),thickness=4)
        # cv2.line(show, tuple(pred[0][1]), tuple(pred[0][5]), (255, 255, 255), thickness=4)
        # cv2.line(show, tuple(pred[0][1]), tuple(pred[0][6]), (255, 0, 255), thickness=4)
        cv2.line(show, tuple(pred[0][2]), tuple(pred[0][9]), (255, 0, 255), thickness=4)
        cv2.line(show,tuple(mid_29),tuple(mid_56),(255,255,255),thickness=4)
        cv2.line(show, tuple(pred[0][5]), tuple(pred[0][6]), (100, 100, 100), thickness=4)
        cv2.line(show, tuple(mid_01), tuple(mid_29), (100, 100, 100), thickness=4)

        cv2.imshow('test',show)
        cv2.waitKey()




if __name__ == '__main__':
    main()
