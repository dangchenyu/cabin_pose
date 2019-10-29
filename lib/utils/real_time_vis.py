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


def main(path, smooth_frames, label_path, thres):
    if 'mp4' in path:
        cam = cv2.VideoCapture(path)
        joints_queue = []
        i = 0
        while True:
            _, origin = cam.read()
            if not _:
                break
            input, lt, rb = get_patch(origin, label_path[i])
            if i%2==0:

                pred, _ = get_pred(input)
                last_pred=pred.copy()
                if len(joints_queue) < smooth_frames:
                    joints_queue.append(pred)
                else:
                    joints_queue = update_queue(joints_queue, pred)

                show_result(input, origin, lt, rb, joints_queue, pred, _, thres,i)

            else:
                show_result(input, origin, lt, rb, joints_queue, last_pred, _, thres,i)
            i += 1
    else:
        img_list = os.listdir(path)

        for img in img_list:
            img_data = cv2.imread(path + img, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            show_result(img_data, img_data, [0, 0], [0, 0])


def get_patch(img, label_path):
    label_path_list = label_path.split()
    patch = img[int(float(label_path_list[2])):int(float(label_path_list[4])),
            int(float(label_path_list[1])):int(float(label_path_list[3]))]
    return patch, [int(float(label_path_list[1])), int(float(label_path_list[2]))], [int(float(label_path_list[3])),
                                                                                     int(float(label_path_list[4]))]


def get_pred(input):
    img_data = cv2.resize(input, tuple(cfg.MODEL.IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)

    # normalize = transforms.Normalize(
    #
    #     mean=[0.485], std=[0.229]
    #
    # )
    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
    img_data = (img_data / 255).astype(np.float32)
    transform = transforms.Compose([
        transforms.ToTensor(),

    ])
    input = transform(img_data)
    input = torch.unsqueeze(input, 0)
    output = model(input)
    pred, _ = get_max_preds(output.detach().cpu().numpy())
    return pred, _


def update_queue(joints_queue, new):
    joints_queue.pop(0)
    joints_queue.append(new)
    return joints_queue


def get_ava(joints_queue, lt, scale):
    add_queue = np.array([0])
    for item in joints_queue:
        item = (item * scale).astype(int)
        item = np.add(item, lt)
        add_queue = np.add(add_queue, item)
    joints_ava = add_queue / len(joints_queue)
    return joints_ava


def get_new(joints_ava, pred, thres):
    distance = np.sqrt(np.square(joints_ava - pred).sum(2))
    wrong_ind = np.where(distance > thres)
    pred[wrong_ind] = joints_ava[wrong_ind]
    return pred


def show_result(img_data, origin, left_top, right_bottom, joints_queue, pred, _, thres, frame_num,if_origin=True):
    show = origin.copy()
    size_x = img_data.shape[1]
    size_y = img_data.shape[0]
    scale_x = size_x / cfg.MODEL.IMAGE_SIZE[1] * 4
    scale_y = size_y / cfg.MODEL.IMAGE_SIZE[0] * 4
    white = np.zeros([192, 192, 3])
    white[:, :, 0] = 0
    white[:, :, 1] = 0
    white[:, :, 2] = 0
    joints_ava = get_ava(joints_queue, left_top, [scale_x, scale_y])
    pred = (pred * [scale_x, scale_y]).astype(int)
    pred = np.add(pred, np.array(left_top))
    pred = get_new(joints_ava, pred, thres)
    mid_29 = ((pred[0][2] + pred[0][9]) / 2).astype(int)
    mid_56 = ((pred[0][5] + pred[0][6]) / 2).astype(int)
    mid_01 = ((pred[0][0] + pred[0][1]) / 2).astype(int)
    if if_origin:
        show = show
    else:
        show = white
    # cv2.line(show,tuple(pred[0][0]),tuple(pred[0][1]),(0,255,0),thickness=4)
    # cv2.circle(show,tuple(mid_01),25,(0,255,0),4)
    cv2.rectangle(show, tuple(left_top), tuple(right_bottom), (0, 255, 0), 2)
    cv2.line(show, tuple(pred[0][2]), tuple(pred[0][3]), (255, 255, 0), thickness=4)
    cv2.line(show, tuple(pred[0][3]), tuple(pred[0][4]), (0, 255, 255), thickness=4)
    cv2.line(show, tuple(pred[0][9]), tuple(pred[0][8]), (255, 0, 0), thickness=4)
    cv2.line(show, tuple(pred[0][8]), tuple(pred[0][7]), (0, 0, 255), thickness=4)
    # cv2.line(show, tuple(pred[0][1]), tuple(pred[0][5]), (255, 255, 255), thickness=4)
    # cv2.line(show, tuple(pred[0][1]), tuple(pred[0][6]), (255, 0, 255), thickness=4)
    cv2.line(show, tuple(pred[0][2]), tuple(pred[0][9]), (255, 0, 255), thickness=4)
    cv2.line(show, tuple(mid_29), tuple(mid_56), (255, 255, 255), thickness=4)
    cv2.line(show, tuple(pred[0][5]), tuple(pred[0][6]), (100, 100, 100), thickness=4)
    cv2.line(show, tuple(mid_01), tuple(mid_29), (100, 100, 100), thickness=4)
    cv2.imwrite("D:\\cabin_test\\internel_1\\{:05d}.jpg".format(frame_num),show)
    cv2.imshow('test', show)
    cv2.waitKey(1)


if __name__ == '__main__':
    args = parse_args()
    update_config(cfg, args)
    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=False
    )
    checkpoint = torch.load(
        'D:\\cabin_test\\ckpts\\8.pth',
        map_location='cpu')
    if list(checkpoint['state_dict'].keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    with open("C:\\Users\\DELL\\Desktop\\#4299框结果.txt") as o:
        label_list = o.readlines()

    o.close()
    # main('D:\cabin_test\\back\\',label_list)
    main('D:\\cabin_test\\test.mp4', 5, label_list,10)
