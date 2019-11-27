import torch
import cv2
from core.inference import get_max_preds
import numpy as np


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals
def
    if len(pose_joints_queue) < smooth_frames:
        pose_joints_queue.append(pred)
    else:
        joints_queue = update_queue(pose_joints_queue, pred)
def _update_queue(joints_queue, new):
    joints_queue.pop(0)
    joints_queue.append(new)
    return joints_queue
class ShowResult:
    def __init__(self,bounding_box,  result_caffemodel, smooth_frames, pose_joints_queue, thres, frame_num):
        self.result_caffemodel=result_caffemodel
        self.bounding_box=bounding_box
        self.smooth_freams=smooth_frames




    def _get_ava(self,joints_queue, lt, scale):
        add_queue = np.array([0])
        for item in self.joints_queue:
            item = (item * scale).astype(int)
            item = np.add(item, lt)
            add_queue = np.add(add_queue, item)
        joints_ava = add_queue / len(joints_queue)
        return joints_ava

    def _get_new(self,joints_ava, pred, thres):
        distance = np.sqrt(np.square(joints_ava - pred).sum(2))
        wrong_ind = np.where(distance > thres)
        pred[wrong_ind] = joints_ava[wrong_ind]
        return pred

    def __call__(self, *args, **kwargs):


def show_resulst(result_caffemodel, bounding_box, smooth_frames, pose_joints_queue, img_data, origin, left_top,
                 right_bottom, joints_queue, pred, _, thres, frame_num):
    pred, _ = get_max_preds(result_caffemodel)
    last_pred = pred.copy()

    size_x = bounding_box.shape[1]
    size_y = bounding_box.shape[0]
    scale_x = size_x / IMAGE_SIZE[1] * 4
    scale_y = size_y / IMAGE_SIZE[0] * 4

    joints_ava = get_ava(joints_queue, left_top, [scale_x, scale_y])
    pred = (pred * [scale_x, scale_y]).astype(int)
    pred = np.add(pred, np.array(left_top))
    pred = get_new(joints_ava, pred, thres)
    mid_29 = ((pred[0][2] + pred[0][9]) / 2).astype(int)
    mid_56 = ((pred[0][5] + pred[0][6]) / 2).astype(int)
    mid_01 = ((pred[0][0] + pred[0][1]) / 2).astype(int)

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
    cv2.imwrite("D:\\cabin_test\\internel_1\\{:05d}.jpg".format(frame_num), show)
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
    main('D:\\cabin_test\\test.mp4', 5, label_list, 10)
