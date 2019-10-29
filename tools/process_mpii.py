import json
import os
import cv2
import numpy as np


def main():
    with open("C:\\Users\\DELL\\Desktop\\#3960结果26-50.json") as anno_file:
        anno = json.load(anno_file)
    new_json = []
    for a in anno:
        try:
            image_name = a['image_name']
            img = cv2.imread(
                "C:\\Users\\DELL\\PycharmProjects\\deep-high-resolution-net.pytorch\\data\\cabin\\images\\" + image_name)
            img_x = img.shape[1]
            img_y = img.shape[0]
            joints = np.array(a['annotations'][0]['anno'][0]['data']['point2d'])
            left_top = np.amin(joints, axis=0) - 20
            left_top = np.where(left_top < 0, 0, left_top)

            right_bottom = np.amax(joints, axis=0) + 50
            if right_bottom[0] > img_x:
                right_bottom[0] = img_x
            elif right_bottom[1] > img_y:
                right_bottom[1] = img_y

            img_crop = img[int(left_top[1]):int(right_bottom[1]), int(left_top[0]):int(right_bottom[0])]
            cv2.imwrite('C:\\Users\\DELL\\PycharmProjects\\deep-high-resolution-net.pytorch\\data\\cabin\\new\\' + a[
                'image_name'], img_crop)
            print("writing", a['image_name'])
            joints_new = joints - np.array([left_top[0], left_top[1]])

            a['annotations'][0]['anno'][0]['data']['point2d'] = joints_new.tolist()
            new_json.append(a)




        except IndexError:
            continue
    with open("C:\\Users\\DELL\\PycharmProjects\\deep-high-resolution-net.pytorch\\data\\cabin\\annot\\mpii.json",
              'w+') as o:
        json.dump(new_json, o)
        print("new file generated")
    anno_file.close()
    o.close()


if __name__ == '__main__':
    main()
