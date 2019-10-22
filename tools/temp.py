import os
import shutil
import json


def move_pic(label_path):
    list = os.listdir(label_path)
    list.sort()

    for index, i in enumerate(list):

        new_path = os.path.join(label_path, i)

        if os.path.isdir(new_path):
            move_pic(new_path)

        if os.path.isfile(new_path):
            if 'jpg' in new_path:
                shutil.copy(new_path,
                            'C:\\Users\\DELL\\PycharmProjects\\deep-high-resolution-net.pytorch\\data\\cabin\\images')


def delete_pic(pic_path):
    list = os.listdir(pic_path)
    list.sort()
    for x in test_list:
        basename = os.path.splitext(x)[0]
        for index, i in enumerate(list):
            if os.path.isfile(i):
                if 'jpg' in i:
                    if basename in i:
                        os.remove(i)


def delete_json(json_file):
    with open(json_file) as ajson:
        anno = json.load(ajson)

    new_list = []
    for a in anno:
        for i in test_list:
            basename = os.path.splitext(i)[0]

            if basename in a['image_name']:
                add = False
                break
            else:
                add = True
        if add is True:
            new_list.append(a)

    with open("C:\\Users\\DELL\\PycharmProjects\\deep-high-resolution-net.pytorch\\data\\cabin\\annot\\temp.json",
              "w+") as f:
        json.dump(new_list, f)
        print("加载入文件完成...")
    ajson.close()
    f.close()
def count_num(json_file):
    i=0
    with open(json_file) as ajson:
        anno = json.load(ajson)
    for a in anno:
        i+=1
    print(i)

if __name__ == '__main__':
    test_list = ['NJ_Y_Y_90_20190814_081400_b4573c64.avi',
                 'NJ_Y_Y_90_20190814_081400_cd097d2b.avi ',
                 'NJ_Y_Y_90_20190815_081500_4dba7e36.avi ',
                 'NJ_Y_Y_90_20190815_081500_d9e32e4d.avi ',
                 'NJ_Y_1_70_20190814_081400_5fcf5b40.avi',
                 'NJ_Y_1_70_20190902_151144_f888aba8.avi',
                 'NJ_Y_1_70_20191012_114758_d7246cac.avi',
                 'NJ_Y_1_70_20191013_170709_1f4076f8.avi',
                 'NJ_Y_1_70_20190814_081400_f610fde8.avi',
                 'NJ_Y_1_70_20190902_151640_c68322a1.avi',
                 'NJ_Y_1_70_20191012_164940_f16b9cec.avi',
                 'NJ_Y_1_70_20190815_081500_02a9176f.avi',
                 'NJ_Y_1_70_20190905_101539_5454c7bb.avi',
                 'NJ_Y_1_70_20191013_122859_881e0700.avi',
                 'NJ_Y_1_70_20190815_081500_f1c6ee45.avi',
                 'NJ_Y_1_70_20190905_120526_0b13801c.avi',
                 'NJ_Y_1_70_20191013_160139_cb79ae51.avi']
    # move_pic('C:\\Users\\DELL\\Downloads\\#3960')
    delete_pic('C:\\Users\\DELL\\PycharmProjects\\deep-high-resolution-net.pytorch\\data\\cabin\\images')
    # delete_json(
        # 'C:\\Users\\DELL\\PycharmProjects\\deep-high-resolution-net.pytorch\\data\\cabin\\annot\\cabin_train.json')
    # count_num("C:\\Users\\DELL\\PycharmProjects\\deep-high-resolution-net.pytorch\\data\\cabin\\annot\\temp.json")
