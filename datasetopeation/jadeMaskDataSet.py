#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：Create on 2019/8/13 16:44 by jade
# 邮箱：jadehh@live.com
# 描述：TODO
# 最近修改：2019/8/13 16:44 modify by jade
from jade import *
def RandomSplitDatasetMask(dir,save_dir):
    """
    Mask数据集随机分割图片
    :param dir:
    :param save_dir:
    :return:
    """
    split_rate = 0.8
    image_file_path = os.path.join(dir,"image")
    mask_file_path = os.path.join(dir,"mask")
    image_list = GetAllImagesPath(image_file_path)
    num = len(image_list)
    train_list = image_list[0:int(num * split_rate)]
    val_list = [image_path for image_path in image_list if image_path not in train_list]
    CreateSavePath(os.path.join(save_dir, "image", "training"))
    CreateSavePath(os.path.join(save_dir, "image", "validation"))
    CreateSavePath(os.path.join(save_dir, "mask", "training"))
    CreateSavePath(os.path.join(save_dir, "mask", "validation"))
    for image_path in train_list:

        shutil.copy(image_path,
                    os.path.join(os.path.join(save_dir, "image","training"), GetLastDir(image_path)))
        shutil.copy(os.path.join(mask_file_path,GetLastDir(image_path)),
                    os.path.join(os.path.join(save_dir, "mask", "training"), GetLastDir(image_path)))




    for image_path in val_list:
        shutil.copy(image_path,
                    os.path.join(os.path.join(save_dir, "image" ,"validation"), GetLastDir(image_path)))
        shutil.copy(os.path.join(mask_file_path,GetLastDir(image_path)),
                    os.path.join(os.path.join(save_dir, "mask", "validation"), GetLastDir(image_path)))


    print("Split Done ....")

def VOCToImages(voc_root_path,save_path):
    create_save_path(os.path.join(save_path,"images"))
    create_save_path(os.path.join(save_path,"masks"))
    image_path = os.path.join(voc_root_path,DIRECTORY_IMAGES)
    xml_path = os.path.join(voc_root_path,DIRECTORY_ANNOTATIONS)
    xmls = Get_All_Files(xml_path,".xml")
    processbar = ProcessBar()
    processbar.count = len(xmls)
    for xml in xmls:
        processbar.start_time = time.time()
        images = CutImagesWithVoc(xml)
        for i in range(len(images)):
            masks = load_random_mask()
            for j in range(len(masks)):
                image = mix_image_and_mask(images[i],masks[j])
                cv2.imwrite(
                    os.path.join(save_path, "images", GetLastDir(xml)[:-4] + "_" + str(i) + "_" + str(j) + ".jpg"),
                    image)
                shutil.copy(masks[j], os.path.join(save_path, "masks", GetLastDir(xml)[:-4] + "_" + str(i) + "_" + str(j) + ".jpg"))

        No_Line_Print("saving images ...",processbar)