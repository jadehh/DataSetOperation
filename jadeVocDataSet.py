#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：Create on 2019/8/13 16:20 by jade
# 邮箱：jadehh@live.com
# 描述：数据集操作 (所有的数据集操作都是在VOC数据集上)
# 最近修改：2019/8/13 16:20 modify by jade
from jade import *
import cv2
def RandomSplitDataset(voc_path,sampleNum):
    """
    随机分割数据集
    :param voc_path: 数据集的路径
    :param sampleNum: 分割后样本的数量
    :return:
    """
    image_path_list = GetAllImagesPath(os.path.join(voc_path,DIRECTORY_IMAGES))
    random.shuffle(image_path_list)
    processBar = ProcessBar()
    processBar.count = len(image_path_list)
    for i in range(len(image_path_list)):
        processBar.start_time = time.time()
        image_path = image_path_list[i]
        if i>0 and i%sampleNum == 0:
            save_voc_path = CreateSavePath(voc_path+"_sample_"+str(i))
            save_image_path = CreateSavePath(save_voc_path,DIRECTORY_IMAGES)
            save_xml_path = CreateSavePath(save_voc_path,DIRECTORY_ANNOTATIONS)
            shutil.copy(image_path,os.path.join(save_image_path,GetLastDir(image_path)))
            shutil.copy(os.path.join(voc_path,DIRECTORY_ANNOTATIONS,GetLastDir(image_path)[:-4]+".xml"),os.path.join(save_voc_path,DIRECTORY_ANNOTATIONS,GetLastDir(image_path)[:-4]+".xml"))
        NoLinePrint("randomSplitDataset...",processBar)

def RemoveFilesWithNoLabels(dir,savedir):
    """
    #去除没有标注的文件
    :param dir: voc
    :param savedir: 重新保存路径
    :return:
    """
    JPEGImages_path = os.path.join(dir,DIRECTORY_IMAGES)
    ANNOTATIONS_path = os.path.join(dir,DIRECTORY_ANNOTATIONS)
    CreateSavePath(os.path.join(savedir,DIRECTORY_IMAGES))
    CreateSavePath(os.path.join(savedir,DIRECTORY_ANNOTATIONS))

    xml_list = GetFilesWithLastNamePath(ANNOTATIONS_path,".xml")
    processbar = ProcessBar()
    processbar.count = len(xml_list)
    for xml in xml_list:
        processbar.start_time = time.time()
        imagename, shape, bboxes, labels, labels_text, difficult, truncated = ProcessXml(xml)
        if len(bboxes) > 0:
            shutil.copy(xml,os.path.join(savedir,DIRECTORY_ANNOTATIONS,GetLastDir(xml)))
            shutil.copy(os.path.join(JPEGImages_path,imagename),os.path.join(savedir,DIRECTORY_IMAGES,imagename))
        NoLinePrint("remove null files ...",processbar)




def RestoreCutImageWithVoc(Voc_path,Cut_path):
    """
    恢复裁剪框到VOC (主要是配合查看图片类别对不对,不对移动图片到另一个文件夹,重新生成xml)
    :param Voc_path:
    :param Cut_path:
    :return:
    """
    image_labels = GetLabelAndImagePath(Cut_path)
    JPEGImages_path = OpsJoin(Voc_path,DIRECTORY_IMAGES)
    ANNOTATIONS_path = OpsJoin(Voc_path,DIRECTORY_ANNOTATIONS)
    xml_list = GetFilesWithLastNamePath(ANNOTATIONS_path,".xml")
    processbar = ProcessBar()

    processbar.count = len(xml_list)
    for xml in xml_list:
        processbar.start_time = time.time()
        imagename, shape, bboxes, labels, labels_type_text, difficult, truncated = ProcessXml(xml)
        image = cv2.imread(os.path.join(JPEGImages_path,imagename))
        new_shape = image.shape
        new_labels = []
        for i in range(len(bboxes)):
            label = image_labels[imagename[:-4]+"_"+str(i)]
            new_labels.append(label)
        GenerateXml(GetLastDir(imagename[:-4]),new_shape,bboxes,new_labels,OpsJoin(Voc_path,"Annotations_new"))
        NoLinePrint("cut image with box ...",processbar)


def CreateVOCDataset(dir, datasetname):
    """
    创建VOC数据集,主要生成train.txt 和 var.txt
    :param dir:
    :param datasetname:
    :return:
    """
    root_path = dir
    dataset_name = datasetname
    Annotations = DIRECTORY_ANNOTATIONS
    JPEGImages = DIRECTORY_IMAGES

    if os.path.exists(os.path.join(root_path, "ImageSets", "Main")) is not True:
        os.makedirs(os.path.join(root_path, "ImageSets", "Main"))
    else:
        shutil.rmtree(os.path.join(root_path, "ImageSets", "Main"))
        os.makedirs(os.path.join(root_path, "ImageSets", "Main"))
    Main_path = os.path.join(root_path, "ImageSets", "Main")
    image_files = os.listdir(os.path.join(root_path, JPEGImages))

    train_image_files = random.sample(image_files, int(len(image_files) * 0.9))
    test_image_files = [file for file in image_files if file not in train_image_files]

    for train_image_file in train_image_files:
        with open(os.path.join(Main_path, "train_var.txt"), "a") as f:
            # with open(os.path.join(Main_path, "train.txt"), "a") as f:
            image_file = dataset_name + "/" + JPEGImages + "/" + train_image_file
            xml_file = dataset_name + "/" + Annotations + "/" + train_image_file[:-4] + ".xml"
            filename = train_image_file[:-4]
            f.write(filename + "\n")
            # f.write(image_file + " " + xml_file + "\n")

    for test_image_file in test_image_files:
        with open(os.path.join(Main_path, "test_var.txt"), "a") as f:
            # with open(os.path.join(Main_path, "test.txt"), "a") as f:
            image_file = dataset_name + "/" + JPEGImages + "/" + test_image_file
            xml_file = dataset_name + "/" + Annotations + "/" + test_image_file[:-4] + ".xml"
            filename = test_image_file[:-4]
            f.write(filename + "\n")
            # f.write(image_file + " " + xml_file + "\n")

    for train_image_file in train_image_files:
        with open(os.path.join(Main_path, "train.txt"), "a") as f:
            # with open(os.path.join(Main_path, "train.txt"), "a") as f:
            image_file = dataset_name + "/" + JPEGImages + "/" + train_image_file
            xml_file = dataset_name + "/" + Annotations + "/" + train_image_file[:-4] + ".xml"
            filename = train_image_file[:-4]
            # f.write(filename + "\n")
            f.write(image_file + " " + xml_file + "\n")

    for test_image_file in test_image_files:
        with open(os.path.join(Main_path, "test.txt"), "a") as f:
            # with open(os.path.join(Main_path, "test.txt"), "a") as f:
            image_file = dataset_name + "/" + JPEGImages + "/" + test_image_file
            xml_file = dataset_name + "/" + Annotations + "/" + test_image_file[:-4] + ".xml"
            filename = test_image_file[:-4]
            # f.write(filename + "\n")
            f.write(image_file + " " + xml_file + "\n")


def CutImagesWithVoc(VocPath,savedir=None,use_chinese_name=True,prototxt_path="/home/jade/Data/StaticDeepFreeze/ThirtyTypes.prototxt"):
    """
    通过VOC数据集来裁剪图片
    :param VocPath: 数据集路径
    :param savedir: 保存路径
    :param use_chinese_name:是否使用中文名做文件夹
    :param prototxt_path: 类别文件
    :return:
    """
    if savedir is  None:
        savedir = os.path.join(VocPath,"CutImages")
    CreateSavePath(savedir)
    JPEGImages_path = OpsJoin(VocPath,DIRECTORY_IMAGES)
    ANNOTATIONS_path = OpsJoin(VocPath,DIRECTORY_ANNOTATIONS)
    xml_list = GetFilesWithLastNamePath(ANNOTATIONS_path,".xml")
    processbar = ProcessBar()
    processbar.count = len(xml_list)
    for xml in xml_list:
        processbar.start_time = time.time()
        imagename, shape, bboxes, labels, labels_type_text, difficult, truncated = ProcessXml(xml)
        if use_chinese_name:
            labels_text = type_to_chinese(labels,prototxt_path)
        else:
            labels_text = labels
        for i in range(len(bboxes)):
            box = bboxes[i]
            image = cv2.imread(os.path.join(JPEGImages_path,imagename))
            cut_image = image[box[1]:box[3],box[0]:box[2],:]
            cut_image = cv2.resize(cut_image,(width,width))
            name = labels_text[i]
            CreateSavePath(os.path.join(savedir,name))
            cv2.imwrite(os.path.join(savedir,name,imagename[:-4]+"_"+str(i)+".png"),cut_image)
        NoLinePrint("cut image with box ...",processbar)

def CutImageWithBoxes(image, bboxes, convert_rgb2bgr=False):
    """
    通过多个框裁剪图片
    :param image:
    :param bboxes:
    :param convert_rgb2bgr:
    :return:
    """
    cut_images = []
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        cut_image = CutImageWithBox(image, bbox)
        cut_images.append(cut_image)
        # cut_images.append(cv2.resize(cut_image,(256,256)))

    return cut_images

def CutImagesWithXml(xml_path):
    """
    #VOC数据集裁剪目标框
    :param xml_path: xml 的路径
    :return: 返回图片和标签
    """
    imagename,shape, bboxes, labels, labels_text, difficult, truncated = ProcessXml(xml_path)
    root_path = GetPreviousDir(GetPreviousDir(xml_path))
    image_path = os.path.join(root_path,DIRECTORY_IMAGES,GetLastDir(xml_path)[:-4]+".jpg")
    image = cv2.imread(image_path)
    images = CutImageWithBoxes(image,bboxes)
    return images,labels

if __name__ == '__main__':

    # years = []
    # CreateSavePath("/home/jade/Data/FaceGesture/")
    # for year in os.listdir("/home/jade/Data/Face-Gesture/"):
    #     if year != "tfrecords" and os.path.isdir(os.path.join("/home/jade/Data/Face-Gesture/",year)):
    #         RemoveFilesWithNoLabels(os.path.join("/home/jade/Data/Face-Gesture/",year), os.path.join("/home/jade/Data/FaceGesture/",year))
    #
    root_dir = "/home/jade/Data/FaceGesture"
    years = os.listdir(root_dir)
    for year in years:
        if os.path.isdir(os.path.join(root_dir,year)):
            CreateVOCDataset(os.path.join(root_dir,year),year)