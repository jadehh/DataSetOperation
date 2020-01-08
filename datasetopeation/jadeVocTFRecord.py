#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：Create on 2019/8/14 17:18 by jade
# 邮箱：jadehh@live.com
# 描述：TODO
# # 最近修改：2019/8/14 17:18 modify by jade

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os

from lxml import etree
import PIL.Image
import tensorflow as tf
from tensorflow.python.platform.gfile import GFile
from object_detection.utils import dataset_util
import argparse
from jade import *
from datasetopeation.image_processing.ssd_processing import process_for_train, process_for_eval


def dict_voc_to_tf_example(data,
                           voc_dir,
                           class_names,
                           xml_name,
                           year,
                           ignore_difficult_instances=False,
                           image_subdirectory='JPEGImages',
                           is_normal=True,
                           ):
    """
    转换成tf格式
    :param data:
    :param dataset_directory:
    :param label_map_dict:
    :param xml_name:
    :param year:
    :param ignore_difficult_instances:
    :param image_subdirectory:
    :return:
    """
    data['folder'] = voc_dir
    img_path = os.path.join(GetPreviousDir(GetPreviousDir(xml_name)),DIRECTORY_IMAGES,GetLastDir(xml_name)[:-4] + '.jpg')
    full_path = img_path
    with GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    if 'object' in data:
        for obj in data['object']:
            # difficult = bool(int(obj['difficult']))
            # if ignore_difficult_instances and difficult:
            # continue

            difficult_obj.append(int(0))
            if obj['name'] == 'face':
            #if int(class_names.index(obj['name'])) > 0:
                if is_normal:
                    xmin.append(float(obj['bndbox']['xmin']) / width)
                    ymin.append(float(obj['bndbox']['ymin']) / height)
                    xmax.append(float(obj['bndbox']['xmax']) / width)
                    ymax.append(float(obj['bndbox']['ymax']) / height)
                else:
                    xmin.append(int(obj['bndbox']['xmin']))
                    ymin.append(int(obj['bndbox']['ymin']))
                    xmax.append(int(obj['bndbox']['xmax']))
                    ymax.append(int(obj['bndbox']['ymax']))
                classes_text.append(obj['name'].encode('utf8'))
                classes.append(int(class_names.index(obj['name'])))
                truncated.append(0)
                obj['pose'] = "Unspecified"
                poses.append(obj['pose'].encode('utf8'))
    if is_normal:
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(
                data['filename'].encode('utf8')),
            'image/source_id': dataset_util.bytes_feature(
                data['filename'].encode('utf8')),
            'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
            'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
            'image/object/truncated': dataset_util.int64_list_feature(truncated),
            'image/object/view': dataset_util.bytes_list_feature(poses),
        }))
    else:
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(
                data['filename'].encode('utf8')),
            'image/source_id': dataset_util.bytes_feature(
                data['filename'].encode('utf8')),
            'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
            'image/object/bbox/xmin': dataset_util.int64_list_feature(xmin),
            'image/object/bbox/xmax': dataset_util.int64_list_feature(xmax),
            'image/object/bbox/ymin': dataset_util.int64_list_feature(ymin),
            'image/object/bbox/ymax': dataset_util.int64_list_feature(ymax),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
            'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
            'image/object/truncated': dataset_util.int64_list_feature(truncated),
            'image/object/view': dataset_util.bytes_list_feature(poses),
        }))
    return example


"""
    years = []
    root_dir = "/home/jade/Data/Hand_Gesture/"
    for year in os.listdir(root_dir):
        if year != "tfrecords" and os.path.isdir(os.path.join(root_dir, year)):
            years.append(year)
            dataset_name = os.path.join(root_dir, year)
            CreateVOCDataset(dataset_name, "HandGesture")
    paraser = argparse.ArgumentParser(description="Create TFRecords")
    paraser.add_argument("--data_dir", default=root_dir, help="")
    paraser.add_argument("--output_path", default=root_dir + "tfrecords/hand_gesture_train_" + GetToday() + ".tfrecord",
                         help="")
    paraser.add_argument("--proto_txt_path", default=root_dir + "hand_gesture.prototxt", help="")
    paraser.add_argument("--years", type=list, default=years, help="")
    args = paraser.parse_args()
    CreateVOCTFRecords(args,True)
"""

def CreateVOCTFRecords(args,isnormal=True):
    """
    制作VOC TFrecords
    :param voc_path:
    :param datasetname:
    :param prototxt:
    :param isnormal:
    :return:
    """
    voc_path = args.data_dir
    datasetname = args.dataset_name
    args = args.proto_txt_path
    train_output_path = CreateSavePath(os.path.join(GetPreviousDir(voc_path), "TFRecords"))
    train_output_path = train_output_path + "/" + datasetname + "_train.tfrecord"
    train_writer = tf.io.TFRecordWriter(train_output_path)

    test_output_path = CreateSavePath(os.path.join(GetPreviousDir(voc_path), "TFRecords"))
    test_output_path = test_output_path + "/" + datasetname + "_test.tfrecord"
    test_writer = tf.io.TFRecordWriter(test_output_path)

    categorites,class_names = ReadProTxt(prototxt)
    train_processBar = ProcessBar()
    train_image_path = GetVOCTrainXmlPath(voc_path)
    test_image_path = GetVOCTestXmlPath(voc_path)
    train_processBar.count = len(train_image_path)
    for idx, example in enumerate(train_image_path):
        with GFile(example, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        train_processBar.start_time = time.time()
        tf_example = dict_voc_to_tf_example(data, voc_path, class_names, example, datasetname,
                                            True, is_normal=isnormal)
        train_writer.write(tf_example.SerializeToString())
        NoLinePrint("Writing train TFrecords ...",train_processBar)

    print(" ")
    test_processBar = ProcessBar()
    test_processBar.count  = len(test_image_path)
    for idx, example in enumerate(test_image_path):
        with GFile(example, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        test_processBar.start_time = time.time()
        tf_example = dict_voc_to_tf_example(data, voc_path, class_names, example, datasetname,
                                            True, is_normal=isnormal)
        test_writer.write(tf_example.SerializeToString())
        NoLinePrint("Writing test TFrecords ...", test_processBar)
    #
    train_writer.close()
    test_writer.close()


def CreateVOCYearTFRecords(root_dir,dataset_name,prototxt_path,isnormal=True):
    """
    主目录下有多个VOC 数据集
    :param args:
    :return:
    """
    years = []
    all_train_path = []
    all_test_path = []
    years = []
    for year in os.listdir(root_dir):
        if year.lower() != "tfrecords" and os.path.isdir(os.path.join(root_dir,year)):
            years.append(year)
            # 获取year下面的VOC数据集
            voc_path = os.path.join(root_dir,year)
            train_path = GetVOCTrainXmlPath(voc_path)
            test_path = GetVOCTestXmlPath(voc_path)
            all_train_path.extend(train_path)
            all_test_path.extend(test_path)

    random.shuffle(all_train_path)
    random.shuffle(all_test_path)
    _,class_names = ReadProTxt(prototxt_path)
    train_output_path = CreateSavePath(os.path.join(GetPreviousDir(root_dir), "TFRecords"))
    train_output_path = train_output_path + "/" + dataset_name + "_train.tfrecord"
    train_writer = tf.io.TFRecordWriter(train_output_path)

    test_output_path = CreateSavePath(os.path.join(GetPreviousDir(root_dir), "TFRecords"))
    test_output_path = test_output_path + "/" + dataset_name + "_test.tfrecord"
    test_writer = tf.io.TFRecordWriter(test_output_path)


    train_processBar = ProcessBar()
    train_processBar.count = len(all_train_path)
    for idx, example in enumerate(all_train_path):
        with GFile(example, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        train_processBar.start_time = time.time()
        tf_example = dict_voc_to_tf_example(data, root_dir, class_names, example, dataset_name,
                                            True, is_normal=isnormal)
        train_writer.write(tf_example.SerializeToString())
        NoLinePrint("Writing train TFrecords ...",train_processBar)

    print(" ")
    test_processBar = ProcessBar()
    test_processBar.count  = len(all_test_path)
    for idx, example in enumerate(all_test_path):
        with GFile(example, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        test_processBar.start_time = time.time()
        tf_example = dict_voc_to_tf_example(data, root_dir, class_names, example, dataset_name,
                                            True, is_normal=isnormal)
        test_writer.write(tf_example.SerializeToString())
        NoLinePrint("Writing test TFrecords ...", test_processBar)
    #
    train_writer.close()
    test_writer.close()









def parse_exmp(serial_exmp, output_height, output_width, is_train):
    """
    解析tfrecord 文件
    :param serial_exmp:
    :param output_height:
    :param output_width:
    :param is_train:
    :return:
    """
    feats = tf.io.parse_single_example(serial_exmp,
                                       features={'image/encoded': tf.io.FixedLenFeature([], tf.string),
                                                 'image/height': tf.io.FixedLenFeature([], tf.int64),
                                                 'image/width': tf.io.FixedLenFeature([], tf.int64),
                                                 'image/object/class/label': tf.io.VarLenFeature(tf.int64),
                                                 'image/object/bbox/xmin':  tf.io.VarLenFeature(tf.float32),
                                                 'image/object/bbox/xmax':  tf.io.VarLenFeature(tf.float32),
                                                 'image/object/bbox/ymin':  tf.io.VarLenFeature(tf.float32),
                                                 'image/object/bbox/ymax':  tf.io.VarLenFeature(tf.float32),
                                                 })

    image = tf.cond(
        tf.image.is_jpeg(feats['image/encoded']),
        lambda: tf.image.decode_jpeg(feats['image/encoded']),
        lambda: tf.image.decode_png(feats['image/encoded']))
    label = feats['image/object/class/label']
    label = tf.cast(label,tf.float32)
    img = tf.cast(image, tf.float32)
    width = feats['image/width']
    height = feats['image/height']
    xmin = feats['image/object/bbox/xmin']
    xmax = feats['image/object/bbox/xmax']
    ymin = feats['image/object/bbox/ymin']
    ymax = feats['image/object/bbox/ymax']
    label = tf.sparse.to_dense(label)
    xmin = tf.sparse.to_dense(xmin) * 300
    xmax = tf.sparse.to_dense(xmax) * 300
    ymin = tf.sparse.to_dense(ymin) * 300
    ymax = tf.sparse.to_dense(ymax) * 300
    predict = tf.stack([label,xmin,ymin,xmax,ymax],axis=1)
    img = tf.image.resize(img,(height,width))
    img.set_shape([None,None,3])
    if is_train:
        img = process_for_train(img, output_height, output_width)
    else:
        img = process_for_eval(img, output_height, output_width)
    return img,predict


def get_dataset(fname, output_height, output_width, is_train):
    dataset = tf.data.TFRecordDataset(fname)
    return dataset.map(lambda serial_exmp: parse_exmp(serial_exmp, output_height, output_width, is_train))


def LoadVOCTFRecord(tfrecord_path, batch_size=32, shuffle=False, repeat=True, output_height=300, output_width=300,
                    is_train=True):
    """
    读取目标检测的TFrecords
    :param tfrecord_path:
    :param batch_size:
    :param shuffle: 这里最好就在做TFRecord之前就已经shuffle过
    :param repeat:
    :return:
    """
    dataset = get_dataset(tfrecord_path, output_height, output_width, is_train)
    if shuffle:
        dataset = dataset.shuffle(1000)
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=batch_size)
    return dataset


def VOCTFRecordShow(tfrecord_path, proto_txt_path):
    _,class_names = ReadProTxt(proto_txt_path)
    train_ds = LoadVOCTFRecord(tfrecord_path,batch_size=1)
    for (image,predicts) in train_ds:
        boxes = []
        scores = []
        labels = []
        label_text = []
        image = image.numpy().astype('uint8')[0]
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        predicts = predicts.numpy()[0]
        print(predicts)
        for i in range(predicts.shape[1]):
            label = predicts[0,i]
            xmin = predicts[1,i]
            xmax = predicts[2,i]
            ymin = predicts[3,i]
            ymax = predicts[4,i]
            boxes.append([xmin,ymin,xmax,ymax])
            scores.append(1)
            labels.append(label)
            label_text.append(class_names[int(label)])
        CVShowBoxes(image,boxes,label_text,labels,scores,waitkey=0)



if __name__ == '__main__':
    #CreateVOCYearTFRecords("/home/jade/Data/GestureFace/","Face","/home/jade/Data/GestureFace/face.prototxt")
    VOCTFRecordShow("/home/jade/Data/Hand_Gesture/TFRecords/UA_Handgesture_train.tfrecord","/home/jade/Data/Hand_Gesture/hand_gesture.prototxt")
    #train_ds = LoadVOCTFRecord("/home/jade/Data/Hand_Gesture/TFRecords/UA_Handgesture_test.tfrecord",batch_size=1)
    #for (image,label) in train_ds:
        #print(label.numpy())