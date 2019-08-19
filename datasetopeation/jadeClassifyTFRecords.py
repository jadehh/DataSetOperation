#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/8/15 by jade
# 邮箱：jadehh@live.com
# 描述：TODO
# 最近修改：2019/8/15  下午3:54 modify by jade
import hashlib
import io
import logging
import os

from lxml import etree
import PIL.Image
import tensorflow as tf
from object_detection.utils import dataset_util

import tensorflow as tf
from datasetopeation.image_processing.vgg_processing import process_for_train,process_for_eval
from jade import *
def parse_exmp(serial_exmp,output_height,output_width,is_train):
    feats = tf.io.parse_single_example(serial_exmp,
                                       features={'image/encoded': tf.io.FixedLenFeature([], tf.string),
                                                 'image/object/class/label': tf.io.FixedLenFeature([], tf.int64)})

    image = tf.cond(
        tf.image.is_jpeg(feats['image/encoded']),
        lambda: tf.image.decode_jpeg(feats['image/encoded']),
        lambda: tf.image.decode_png(feats['image/encoded']))
    label = feats['image/object/class/label']
    img = tf.cast(image, tf.float32)
    #在这里做图像预处理

    if is_train:
        img = process_for_train(img,output_height,output_width)
    else:
        img = process_for_eval(img,output_height,output_width)

    return  img,label

def get_dataset(fname,output_height,output_width,is_train):
    dataset = tf.data.TFRecordDataset(fname)
    return dataset.map(lambda serial_exmp:parse_exmp(serial_exmp,output_height,output_width,is_train))


def LoadClassifyTFRecord(tfrecord_path,batch_size=32,shuffle=True,repeat=True,output_height=224,output_width=224,is_train=True):
    """
    读取分类的TFrecords
    :param tfrecord_path:
    :param batch_size:
    :param shuffle:
    :param repeat:
    :return:
    """
    dataset = get_dataset(tfrecord_path,output_height,output_width,is_train)
    if shuffle:
        dataset = dataset.shuffle(1000)
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=batch_size)
    return dataset

def dict_to_tf_example(img_path,
                       categoty):
    with tf.io.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    key = hashlib.sha256(encoded_jpg).hexdigest()
    width = image.width
    height = image.height
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            img_path.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/class/label': dataset_util.int64_feature(categoty),
    }))
    return example

def CreateClassifyTFRecorder(classify_path, datasetname,prototxt):
    train_output_path = CreateSavePath(os.path.join(GetPreviousDir(classify_path), "TFRecords"))
    train_output_path = train_output_path + "/" + datasetname + "_train.tfrecord"
    train_writer = tf.io.TFRecordWriter(train_output_path)

    test_output_path = CreateSavePath(os.path.join(GetPreviousDir(classify_path), "TFRecords"))
    test_output_path = test_output_path + "/" + datasetname + "_test.tfrecord"
    test_writer = tf.io.TFRecordWriter(test_output_path)

    categorites,class_names = ReadProTxt(prototxt)
    class_names = class_names[1:]
    processBar = ProcessBar()
    filename = os.listdir(classify_path)

    img_label_list = []
    for i in range(len(filename)):
        imagepaths = GetAllImagesPath(os.path.join(classify_path, filename[i]))
        for j in range(len(imagepaths)):
            img_label_list.append((imagepaths[j], class_names.index(filename[i])))

    index = 0
    processBar.count = len(img_label_list)
    for (img,label) in img_label_list:
        processBar.start_time = time.time()
        tf_example = dict_to_tf_example(img, label)
        if index < int(0.8 * len(img_label_list)):
            train_writer.write(tf_example.SerializeToString())
        else:
            test_writer.write(tf_example.SerializeToString())
        index = index + 1
        NoLinePrint("Writing TFrecords ...",processBar)

def ClassifyTFRecordShow(tf_path,proto_txt_path,is_train=False):
    test_generator = LoadClassifyTFRecord(tf_path, repeat=False,is_train=is_train)
    categories,class_names = ReadProTxt(proto_txt_path)
    class_names = class_names[1:]
    for (img,label) in test_generator:
        img = img.numpy().astype("uint8")
        label = label.numpy()
        for i in range(img.shape[0]):
            cv2.imshow(class_names[label[i]],img[i,:,:,:])
            cv2.waitKey(1000)

if __name__ == '__main__':
    #CreateClassifyTFRecorder("/home/jade/Data/VOCdevkit/Classify","VOC","/home/jade/Data/VOCdevkit/VOC.prototxt")
    ClassifyTFRecordShow("/home/jade/Data/VOCdevkit/TFRecords/VOC_train.tfrecord","/home/jade/Data/VOCdevkit/VOC.prototxt",is_train=False)