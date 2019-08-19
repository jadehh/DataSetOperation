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
from object_detection.utils import dataset_util
import argparse
from jade import *

def dict_voc_to_tf_example(data,
                           dataset_directory,
                           label_map_dict,
                           xml_name,
                           year,
                           ignore_difficult_instances=False,
                           image_subdirectory='JPEGImages'
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
    data['folder'] = dataset_directory + year
    img_path = os.path.join(data['folder'], image_subdirectory, GetLastDir(xml_name)[:-4] + '.jpg')
    full_path = os.path.join(dataset_directory, img_path)
    with tf.gfile.GFile(full_path, 'rb') as fid:
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
            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(int(label_map_dict[obj['name']]["id"]))
            truncated.append(0)
            obj['pose'] = "Unspecified"
            poses.append(obj['pose'].encode('utf8'))

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
    return example



def CreateVOCTFRecords(args):
    CreateSavePath(GetPreviousDir(args.output_path))
    data_dir = args.data_dir
    CreateSavePath(GetPreviousDir(args.output_path))
    writer = tf.python_io.TFRecordWriter(args.output_path)
    label_map_dict, _ = ReadProTxt(args.proto_txt_path, id=False)
    for year in args.years:
        logging.info('Reading from VOC %s dataset.', year)
        examples_list = GetVOCTrainXmlPath(os.path.join(data_dir, year))
        for idx, example in enumerate(examples_list):
            if idx % 100 == 0:
                print('On image %d of %d' % (idx, len(examples_list)))
            with tf.gfile.GFile(example, 'r') as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

            tf_example = dict_voc_to_tf_example(data, args.data_dir, label_map_dict, example, year,
                                                True)
            writer.write(tf_example.SerializeToString())
    writer.close()


def VOCTFRecordShow(tfrecord_path,proto_txt_path):
    categories, _ = ReadProTxt(proto_txt_path)
    with tf.Session() as sess:
        example = tf.train.Example()
        # train_records 表示训练的tfrecords文件的路径
        record_iterator = tf.python_io.tf_record_iterator(path=tfrecord_path)
        for record in record_iterator:
            example.ParseFromString(record)
            f = example.features.feature
            # 解析一个example
            # image_name = f['image/filename'].bytes_list.value[0]
            image_encode = f['image/encoded'].bytes_list.value[0]
            image_height = f['image/height'].int64_list.value[0]
            image_width = f['image/width'].int64_list.value[0]
            xmin = f['image/object/bbox/xmin'].float_list.value
            ymin = f['image/object/bbox/ymin'].float_list.value
            xmax = f['image/object/bbox/xmax'].float_list.value
            ymax = f['image/object/bbox/ymax'].float_list.value
            labels = f['image/object/class/label'].int64_list.value
            text = f['image/object/class/text'].bytes_list.value
            image = io.BytesIO(image_encode)
            labels = list(labels)
            xmin = list(xmin)
            ymin = list(ymin)
            xmax = list(xmax)
            ymax = list(ymax)
            image = Image.open(image)

            image = np.asarray(image)
            bboxes = []
            scores = []
            label_texts = []
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            for i in range(len(xmin)):
                if labels[i] != 0:
                    print(labels[i])
                    bboxes.append(
                        (xmin[i] * image_width, ymin[i] * image_height, xmax[i] * image_width, ymax[i] * image_height))
                    scores.append(1)
                    label_texts.append(categories[labels[i]]["display_name"])
            print("**********************")
            CVShowBoxes(image, bboxes, label_texts, labels, scores=scores, waitkey=0)


if __name__ == '__main__':
    root_dir = "/home/jade/Data/UAFaceGesture/checked/"
    years = []
    for year in os.listdir(root_dir):
        if year != "tfrecords" and os.path.isdir(os.path.join(root_dir,year)):
            years.append(year)
    paraser = argparse.ArgumentParser(description="Create TFRecords")
    paraser.add_argument("--data_dir",default=root_dir,help="")
    paraser.add_argument("--output_path",default=root_dir+"tfrecords/face_gesture_train.tfrecord",help="")
    paraser.add_argument("--proto_txt_path",default=root_dir+"face_gesture.prototxt",help="")
    paraser.add_argument("--years",type=list,default=years,help="")
    args = paraser.parse_args()
    CreateVOCTFRecords(args)

    VOCTFRecordShow("/home/jade/Data/UAFaceGesture/checked/tfrecords/face_gesture_train.tfrecord","/home/jade/Data/UAFaceGesture/checked/face_gesture.prototxt")