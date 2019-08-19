#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/8/15 by jade
# 邮箱：jadehh@live.com
# 描述：TODO
# 最近修改：2019/8/15  下午4:11 modify by jade

import os
from jade.ReadVocData import *
import cv2
from jade import ProcessBar,NoLinePrint
def VOCTOClassify(VOC_PATH):
    xmlPaths = GetFilesWithLastNamePath(os.path.join(VOC_PATH,"Annotations",),".xml")
    save_path = GetPreviousDir(VOC_PATH)
    processbar = ProcessBar()
    processbar.count = len(xmlPaths)
    save_path = CreateSavePath(os.path.join(save_path,"Classify"))
    for xml_path in xmlPaths:
        processbar.start_time = time.time()
        imagename,shape, bboxes, labels_text,labels, difficult, truncated = ProcessXml(xml_path)
        image = cv2.imread(os.path.join(VOC_PATH,"JPEGImages",imagename))
        shape = image.shape
        for i in range(len(bboxes)):
            img = image[int(bboxes[i][1]*shape[0]):int(bboxes[i][3]*shape[0]),int(bboxes[i][0]*shape[1]):int(bboxes[i][2]*shape[1]),]
            height = img.shape[0]
            width = img.shape[1]
            if height > 224 and width > 224:
                CreateSavePath(os.path.join(save_path,labels_text[i]))
                cv2.imwrite(os.path.join(save_path,labels_text[i],imagename[:-4]+"_"+str(i)+".jpg"),img)
        NoLinePrint("writing images",processbar)
if __name__ == '__main__':
    voc_path = "/home/jade/Data/VOCdevkit/VOC2012"
    VOCTOClassify(voc_path)