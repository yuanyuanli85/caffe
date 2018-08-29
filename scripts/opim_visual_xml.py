
from opim_datacfg import DataConfig
from opim_convert_anno import load_annotation
import pandas as pd
from generate_labelmap import generate_classname_map
import os
import cv2
from PIL import Image
import json

import xml.etree.ElementTree as ET

def get_bbox_from_xml(xmlfile):
    tree = ET.parse(xmlfile)
    root = tree.getroot()

    mlist = list()
    for object in root.findall('object'):
        name = object.find('name')
        bbox = object.find('bndbox')
        xmin = bbox.find('xmin')
        ymin = bbox.find('ymin')
        xmax = bbox.find('xmax')
        ymax = bbox.find('ymax')
        mlist.append((name.text, (xmin.text, ymin.text, xmax.text, ymax.text)))
    return mlist

def visual_xml(imagefile, xmlfile):
    cvmat   = cv2.imread(imagefile)
    boxlist = get_bbox_from_xml(xmlfile)

    for name, bbox in boxlist:
        xmin, ymin, xmax, ymax = bbox
        cv2.rectangle(cvmat, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0))
        cv2.putText(cvmat, name, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

    cv2.imshow('image', cvmat)
    cv2.waitKey()

def main_vis(listfile):
    with open(listfile) as f:
        lines = f.readlines()

    xpath = "/home/yli150/mobilenet_ssd_ws/caffe/data/open_images"
    for xline in lines:
        imagefile = xline.strip().split()[0]
        xmlfile = xline.strip().split()[-1]

        imagefile = os.path.join(xpath, imagefile)
        xmlfile = os.path.join(xpath, xmlfile)

        visual_xml(imagefile, xmlfile)

if __name__ == "__main__":
    filelst = "/home/yli150/mobilenet_ssd_ws/caffe/data/open_images/val.txt"
    main_vis(filelst)

