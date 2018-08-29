
from opim_datacfg import DataConfig
from opim_convert_anno import load_annotation
import pandas as pd
from generate_labelmap import generate_classname_map
import os
import cv2
from PIL import Image
import json
from collections import Counter
import xml.etree.ElementTree as ET

def get_bbox_from_xml(xmlfile):
    tree = ET.parse(xmlfile)
    root = tree.getroot()

    mlist = list()
    for object in root.findall('object'):
        name = object.find('name')
        mlist.append(name.text)
    return mlist


def main_vis(listfile):
    with open(listfile) as f:
        lines = f.readlines()
    print(len(lines), 'samples')

    xpath = "/home/yli150/mobilenet_ssd_ws/caffe/data/open_images"
    xdict = Counter()
    for i, xline in enumerate(lines):
        xmlfile = xline.strip().split()[-1]
        xmlfile = os.path.join(xpath, xmlfile)

        if i % 10000 == 0 :
            print (i, xdict)
        mlist = get_bbox_from_xml(xmlfile)
        for bbox in mlist:
            xdict[bbox] +=  1

    print xdict

if __name__ == "__main__":
    filelst = "/home/yli150/mobilenet_ssd_ws/caffe/data/open_images/train.txt"
    main_vis(filelst)

