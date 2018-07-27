import os
import pandas as pd
from generate_labelmap import  generate_classname_map
from collections import defaultdict
from dataset_xml import AnnoXml
from opim_datacfg import DataConfig
import cv2
import argparse

def get_image_shape(imgfile):
    cvmat = cv2.imread(imgfile)
    return cvmat.shape

def load_annotation(annofile, classdescfile):
    xdf = pd.read_csv(annofile)
    classnameMap = generate_classname_map(classdescfile)

    xdict = defaultdict(list)
    for _index, _row in xdf.iterrows():
        imageid = _row['ImageID']
        imagelabel = _row['LabelName']
        classname = classnameMap[imagelabel][0]
        occuluded = _row['IsOccluded']
        xitem = { 'imageid' : imageid, 'classname' : classname,
                  'xmin':_row['XMin'], 'ymin': _row['YMin'], 'xmax':_row['XMax'], 'ymax':_row['YMax'],
                  'occluded':occuluded,'truncated':_row['IsTruncated']}
        xdict[imageid].append(xitem)
    return xdict

def convert_valanno_to_xml(annofile, imagepath, xmlpath, class_description_file):
    xdict = load_annotation(annofile, class_description_file)
    for key, value in xdict.items():
        outxml = os.path.join(xmlpath, key+'.xml')
        imgfile = os.path.join(imagepath, key+'.jpg')
        xml = AnnoXml(imgfile, get_image_shape(imgfile))
        for object in value:
            xml.insert_object(object)
        xml.dump_to_file(outxml)
        print key, 'save to', outxml

def main(dataset):
    cfg = DataConfig()
    if dataset == 'val':
        xmlpath = cfg.validation_xml_path
        imgpath = cfg.val_images_path
        annofile = cfg.validation_anno_file
    elif dataset == 'train':
        xmlpath = cfg.train_xml_path
        imgpath = cfg.train_images_path
        annofile = cfg.train_anno_file
    else:
        assert (0), dataset + ' not supported, only val or train'

    convert_valanno_to_xml(annofile, imgpath, xmlpath, cfg.class_description_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert open image annotation from csv to xml")
    parser.add_argument("--dataset", help=" val or train", required=True)
    args = parser.parse_args()
    main(args.dataset)