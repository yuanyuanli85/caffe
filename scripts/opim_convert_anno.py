import os
import pandas as pd
from generate_labelmap import  generate_classname_map
from collections import defaultdict
from dataset_xml import AnnoXml
from opimg_datacfg import DataConfig
import cv2

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

def convert_valanno_to_xml():
    datacfg = DataConfig()
    xdict = load_annotation(datacfg.validation_anno_file, datacfg.class_description_file)
    for key, value in xdict.items():
        outxml = os.path.join(datacfg.validation_xml_path, key+'.xml')
        imgfile = os.path.join(datacfg.val_images_path, key+'.jpg')
        xml = AnnoXml(imgfile, get_image_shape(imgfile))
        for object in value:
            xml.insert_object(object)
        xml.dump_to_file(outxml)
        print key, 'save to', outxml