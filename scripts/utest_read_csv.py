import pandas as pd
import os
from collections import defaultdict
from dataset_xml import AnnoXml

def generate_classname_map(clasnamefile):
    '''

    Args:
        clasnamefile:

    Returns: name code -> human readable name

    '''
    cdf = pd.read_csv(clasnamefile, header=None, names=['namecode', 'name'] )
    map = dict()
    classid = 1 # start from 1, since 0 reserved by background type
    for _index, _row in cdf.iterrows():
        key = _row['namecode']
        value = _row['name']
        map[key] = (value, classid)
        classid += 1
    return map

class DataConfig(object):
    anno_file_path = "../data/open_images/annotations/"
    val_images_path = "../data/open_images/images"
    validation_anno_file = os.path.join(anno_file_path, "validation-annotations-bbox.csv")
    class_description_file = os.path.join(anno_file_path, "class-descriptions-boxable.csv")
    validation_xml_path = "../data/open_images/xml"

def main_xml():
    ann = AnnoXml(filename="../x.jpg", filefolder='images', imageshape=[200, 300, 3])
    xdict = {'classname':'person', 'truncated':1, 'xmin':100, 'ymin':120, 'xmax':200, 'ymax':220}
    ann.insert_object(xdict)
    ann.dump_to_file('dummy.xml')

def get_image_shape():
    #fixme: dummy shape
    return (100, 100, 3)

def main_xml_v2():
    datacfg = DataConfig()
    xdict = load_annotation(datacfg.validation_anno_file, datacfg.class_description_file)
    for key, value in xdict.items():
        outxml = os.path.join(datacfg.validation_xml_path, key+'.xml')
        xml = AnnoXml(key+'.jpg', datacfg.val_images_path, get_image_shape())
        for object in value:
            xml.insert_object(object)
        xml.dump_to_file(outxml)
        print key, 'save to', outxml

def main():
    datacfg = DataConfig()
    classnameMap = generate_classname_map(datacfg.class_description_file)
    print len(classnameMap)
    xdict = load_annotation(datacfg.validation_anno_file)
    print len(xdict)


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

if __name__ == "__main__":
    main_xml_v2()
