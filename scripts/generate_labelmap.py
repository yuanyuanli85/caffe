import pandas as pd
import os
from collections import defaultdict
from utest_read_csv import DataConfig

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


def generate_labelmap():
    datacfg = DataConfig()
    classmap = generate_classname_map(datacfg.class_description_file)
    labelmapfile = 'bbox_labelmap.prototxt'
    with open(labelmapfile, 'w') as labelfile:
        #write background at first
        pre_fix_item="item {\n"
        post_fix_item = "}\n"
        str = "name: \"{0}\" \nlabel: {1} \ndisplay_name: \"{2}\" \n".format('non_of_the_above', 0, 'background')
        labelfile.write(pre_fix_item+str+post_fix_item)

        #wirte each item into labelmap
        for  key, value in classmap.items():
            classname, classid = value
            xstr = "name: \"{0}\" \nlabel: {1} \ndisplay_name: \"{2}\" \n".format(classname, classid , classname)
            labelfile.write(pre_fix_item + xstr + post_fix_item)

if __name__ == "__main__":
    generate_labelmap()
