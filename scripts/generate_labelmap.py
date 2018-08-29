import pandas as pd
import os
from collections import defaultdict
from opim_datacfg import DataConfig
from opim_class_hierarchy import get_2ndlevel_class

def generate_classname_map(clasnamefile, class_hierarchy_json):
    '''

    Args:
        clasnamefile:

    Returns: name code -> human readable name

    '''
    fixed = True
    if fixed:
        parent_class = get_fixed_12classes('selected_12classes.prototxt')
    else:
        parent_class = get_2ndlevel_class(class_hierarchy_json)

    cdf = pd.read_csv(clasnamefile, header=None, names=['namecode', 'name'] )
    map = dict()
    classid = 1 # start from 1, since 0 reserved by background type
    for _index, _row in cdf.iterrows():
        key = _row['namecode']
        value = _row['name']
        if value in parent_class:
            map[key] = (value, classid)
            classid += 1
    return map




def get_fixed_12classes(xfile):
    with open(xfile) as f:
        lines = f.readlines()

    classlist = list()
    for xline in lines:
        classname = xline.strip()
        classlist.append(classname)

    return classlist

def generate_labelmap():
    datacfg = DataConfig()
    classmap = generate_classname_map(datacfg.class_description_file, datacfg.class_hierarchy)
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
