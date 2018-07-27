
from opim_datacfg import DataConfig
from opim_convert_anno import load_annotation
import pandas as pd
from generate_labelmap import generate_classname_map
import os
import cv2

def main():
    cfg = DataConfig()
    xdf = pd.read_csv(cfg.validation_anno_file)
    classnameMap = generate_classname_map(cfg.class_description_file)

    for _index, _row in xdf.iterrows():
        imageid = _row['ImageID']
        imagelabel = _row['LabelName']
        classname = classnameMap[imagelabel][0]
        occuluded = _row['IsOccluded']
        xitem = { 'imageid' : imageid, 'classname' : classname,
                  'xmin':_row['XMin'], 'ymin': _row['YMin'], 'xmax':_row['XMax'], 'ymax':_row['YMax'],
                  'occluded':occuluded,'truncated':_row['IsTruncated']}

        imgfile = os.path.join(cfg.val_images_path, imageid+'.jpg')

        cvmat = cv2.imread(imgfile)
        h, w, c = cvmat.shape

        xmin, xmax = _row['XMin'], _row['XMax']
        ymin, ymax = _row['YMin'], _row['YMax']

        xmin, xmax = int(xmin*w), int(xmax*w)
        ymin, ymax = int(ymin*h), int(ymax*h)

        cv2.rectangle(cvmat, (xmin, ymin), (xmax, ymax),
                      color=(255, 0, 0), thickness=3)

        cv2.imshow('image', cvmat)
        cv2.waitKey()


if __name__ == "__main__":
    main()

