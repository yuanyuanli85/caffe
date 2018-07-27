from opim_datacfg import DataConfig
import argparse
import os

def get_str_from_path(xfile):
    xlist = xfile.split('/')[-3:]
    return os.path.join(xlist[0], xlist[1], xlist[2])

def main(dataset):
    cfg = DataConfig()
    if dataset == 'val':
        xmlpath = cfg.validation_xml_path
        txtfile = cfg.val_list_txt
        imgpath = cfg.val_images_path
    elif dataset == 'train':
        xmlpath = cfg.train_xml_path
        txtfile = cfg.train_list_txt
        imgpath = cfg.train_images_path
    else:
        assert (0), dataset + ' not supported, only val or train'

    with open(txtfile, 'w') as f:
        for xml in os.listdir(xmlpath):
            xmlfile = os.path.abspath(os.path.join(xmlpath, xml))
            imgfile = os.path.abspath(os.path.join(imgpath, xml[:-4]+'.jpg'))
            f.write(get_str_from_path(imgfile)+' '+get_str_from_path(xmlfile)+'\n')

    print 'File list saved to ' + txtfile

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train/val file list of open images database")
    parser.add_argument("--dataset", help=" val or train", required=True)
    args = parser.parse_args()
    main(args.dataset)