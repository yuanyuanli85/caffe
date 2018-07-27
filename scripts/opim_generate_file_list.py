from opim_datacfg import DataConfig
import os


def get_str_from_path(xfile):
    xlist = xfile.split('/')[-3:]
    return os.path.join(xlist[0], xlist[1], xlist[2])

def main():
    cfg = DataConfig()
    xmlpath = cfg.validation_xml_path
    txtfile = cfg.val_list_txt

    with open(txtfile, 'w') as f:
        for xml in os.listdir(xmlpath):
            xmlfile = os.path.abspath(os.path.join(xmlpath, xml))
            imgfile = os.path.abspath(os.path.join(cfg.val_images_path, xml[:-4]+'.jpg'))
            f.write(get_str_from_path(imgfile)+' '+get_str_from_path(xmlfile)+'\n')

    print 'Done'

if __name__ == "__main__":
    main()