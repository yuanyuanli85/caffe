import os

class DataConfig(object):
    anno_file_path = "../data/open_images/annotations/"

    class_description_file = os.path.join(anno_file_path, "class-descriptions-boxable.csv")
    class_hierarchy = os.path.join(anno_file_path, "bbox_labels_600_hierarchy.json")

    #validation cfg
    val_list_txt = "../data/open_images/val.txt"
    val_images_path = "../data/open_images/validation/validation"
    validation_xml_path = "../data/open_images/validation/xml"
    validation_anno_file = os.path.join(anno_file_path, "validation-annotations-bbox.csv")

    #train cfg
    train_images_path = "../data/open_images/train/train"
    train_list_txt = "../data/open_images/train.txt"
    train_xml_path = "../data/open_images/train/xml"
    train_anno_file = os.path.join(anno_file_path, "train-annotations-bbox.csv")

    #test cfg
    test_images_path = "../data/open_images/test/test"
    test_list_txt = "../data/open_images/test.txt"
    test_xml_path = "../data/open_images/test/xml"