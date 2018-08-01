import os
import pandas as pd
from generate_labelmap import  generate_classname_map
from collections import defaultdict
from dataset_xml import AnnoXml
from opim_datacfg import DataConfig
import cv2
import argparse
import time
from PIL import Image
from multiprocessing import Process, Queue
from opim_convert_anno import get_image_shape_v2,  load_annotation

class XmlWorker(Process):
    def __init__(self, id, queue, imagepath, xmlpath):
        Process.__init__(self, name='ModelProcessor')
        self._id = id
        self._queue = queue
        self._imagepath = imagepath
        self._xmlpath = xmlpath

    def run(self):
        while True:
            xdata = self._queue.get()
            if xdata == None:
                self._queue.put(None)
                break
            self.generate_xml(xdata[0], xdata[1])

    def generate_xml(self, xkey, xvalue):
        outxml = os.path.join(self._xmlpath, xkey+'.xml')
        if os.path.exists(outxml):
            return
        imgfile = os.path.join(self._imagepath, xkey+'.jpg')
        xml = AnnoXml(imgfile, get_image_shape_v2(imgfile))
        for object in xvalue:
            xml.insert_object(object)
        xml.dump_to_file(outxml)

class Scheduler:
    def __init__(self, gpuids, imagepath, xmlpath):
        self._queue = Queue()
        self._gpuids = gpuids

        self.__init_workers(imagepath, xmlpath)

    def __init_workers(self, imagepath, xmlpath):
        self._workers = list()
        for gpuid in self._gpuids:
            self._workers.append(XmlWorker(gpuid, self._queue, imagepath, xmlpath))

    def start(self, xdict):

        # put all of files into queue
        for key, value in xdict.items():
            self._queue.put((key, value))

        #add a None into queue to indicate the end of task
        self._queue.put(None)

        #start the workers
        for worker in self._workers:
            worker.start()

        # wait all fo workers finish
        for worker in self._workers:
            worker.join()
        print "all of workers have been done"


def convert_valanno_to_xml(annofile, core_num, imagepath, xmlpath, class_description_file):
    xscheduler = Scheduler(range(core_num), imagepath, xmlpath)

    xdict = load_annotation(annofile, class_description_file)

    xscheduler.start(xdict)


def main(dataset, cores):
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

    start = time.time()
    convert_valanno_to_xml(annofile, cores, imgpath, xmlpath, cfg.class_description_file)
    end  = time.time()
    print dataset, ' finished in ', end-start

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert open image annotation from csv to xml")
    parser.add_argument("--dataset", help=" val or train", required=True)
    parser.add_argument("--cores", type=int, help=" number of process",  required=True, default=4)

    args = parser.parse_args()
    main(args.dataset, args.cores)