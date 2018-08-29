import numpy as np  
import sys,os  
import cv2
sys.path.insert(0, '/home/yli150/mobilenet_ssd_ws/caffe/python')
import caffe  



caffe_model= '/home/yli150/mobilenet_ssd_ws/train_opim/snapshot/mmsd_13c_iter_200000.caffemodel'
net_file= '/home/yli150/mobilenet_ssd_ws/train_opim/mobilenet_1.0_13class/mobilenet_ssd_1.0_deploy.prototxt'

test_dir = "../../train_opim/images"

if not os.path.exists(caffe_model):
    print("MobileNetSSD_deploy.caffemodel does not exist,")
    print("use merge_bn.py to generate it.")
    exit()
net = caffe.Net(net_file,caffe_model,caffe.TEST)
caffe.set_mode_gpu()
caffe.set_device(2)

CLASSES = ('background', 'Toy', 'Bicycle', 'Home application',
           'Couch', 'Human body', 'Plumbing fixture',
           'Bed', 'Table', 'Telephone',
           'Auto part', 'Kitchen application', 'Car')

def preprocess(src):
    img = cv2.resize(src, (300,300))
    img = img - 127.5
    img = img * 0.007843
    return img

def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

def detect(imgfile):
    origimg = cv2.imread(imgfile)
    img = preprocess(origimg)
    
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    out = net.forward()  
    box, conf, cls = postprocess(origimg, out)

    for i in range(len(box)):
       p1 = (box[i][0], box[i][1])
       p2 = (box[i][2], box[i][3])
       if conf[i] < 0.3:
          continue
       cv2.rectangle(origimg, p1, p2, (0,255,0))
       p3 = (max(p1[0], 15), max(p1[1], 15))
       title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
       cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
    cv2.imshow("SSD", origimg)
    
    outfile = os.path.join('../../train_opim/outimages', os.path.basename(imgfile))
    cv2.imwrite(outfile, origimg)
 
    k = cv2.waitKey(0) & 0xff
        #Exit if ESC pressed
    if k == 27 : return False
    return True

for f in os.listdir(test_dir):
    if detect(test_dir + "/" + f) == False:
       break
