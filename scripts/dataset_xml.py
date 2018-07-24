from xml.dom.minidom import Document


class AnnoXml(object):

    def __init__(self, filename, filefolder, imageshape):
        self.filename = filename
        self.imageshape = imageshape
        self.filefolder    = filefolder
        self._xml_header()

    def _xml_header(self):
        self.doc = Document()
        root = self.doc.createElement('annotation')
        self.doc.appendChild(root)

        folder_elem = self.doc.createElement('folder')
        folder_elem.appendChild(self.doc.createTextNode(self.filefolder))
        root.appendChild(folder_elem)

        filename_elem = self.doc.createElement('filename')
        filename_elem.appendChild(self.doc.createTextNode(self.filename))
        root.appendChild(filename_elem)

        size_elem = self.doc.createElement('size')
        width = self.doc.createElement('width')
        width.appendChild(self.doc.createTextNode(str(self.imageshape[1])))
        size_elem.appendChild(width)

        height = self.doc.createElement('height')
        height.appendChild(self.doc.createTextNode(str(self.imageshape[0])))
        size_elem.appendChild(height)

        depth = self.doc.createElement('depth')
        depth.appendChild(self.doc.createTextNode(str(self.imageshape[2])))
        size_elem.appendChild(depth)

        root.appendChild(size_elem)
        self.root = root

    def dump_to_file(self, outfile):
        with open(outfile, "w") as xfile:
            xfile.write(self.doc.toprettyxml())

    def create_object(self, objectdict):
        '''
                <object>
                <name>diningtable</name>
                <pose>Unspecified</pose>
                <truncated>1</truncated>
                <difficult>0</difficult>
                <bndbox>
                        <xmin>199</xmin>
                        <ymin>298</ymin>
                        <xmax>466</xmax>
                        <ymax>375</ymax>
                </bndbox>
        </object>
        '''

        object = self.doc.createElement('object')

        name = self.doc.createElement('name')
        name.appendChild(self.doc.createTextNode(objectdict['classname']))
        object.appendChild(name)

        pose = self.doc.createElement('pose')
        pose.appendChild(self.doc.createTextNode('Unspecified'))
        object.appendChild(pose)

        truncated = self.doc.createElement('truncated')
        truncated.appendChild(self.doc.createTextNode(str(objectdict['truncated'])))
        object.appendChild(truncated)

        difficult = self.doc.createElement('difficult')
        difficult.appendChild(self.doc.createTextNode('0'))
        object.appendChild(difficult)


        x_min, x_max = objectdict['xmin'], objectdict['xmax']
        y_min, y_max = objectdict['ymin'], objectdict['ymax']

        img_height, img_width, img_channels = self.imageshape
        x_min, x_max = x_min*img_width, x_max*img_width
        y_min, y_max = y_min*img_height, y_max*img_height


        bbox = self.doc.createElement('bndbox')
        xmin = self.doc.createElement('xmin')
        xmin.appendChild(self.doc.createTextNode(str(int(x_min))))
        bbox.appendChild(xmin)

        ymin = self.doc.createElement('ymin')
        ymin.appendChild(self.doc.createTextNode(str(int(y_min))))
        bbox.appendChild(ymin)

        xmax = self.doc.createElement('xmax')
        xmax.appendChild(self.doc.createTextNode(str(int(x_max))))
        bbox.appendChild(xmax)

        ymax = self.doc.createElement('ymax')
        ymax.appendChild(self.doc.createTextNode(str(int(y_max))))
        bbox.appendChild(ymax)

        object.appendChild(bbox)
        return object


    def insert_object(self, objectdict):
        obj_node = self.create_object(objectdict)
        self.root.appendChild(obj_node)
