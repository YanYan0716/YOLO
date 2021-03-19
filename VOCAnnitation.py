import xml.dom.minidom as minidom


class VOCAnnotationTransform(object):
    '''
    we use this class to know one image's info:
        image_name
        image_boxes
        image_class
    '''

    def __init__(self):
        VOC_CLASS = {'background': 0, 'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5,
                     'bus': 6, 'car': 7, 'cat': 8, 'chair': 9, 'cow': 10, 'diningtable': 11,
                     'dog': 12, 'horse': 13, 'motorbike': 14, 'person': 15, 'pottedplant': 16, 'sheep': 17,
                     'sofa': 18, 'train': 19, 'tvmonitor': 20}
        self.class_dict = VOC_CLASS

    def get_info(self, xml_path):
        '''
        xml_path: the xml's path about an image
        '''
        img_xml = minidom.parse(xml_path)
        root_data = img_xml.documentElement

        # img's name
        img_name = root_data.getElementsByTagName('filename')[0].childNodes[0].nodeValue

        # img's size
        img_size = root_data.getElementsByTagName('size')[0]
        width = img_size.getElementsByTagName('width')[0].childNodes[0].nodeValue
        height = img_size.getElementsByTagName('height')[0].childNodes[0].nodeValue

        objects_info = []
        # img's objects: rect class
        object_list = root_data.getElementsByTagName('object')
        for i in range(len(object_list)):
            class_name = object_list[i].getElementsByTagName('name')[0].childNodes[0].nodeValue
            class_index = self.class_dict[class_name]

            box_info = object_list[i].getElementsByTagName('bndbox')[0]

            xmin = int(box_info.childNodes[1].childNodes[0].nodeValue) #/ float(width)
            ymin = int(box_info.childNodes[3].childNodes[0].nodeValue) #/ float(height)
            xmax = int(box_info.childNodes[5].childNodes[0].nodeValue) #/ float(width)
            ymax = int(box_info.childNodes[7].childNodes[0].nodeValue) #/ float(height)

            # 转换为中心坐标和宽高
            xmean = ( (xmin+xmax)/2 ) / float(width)
            ymean = ( (ymin+ymax)/2 ) / float(height)
            box_width = abs(xmax - xmin) / float(width)
            box_height = abs(ymax - ymin) / float(height)

            info_list = [class_index, round(xmean, 3), round(ymean, 3), round(box_width, 3), round(box_height, 3)]
            objects_info.append(info_list)

        return (img_name, objects_info)


if __name__ == '__main__':
    test = VOCAnnotationTransform()
    result = test.get_info('./000009.xml')
    print(len(result))