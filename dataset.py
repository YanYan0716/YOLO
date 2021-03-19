import torch
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


from VOCAnnitation import VOCAnnotationTransform


class VOCDataset(Dataset):
    def __init__(self, csv_file, img_root, S=7, B=2, C=20, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_root = img_root
        self.transform = transform
        self.xml_tool = VOCAnnotationTransform()
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        xml_path = self.annotations[index]
        img_name, boxes = self.xml_tool.get_info(xml_path)
        img_path = os.path.join(self.img_root, img_name)

        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        # convert to cell
        label_matrix = torch.zeros((self.S, self.S, self.C+self.B*5))
        for box in boxes:
            class_label, x, y, width, height = box
            class_label = int(class_label)
            # (i,j):object的中心坐标坐在的grid boxes是第几个
            i, j = int(self.S*y), int(self.S*x)
            # bounding boxes的坐标转换到对应的grid boxes的坐标比例上
            x_cell, y_cell = self.S*x-j, self.S*y-i
            # bounding boxes的宽高转换成对应的grid boxes的比例
            width_cell, height_cell = (width*self.S, height*self.S)

            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1

            box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
            label_matrix[i, j, 21:25] = box_coordinates
            label_matrix[i, j, class_label] = 1

        return image, label_matrix


def test():
    xml_path = './000009.xml'
    VOC = VOCAnnotationTransform()
    img_name, boxes = VOC.get_info(xml_path=xml_path)
    label_matrix = torch.zeros((7, 7, 20 + 2 * 5))
    S = 7
    B = 2
    C = 20
    for box in boxes:
        print('-----------')
        class_label, x, y, width, height = box
        print(class_label, x, y, width, height)
        class_label = int(class_label)
        i, j = int(S * y), int(S * x)  # (i,j):object的中心坐标坐在的grid boxes是第几个
        # bounding boxes的坐标转换到对应的grid boxes的坐标比例上
        x_cell, y_cell = S * x - j, S * y - i
        # bounding boxes的宽高转换成对应的grid boxes的比例
        width_cell, height_cell = (width * S, height * S)

        if label_matrix[i, j, 20] == 0:
            label_matrix[i, j, 20] = 1

        box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
        label_matrix[i, j, 21:25] = box_coordinates
        label_matrix[i, j, class_label] = 1
        # 最后30维存储：one-hot的类别信息和物体的坐标信息，这里给了两组坐标信息的位置，一般只用了一个
        print(i, j, label_matrix[i, j, ...])


if __name__ == '__main__':
    test()