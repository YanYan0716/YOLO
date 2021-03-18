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
        img_name, boxes = self.xml_tool(xml_path)
        img_path = os.path.join(self.img_root, img_name)

        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        # convert to cell
        label_matrix = torch.zeros((self.S, self.S, self.C+self.B*5))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            i, j = int(self.S*y), int(self.S*x)
            x_cell, y_cell = self.S*x-j, self.S*y-i
            width_cell, height_cell = (width*self.S, height*self.S)

            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1

            box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
            label_matrix[i, j, 21:25] = box_coordinates
            label_matrix[i, j, class_label] = 1

        return image, label_matrix