import torch
import numpy as np


def intersection_over_union(boxes_preds, boxes_labels, box_format='midpoint'):
    if box_format == 'midpoint':
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2-x1).clamp(0) * (y2-y1).clamp(0)

    box1_area = abs((box1_x2-box1_x1)*(box1_y2-box1_y1))
    box2_area = abs((box2_x2-box2_x1)*(box2_y2-box2_y1))
    return intersection / (box1_area+box2_area-intersection + (1e-6))


def testIOU():
    width = 500
    height = 330
    xmin_tag, xmax_tag, ymin_tag, ymax_tag = 140/width, 500/width, 50/height, 300/height
    xmin_pre, xmax_pre, ymin_pre, ymax_pre = 120/width, 450/width, 60/height, 280/height
    x1 = torch.max(torch.tensor(xmin_tag), torch.tensor(xmin_pre))
    y1 = torch.max(torch.tensor(ymin_tag), torch.tensor(ymin_pre))
    x2 = torch.min(torch.tensor(xmax_tag), torch.tensor(xmax_pre))
    y2 = torch.min(torch.tensor(ymax_tag), torch.tensor(ymax_pre))
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    xmean_tag, ymean_tag, width_tag, height_tag = 320/width, 175/height, 360/width, 250/height
    xmean_pre, ymean_pre, width_pre, height_pre = 285/width, 170/height, 330/width, 220/height
    target = torch.from_numpy(np.array([
        [xmean_tag, ymean_tag, width_tag, height_tag],
        [xmean_tag, ymean_tag, width_tag, height_tag]
    ]))
    pred = torch.from_numpy(np.array([
        [xmean_pre, ymean_pre, width_pre, height_pre],
        [xmean_pre, ymean_pre, width_pre, height_pre]
    ]))

    result = intersection_over_union(pred, target)
    print(result)


if __name__ == '__main__':
    testIOU()