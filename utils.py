import torch
import numpy as np
from collections import Counter


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


def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format='midpoint', num_classes=20):
    average_precision = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        for detection in pred_boxes:
            if detection[1] == c:
                detection.append(detection)
        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        detections.sort(key=lambda  x: x[2], reverse=True)
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_bboxes = len(ground_truths)

        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]
            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format
                )
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes+epsilon)
        precision = torch.divide(TP_cumsum, (TP_cumsum+FP_cumsum+epsilon))
        precision = torch.cat((torch.tensor([1]), precision))
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precision.append(torch.trapz(precision, recalls))
    return sum(average_precision)/len(average_precision)


def convert_cellboxes(predictions, S=7):
    predictions = predictions.to('cpu')
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat((predictions[..., 20].unsqeeze(0), predictions[..., 25].unsqueeze(0)), dim=0)
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1*(1-best_box)+bboxes2*(best_box)
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1/S*(best_boxes[..., :1]+cell_indices)
    y = 1/S*(best_boxes[..., 1:2] +cell_indices.permute(0, 2, 1, 3))
    w_y = 1/S*best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., 20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(-1)
    converted_preds = torch.cat((predicted_class, best_confidence, converted_bboxes), dim=-1)
    return converted_preds


def cellboxes_to_boxes(out, S=7):
    convert_pred = convert_cellboxes(out).reshape(out.shape[0], S*S, -1)
    convert_pred[..., 0] = convert_pred[..., 0].long()
    all_bboxes = []

    for ex_iou in range(out.shape[0]):
        bboxes = []
        for bbox_idx in range(S*S):
            bboxes.append([x.item() for x in convert_pred[ex_iou, bbox_idx, :]])
        all_bboxes.append(bboxes)
    return all_bboxes


def get_bboxes(loader, model, iou_threshold, pred_format='cells', box_format='midpoint', device='CUDA'):
    all_pred_boxes = []
    all_true_boxes = []

    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]


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