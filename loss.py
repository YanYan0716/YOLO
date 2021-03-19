import torch
import torch.nn as nn


from utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C+self.B*5)
        # 计算预测框和真实框的iou
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])

        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, bestbox = torch.max(ious, dim=0)  # 得到最合适的框
        exists_box = target[..., 20].unsqueeze(3)

        """
        对于每个grid box，最终有两个预测框，所以bestbox的取值只有0和1两种取值
        exists_box:取值为0或1，表示这里是否有真实框
        bestbox=0: 表示第一个框的预测是正确的，那么留下第一个框的值，
        bestbox=1：表示第二个框的预测是正确的，就留下第二个框的值
        box_prediction包含了（x, y, width, height)
        损失函数.png 红框
        """
        box_prediction = exists_box * (
            (bestbox*predictions[..., 26:30] + (1-bestbox)*predictions[..., 21:25])
        )
        box_targets = exists_box*target[..., 21:25]
        # sign 判断数的正负号， 对w和h开根号
        box_prediction[..., 2:4] = torch.sign(box_prediction[..., 2:4]) * \
                                   torch.sqrt(torch.abs(box_prediction[..., 2:4]+ 1e-6 ))
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        box_loss = self.mse(
            torch.flatten(box_prediction, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )
        """损失函数.png 蓝框 第一行"""
        pred_box = (bestbox*predictions[..., 25:26]+(1-bestbox)*predictions[..., 20:21])
        object_loss = self.mse(
            torch.flatten(exists_box*pred_box),
            torch.flatten(exists_box*target[..., 20:21])
        )
        """
        损失函数.png 蓝框 第二行
        将没有物体的框筛选出来，和真实框做loss， 这里因为有两个预测框，所以要加两次
        """
        no_object_loss = self.mse(
            torch.flatten((1-exists_box)*predictions[..., 20:21], start_dim=1),
            torch.flatten((1-exists_box)*target[..., 20:21])
        )
        no_object_loss += self.mse(
            torch.flatten((1-exists_box)*predictions[..., 25:26], start_dim=1),
            torch.flatten((1-exists_box)*target[..., 20:21], start_dim=1)
        )
        """损失函数.png 绿框"""
        class_loss = self.mse(
            torch.flatten(exists_box*predictions[..., :20], end_dim=-2,),
            torch.flatten(exists_box*target[..., 20], end_dim=-2)
        )

        loss = (
            self.lambda_coord*box_loss
            + object_loss
            + self.lambda_noobj*no_object_loss
            + class_loss
        )
        return loss