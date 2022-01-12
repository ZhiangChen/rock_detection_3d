"""
model.py
Zhiang Chen, Dec 25 2019
Mask RCNN model from torchvision
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from visualize import *

def get_model_instance_segmentation(num_classes, image_mean, image_std, stats=False):
    # load an instance segmentation model pre-trained pre-trained on COCO
    #model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # the size shape and the aspect_ratios shape should be the same as the shape in the loaded model
    anchor_generator = AnchorGenerator(sizes=((32,), (64,), (128,), (256,), (512,)),
                                       aspect_ratios=((0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0)))
    model.rpn.anchor_generator = anchor_generator

    if stats:
        model.transform.image_mean = image_mean
        model.transform.image_std = image_std
    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    model.roi_heads.detections_per_img = 256

    return model

def get_rock_model_instance_segmentation(num_classes, input_channel=8, image_mean=None, image_std=None, pretrained=True):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained)
    if input_channel != 3:
        model.transform.image_mean = image_mean
        model.transform.image_std = image_std

    input_channel = abs(input_channel)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # the size shape and the aspect_ratios shape should be the same as the shape in the loaded model
    anchor_generator = AnchorGenerator(sizes=((16,), (32,), (64,), (128,), (256,)),
                                       aspect_ratios=((0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0)))
    model.rpn.anchor_generator = anchor_generator

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    model.backbone.body.conv1 = nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.roi_heads.detections_per_img = 200

    return model


def predict(model, data, device, batch=False):
    model.eval()
    if not batch:
        pred = model(data.unsqueeze(0).to(device))
    else:
        pred = model(data.to(device))

    return pred

def visualize_pred(image, pred, thred=0.8, display=True):
    """
    visualize only one prediction
    :param pred:
    :return:
    """
    if image.shape[0] > 3:
        image = image[:3, :, :]
    boxes_ = pred["boxes"].cpu().detach().numpy().astype(int)
    boxes = np.empty_like(boxes_)
    boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3] = boxes_[:, 1], boxes_[:, 0], boxes_[:, 3], boxes_[:, 2]
    labels = pred["labels"].cpu().detach().numpy()
    scores = pred["scores"].cpu().detach().numpy()
    masks = pred["masks"]
    indices = scores > thred
    boxes = boxes[indices]
    labels = labels[indices]
    scores = scores[indices]
    masks = masks[indices].squeeze(1)
    masks = (masks.permute((1, 2, 0)).cpu().detach().numpy() > 0.5).astype(np.uint8)
    image = image.permute((1, 2, 0)).cpu().detach().numpy()*255
    #return display_instances(image, boxes, masks, labels, class_names=["background", "non-damaged", "damaged"], scores=scores)
    return display_instances(image, boxes, masks, labels, class_names=["background", "hyp"], scores=scores, display=display)

def visualize_gt(image, target, display=True):
    if image.shape[0] == 3:
        image = image.permute((1, 2, 0)).cpu().detach().numpy() * 255
    else:
        image = image[:3, :, :].permute((1, 2, 0)).cpu().detach().numpy() * 255

    boxes_ = target["boxes"].cpu().detach().numpy().astype(int)
    boxes = np.empty_like(boxes_)
    boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3] = boxes_[:, 1], boxes_[:, 0], boxes_[:, 3], boxes_[:, 2]
    masks = target["masks"].permute((1, 2, 0)).cpu().detach().numpy()
    labels = target["labels"].cpu().detach().numpy()
    print(boxes.shape)
    print(masks.shape)
    print(labels.shape)
    return display_instances(image, boxes, masks, labels, class_names=["background", "hyp"], display=display)

def visualize_result(model, data):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    image, target = data
    visualize_gt(image, target)
    pred = model(image.unsqueeze(0).to(device))
    visualize_pred(image, pred[0])
    visualize_gt(image, target)

def train(model, epochs, device):
    pass

if __name__  ==  "__main__":
    model = get_model_instance_segmentation(3)
