# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.modeling import registry
from torch import nn
import torch


@registry.ROI_BOX_PREDICTOR.register("FastRCNNPredictor")
class FastRCNNPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(FastRCNNPredictor, self).__init__()
        assert in_channels is not None

        num_inputs = in_channels

        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        num_bbox_reg_classes = 2 if config.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_logit, bbox_pred


@registry.ROI_BOX_PREDICTOR.register("FPNPredictor")
class FPNPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPNPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = in_channels
        # print('roi_box_predictors.py | length of roi-subnet output: {0}, length of input feature: {1}'.format(num_classes, representation_size))

        self.cls_score_offset_flag = cfg.MODEL.ROI_HEADS.CLS_OFFSET
        self.bbox_pred_offset_flag = cfg.MODEL.ROI_HEADS.BBS_OFFSET
        self.cls_score_freeze_flag = cfg.MODEL.ROI_HEADS.CLS_FREEZE
        self.bbox_pred_freeze_flag = cfg.MODEL.ROI_HEADS.BBS_FREEZE
        self.num_old_classes = len(cfg.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES) + 1  # 1 for background
        self.num_new_classes = len(cfg.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES)
        # print('roi_box_predictors.py | number of old classes: {0}, number of new classes: {1}'.format(self.num_old_classes, self.num_new_classes))

        # define the model structure
        if self.cls_score_offset_flag:
            print('roi_box_predictors.py | add offset layer (FC layer) to ROI sub-network classification layer')
            self.cls_score = nn.Linear(representation_size, num_classes)
            self.cls_score_offset_weight = nn.Parameter(torch.ones(1))
            self.cls_score_offset_bias = nn.Parameter(torch.zeros(1))
            print('roi_box_predictors.py | cls_score_weight: {0}'.format(self.cls_score_offset_weight))
            print('roi_box_predictors.py | cls_score_bias: {0}'.format(self.cls_score_offset_bias))
        else:
            self.cls_score = nn.Linear(representation_size, num_classes)
            # print('roi_box_predictors.py | cls_score: {0}'.format(self.cls_score))

        num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        if self.bbox_pred_offset_flag:
            print('roi_box_predictors.py | add offset layer (FC layer) to ROI sub-network bounding box regression layer')
            self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)
            self.bbox_pred_offset_weight = nn.Parameter(torch.ones(1, 4))
            self.bbox_pred_offset_bias = nn.Parameter(torch.zeros(1, 4))
            print('roi_box_predictors.py | bbox_pred_weight: {0}'.format(self.bbox_pred_offset_weight))
            print('roi_box_predictors.py | bbox_pred_bias: {0}'.format(self.bbox_pred_offset_bias))
        else:
            self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)
            # print('roi_box_predictors.py | bbox_pred: {0}'.format(self.bbox_pred))

        # initialize the model parameters
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

        # define model parameters' gradient update capability
        if self.cls_score_freeze_flag:
            print('roi_box_predictors.py | freeze ROI sub-network classification layer')
            for name, param in self.cls_score.named_parameters():
                # print('parameter name: {0}, size: {1}'.format(name, param.size()))
                # print('requires_grad: {0}'.format(param.requires_grad))
                param.requires_grad = False
                # print('requires_grad: {0}'.format(param.requires_grad))

        if self.bbox_pred_freeze_flag:
            print('roi_box_predictors.py | freeze ROI sub-network bounding box regression layer')
            for name, param in self.bbox_pred.named_parameters():
                # print('parameter name: {0}, size: {1}'.format(name, param.size()))
                # print('requires_grad: {0}'.format(param.requires_grad))
                param.requires_grad = False
                # print('requires_grad: {0}'.format(param.requires_grad))

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)

        if self.cls_score_offset_flag:
            scores = self.cls_score(x)
            # print('roi_box_predictors.py | size of score : {0}'.format(scores.size()))
            buff = self.cls_score(x)[:, self.num_old_classes:]
            # print('roi_box_predictors.py | size of buff : {0}'.format(buff.size()))
            scores[:, self.num_old_classes:] = torch.mul(buff, self.cls_score_offset_weight) + self.cls_score_offset_bias
            # print('roi_box_predictors.py | size of scores : {0}'.format(scores.size()))
        else:
            scores = self.cls_score(x)

        if self.bbox_pred_offset_flag:
            bbox_deltas = self.bbox_pred(x)
            # print('roi_box_predictors.py | size of bbox_deltas : {0}'.format(bbox_deltas.size()))
            buff = self.bbox_pred(x)[:, self.num_old_classes * 4:]
            # print('roi_box_predictors.py | size of buff : {0}'.format(buff.size()))
            bbox_deltas[:, self.num_old_classes * 4:] = torch.mul(buff, self.bbox_pred_offset_weight) + self.bbox_pred_offset_bias
            # print('roi_box_predictors.py | size of bbox_deltas : {0}'.format(bbox_deltas.size()))
        else:
            bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


def make_roi_box_predictor(cfg, in_channels):
    func = registry.ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR]
    return func(cfg, in_channels)
