# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
import torch.nn.functional as F
from torch import nn
import random
import numpy as np

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.layers import smooth_l1_loss

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from ..rpn.utils import permute_and_flatten


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):

        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        if not cfg.MODEL.RPN.EXTERNAL_PROPOSAL:
            print('generalized_rcnn.py | Do not use external proposals, so use RPN.')
            self.rpn = build_rpn(cfg, self.backbone.out_channels)
        else:
            print('generalized_rcnn.py | Use external proposals.')
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)

        features, backbone_features = self.backbone(images.tensors)

        (proposals, proposal_losses), anchors, rpn_output = self.rpn(images, features, targets)

        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)

        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, features, backbone_features, anchors, rpn_output

        return result, features,

    def use_external_proposals_edgeboxes(self, images, proposals, targets=None):

        if self.training and targets is None:
            raise ValueError("In external proposal training mode, targets should be passed")
        if proposals is None:
            raise ValueError("In external proposal mode, proposals should be passed")

        images = to_image_list(images)
        features, backbone_features = self.backbone(images.tensors)

        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:  # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            return losses

        return result

    def generate_soften_proposal(self, images, targets=None):

        images = to_image_list(images)  # convert images to image_list type
        features, backbone_features = self.backbone(images.tensors)  # extra image features from backbone network
        (proposals, proposal_losses), anchors, rpn_output = self.rpn(images, features, targets)  # use RPN to generate ROIs

        # sort proposals according to their objectness score
        inds = [proposals[0].get_field("objectness").sort(descending=True)[1]]
        proposals = [proposal[ind] for proposal, ind in zip(proposals, inds)]
        num_proposals = len(proposals[0])

        # get proposal information: bbox, objectness score, proposal mode & image size
        proposal_bbox = proposals[0].bbox
        proposal_score = proposals[0].get_field("objectness")
        proposal_mode = proposals[0].mode
        image_size = proposals[0].size

        # choose first 128 highest objectness score proposals  and then random choose 64 proposals from them
        if num_proposals < 64:
            list = range(0, num_proposals, 1)
            selected_proposal_index = random.sample(list, num_proposals)
        elif num_proposals < 128:
            list = range(0, num_proposals, 1)
            selected_proposal_index = random.sample(list, 64)
        else:
            list = range(0, 128, 1)
            selected_proposal_index = random.sample(list, 64)

        for i, element in enumerate(selected_proposal_index):
            if i == 0:
                selected_proposal_bbox = proposal_bbox[element]
                selected_proposal_bbox = selected_proposal_bbox.view(-1, 4)
                selected_proposal_score = proposal_score[element].view(-1, 1)
            else:
                selected_proposal_bbox = torch.cat((selected_proposal_bbox, proposal_bbox[element].view(-1, 4)), 0)  # vertical tensor cascated
                selected_proposal_score = torch.cat((selected_proposal_score, proposal_score[element].view(-1, 1)), 1)  # horizontal cascate tensors
        selected_proposal_bbox = selected_proposal_bbox.view(-1, 4)
        selected_proposal_score = selected_proposal_score.view(-1)
        selected_proposals = BoxList(selected_proposal_bbox, image_size, proposal_mode)
        selected_proposals.add_field("objectness", selected_proposal_score)
        selected_proposals = [selected_proposals]

        # keep top-5 proposals for feature distillation
        for i in range(5):
            if i == 0:
                feature_proposal_bbox = proposal_bbox[i]
                feature_proposal_bbox = feature_proposal_bbox.view(-1, 4)
                feature_proposal_score = proposal_score[i].view(-1, 1)
            else:
                feature_proposal_bbox = torch.cat((feature_proposal_bbox, proposal_bbox[i].view(-1, 4)), 0)
                feature_proposal_score = torch.cat((feature_proposal_score, proposal_score[i].view(-1, 1)), 1)
        feature_proposal_bbox = feature_proposal_bbox.view(-1, 4)
        feature_proposal_score = feature_proposal_score.view(-1)
        feature_proposals = BoxList(feature_proposal_bbox, image_size, proposal_mode)
        feature_proposals.add_field("objectness", feature_proposal_score)
        feature_proposals = [feature_proposals]

        # generate soften proposal labels
        soften_scores, soften_bboxes = self.roi_heads.calculate_soften_label(features, selected_proposals, targets)  # use ROI-subnet to generate final results

        return (soften_scores, soften_bboxes), selected_proposals, features, backbone_features, anchors, rpn_output, feature_proposals

    def generate_soften_label_external_proposal(self, images, proposals, targets=None):

        images = to_image_list(images)  # convert images to image_list type
        features, backbone_features = self.backbone(images.tensors)  # extra image features from backbone network

        # get proposal information: bbox, proposal mode & image size
        proposal_bbox = proposals[0].bbox
        proposal_mode = proposals[0].mode
        image_size = proposals[0].size

        # choose first 128 highest objectness score proposals  and then random choose 64 proposals from them
        list = range(0, 128, 1)
        selected_proposal_index = random.sample(list, 64)
        for i, element in enumerate(selected_proposal_index):
            if i == 0:
                selected_proposal_bbox = proposal_bbox[element]
                selected_proposal_bbox = selected_proposal_bbox.view(-1, 4)
            else:
                selected_proposal_bbox = torch.cat((selected_proposal_bbox, proposal_bbox[element].view(-1, 4)), 0)  # vertical tensor cascated
        selected_proposal_bbox = selected_proposal_bbox.view(-1, 4)
        selected_proposals = BoxList(selected_proposal_bbox, image_size, proposal_mode)
        selected_proposals = [selected_proposals]

        # generate soften labels
        soften_scores, soften_bboxes = self.roi_heads.calculate_soften_label(features, selected_proposals, targets)

        return (soften_scores, soften_bboxes), selected_proposals

    def calculate_roi_distillation_loss(self, images, soften_proposals, soften_results, cls_preprocess=None, cls_loss=None, bbs_loss=None, temperature=1):

        soften_scores, soften_bboxes = soften_results
        images = to_image_list(images)
        features, backbone_features = self.backbone(images.tensors)  # extra image features from backbone network
        target_scores, target_bboxes = self.roi_heads.calculate_soften_label(features, soften_proposals, soften_results)
        num_of_distillation_categories = soften_scores.size()[1]

        # compute distillation loss
        if cls_preprocess == 'sigmoid':
            soften_scores = F.sigmoid(soften_scores)
            target_scores = F.sigmoid(target_scores)
            modified_soften_scores = soften_scores[:, : num_of_distillation_categories]  # include background
            modified_target_scores = target_scores[:, : num_of_distillation_categories]  # include background
        elif cls_preprocess == 'softmax':  # exp(x_i) / exp(x).sum()
            soften_scores = F.softmax(soften_scores)
            target_scores = F.softmax(target_scores)
            modified_soften_scores = soften_scores[:, : num_of_distillation_categories]  # include background
            modified_target_scores = target_scores[:, : num_of_distillation_categories]  # include background
        elif cls_preprocess == 'log_softmax':  # log( exp(x_i) / exp(x).sum() )
            soften_scores = F.log_softmax(soften_scores)
            target_scores = F.log_softmax(target_scores)
            modified_soften_scores = soften_scores[:, : num_of_distillation_categories]  # include background
            modified_target_scores = target_scores[:, : num_of_distillation_categories]  # include background
        elif cls_preprocess == 'normalization':
            class_wise_soften_scores_avg = torch.mean(soften_scores, dim=1).view(-1, 1)
            class_wise_target_scores_avg = torch.mean(target_scores, dim=1).view(-1, 1)
            normalized_soften_scores = torch.sub(soften_scores, class_wise_soften_scores_avg)
            normalized_target_scores = torch.sub(target_scores, class_wise_target_scores_avg)
            modified_soften_scores = normalized_target_scores[:, : num_of_distillation_categories]  # include background
            modified_target_scores = normalized_soften_scores[:, : num_of_distillation_categories]  # include background
        elif cls_preprocess == 'raw':
            modified_soften_scores = soften_scores[:, : num_of_distillation_categories]  # include background
            modified_target_scores = target_scores[:, : num_of_distillation_categories]  # include background
        else:
            raise ValueError("Wrong preprocessing method for raw classification output")

        if cls_loss == 'l2':
            l2_loss = nn.MSELoss(size_average=False, reduce=False)
            class_distillation_loss = l2_loss(modified_soften_scores, modified_target_scores)
            class_distillation_loss = torch.mean(torch.mean(class_distillation_loss, dim=1), dim=0)  # average towards categories and proposals
        elif cls_loss == 'cross-entropy':  # softmax/sigmoid + cross-entropy
            class_distillation_loss = - modified_soften_scores * torch.log(modified_target_scores)
            class_distillation_loss = torch.mean(torch.mean(class_distillation_loss, dim=1), dim=0)  # average towards categories and proposals
        elif cls_loss == 'softmax cross-entropy with temperature':  # raw + softmax cross-entropy with temperature
            log_softmax = nn.LogSoftmax()
            softmax = nn.Softmax()
            class_distillation_loss = - softmax(modified_soften_scores/temperature) * log_softmax(modified_target_scores/temperature)
            class_distillation_loss = class_distillation_loss * temperature * temperature
            class_distillation_loss = torch.mean(torch.mean(class_distillation_loss, dim=1), dim=0)  # average towards categories and proposals
        elif cls_loss == 'filtered_l2':
            cls_difference = modified_soften_scores - modified_target_scores
            filter = torch.zeros(modified_soften_scores.size()).to('cuda')
            class_distillation_loss = torch.max(cls_difference, filter)
            class_distillation_loss = class_distillation_loss * class_distillation_loss
            class_distillation_loss = torch.mean(torch.mean(class_distillation_loss, dim=1), dim=0)  # average towards categories and proposals
            del filter
            torch.cuda.empty_cache()  # Release unoccupied memory
        else:
            raise ValueError("Wrong loss function for classification")

        # compute distillation bbox loss
        modified_soften_boxes = soften_bboxes[:, 1:, :]  # exclude background bbox
        modified_target_bboxes = target_bboxes[:, 1:num_of_distillation_categories, :]  # exclude background bbox
        if bbs_loss == 'l2':
            l2_loss = nn.MSELoss(size_average=False, reduce=False)
            bbox_distillation_loss = l2_loss(modified_target_bboxes, modified_soften_boxes)
            bbox_distillation_loss = torch.mean(torch.mean(torch.sum(bbox_distillation_loss, dim=2), dim=1), dim=0)  # average towards categories and proposals
        elif bbs_loss == 'smooth_l1':
            num_bboxes = modified_target_bboxes.size()[0]
            num_categories = modified_target_bboxes.size()[1]
            bbox_distillation_loss = smooth_l1_loss(modified_target_bboxes, modified_soften_boxes, size_average=False, beta=1)
            bbox_distillation_loss = bbox_distillation_loss / (num_bboxes * num_categories)  # average towards categories and proposals
        else:
            raise ValueError("Wrong loss function for bounding box regression")

        roi_distillation_losses = torch.add(class_distillation_loss, bbox_distillation_loss)

        return roi_distillation_losses

    def feature_extraction_by_rpn(self, features):
        class_logits = self.rpn.feature_extraction(features)
        return class_logits
