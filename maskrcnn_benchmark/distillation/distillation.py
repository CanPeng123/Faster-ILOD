import argparse
import os
import datetime
import logging
import time
import torch
import torch.distributed as dist
from torch import nn
import numpy as np

from maskrcnn_benchmark.modeling.rpn.utils import permute_and_flatten
from maskrcnn_benchmark.layers import smooth_l1_loss


def calculate_rpn_distillation_loss(rpn_output_source, rpn_output_target, cls_loss=None, bbox_loss=None, bbox_threshold=None):

    rpn_objectness_source, rpn_bbox_regression_source = rpn_output_source
    rpn_objectness_target, rpn_bbox_regression_target = rpn_output_target

    # calculate rpn classification loss
    num_source_rpn_objectness = len(rpn_objectness_source)
    num_target_rpn_objectness = len(rpn_objectness_target)
    final_rpn_cls_distillation_loss = []
    objectness_difference = []

    if num_source_rpn_objectness == num_target_rpn_objectness:
        for i in range(num_target_rpn_objectness):
            current_source_rpn_objectness = rpn_objectness_source[i]
            current_target_rpn_objectness = rpn_objectness_target[i]
            if cls_loss == 'filtered_l1':
                rpn_objectness_difference = current_source_rpn_objectness - current_target_rpn_objectness
                objectness_difference.append(rpn_objectness_difference)
                filter = torch.zeros(current_source_rpn_objectness.size()).to('cuda')
                rpn_distillation_loss = torch.max(rpn_objectness_difference, filter)
                final_rpn_cls_distillation_loss.append(torch.mean(rpn_distillation_loss))
                del filter
                torch.cuda.empty_cache()  # Release unoccupied memory
            elif cls_loss == 'filtered_l2':
                rpn_objectness_difference = current_source_rpn_objectness - current_target_rpn_objectness
                objectness_difference.append(rpn_objectness_difference)
                filter = torch.zeros(current_source_rpn_objectness.size()).to('cuda')
                rpn_difference = torch.max(rpn_objectness_difference, filter)
                rpn_distillation_loss = torch.mul(rpn_difference, rpn_difference)
                final_rpn_cls_distillation_loss.append(torch.mean(rpn_distillation_loss))
                del filter
                torch.cuda.empty_cache()  # Release unoccupied memory
            elif cls_loss == 'normalized_filtered_l2':
                avrage_source_rpn_objectness = torch.mean(current_source_rpn_objectness)
                average_target_rpn_objectness = torch.mean(current_target_rpn_objectness)
                normalized_source_rpn_objectness = current_source_rpn_objectness - avrage_source_rpn_objectness
                normalized_target_rpn_objectness = current_target_rpn_objectness - average_target_rpn_objectness
                rpn_objectness_difference = normalized_source_rpn_objectness - normalized_target_rpn_objectness
                objectness_difference.append(rpn_objectness_difference)
                filter = torch.zeros(current_source_rpn_objectness.size()).to('cuda')
                rpn_difference = torch.max(rpn_objectness_difference, filter)
                rpn_distillation_loss = torch.mul(rpn_difference, rpn_difference)
                final_rpn_cls_distillation_loss.append(torch.mean(rpn_distillation_loss))
                del filter
                torch.cuda.empty_cache()  # Release unoccupied memory
            elif cls_loss == 'masked_filtered_l2':
                source_mask = current_source_rpn_objectness.clone()
                source_mask[current_source_rpn_objectness >= 0.7] = 1  # rpn threshold for foreground
                source_mask[current_source_rpn_objectness < 0.7] = 0
                rpn_objectness_difference = current_source_rpn_objectness - current_target_rpn_objectness
                masked_rpn_objectness_difference = rpn_objectness_difference * source_mask
                objectness_difference.append(masked_rpn_objectness_difference)
                filter = torch.zeros(current_source_rpn_objectness.size()).to('cuda')
                rpn_difference = torch.max(masked_rpn_objectness_difference, filter)
                rpn_distillation_loss = torch.mul(rpn_difference, rpn_difference)
                final_rpn_cls_distillation_loss.append(torch.mean(rpn_distillation_loss))
                del filter
                torch.cuda.empty_cache()  # Release unoccupied memory
            else:
                raise ValueError("Wrong loss function for rpn classification distillation")
    else:
        raise ValueError("Wrong rpn objectness output")
    final_rpn_cls_distillation_loss = sum(final_rpn_cls_distillation_loss)/num_source_rpn_objectness

    # calculate rpn bounding box regression loss
    num_source_rpn_bbox = len(rpn_bbox_regression_source)
    num_target_rpn_bbox = len(rpn_bbox_regression_target)
    final_rpn_bbs_distillation_loss = []
    l2_loss = nn.MSELoss(size_average=False, reduce=False)

    if num_source_rpn_bbox == num_target_rpn_bbox:
        for i in range(num_target_rpn_bbox):
            current_source_rpn_bbox = rpn_bbox_regression_source[i]
            current_target_rpn_bbox = rpn_bbox_regression_target[i]
            current_objectness_difference = objectness_difference[i]
            [N, A, H, W] = current_objectness_difference.size()  # second dimention contains location shifting information for each anchor
            current_objectness_difference = permute_and_flatten(current_objectness_difference, N, A, 1, H, W)
            current_source_rpn_bbox = permute_and_flatten(current_source_rpn_bbox, N, A, 4, H, W)
            current_target_rpn_bbox = permute_and_flatten(current_target_rpn_bbox, N, A, 4, H, W)
            current_objectness_mask = current_objectness_difference.clone()
            current_objectness_mask[current_objectness_difference > bbox_threshold] = 1
            current_objectness_mask[current_objectness_difference <= bbox_threshold] = 0
            masked_source_rpn_bbox = current_source_rpn_bbox * current_objectness_mask
            masked_target_rpn_bbox = current_target_rpn_bbox * current_objectness_mask
            if bbox_loss == 'l2':
                current_bbox_distillation_loss = l2_loss(masked_source_rpn_bbox, masked_target_rpn_bbox)
                final_rpn_bbs_distillation_loss.append(torch.mean(torch.mean(torch.sum(current_bbox_distillation_loss, dim=2), dim=1), dim=0))
            elif bbox_loss == 'l1':
                current_bbox_distillation_loss = torch.abs(masked_source_rpn_bbox - masked_source_rpn_bbox)
                final_rpn_bbs_distillation_loss.append(torch.mean(torch.mean(torch.sum(current_bbox_distillation_loss, dim=2), dim=1), dim=0))
            elif bbox_loss == 'None':
                final_rpn_bbs_distillation_loss.append(0)
            else:
                raise ValueError('Wrong loss function for rpn bounding box regression distillation')
    else:
        raise ValueError('Wrong RPN bounding box regression output')
    final_rpn_bbs_distillation_loss = sum(final_rpn_bbs_distillation_loss)/num_source_rpn_bbox

    final_rpn_loss = final_rpn_cls_distillation_loss + final_rpn_bbs_distillation_loss
    final_rpn_loss.to('cuda')

    return final_rpn_loss


def calculate_feature_distillation_loss(source_features, target_features, loss=None):  # pixel-wise

    num_source_features = len(source_features)
    num_target_fetures = len(target_features)
    final_feature_distillation_loss = []

    if num_source_features == num_target_fetures:
        for i in range(num_source_features):
            source_feature = source_features[i]
            target_feature = target_features[i]
            if loss == 'l2':
                l2_loss = nn.MSELoss(size_average=False, reduce=False)
                feature_distillation_loss = l2_loss(source_feature, target_feature)
                final_feature_distillation_loss.append(torch.mean(feature_distillation_loss))
            elif loss == 'l1':
                feature_distillation_loss = torch.abs(source_feature - target_feature)
                final_feature_distillation_loss.append(torch.mean(feature_distillation_loss))
            elif loss == 'smooth_l1':
                feature_distillation_loss = smooth_l1_loss(source_feature, target_feature, size_average=True, beta=1)
                final_feature_distillation_loss.append(feature_distillation_loss)
            elif loss == 'normalized_filtered_l1':
                source_feature_avg = torch.mean(source_feature)
                target_feature_avg = torch.mean(target_feature)
                normalized_source_feature = source_feature - source_feature_avg  # normalize features
                normalized_target_feature = target_feature - target_feature_avg
                feature_difference = normalized_source_feature - normalized_target_feature
                feature_size = feature_difference.size()
                filter = torch.zeros(feature_size).to('cuda')
                feature_distillation_loss = torch.max(feature_difference, filter)
                final_feature_distillation_loss.append(torch.mean(feature_distillation_loss))
                del filter
                torch.cuda.empty_cache()  # Release unoccupied memory
            elif loss == 'normalized_filtered_l2':
                source_feature_avg = torch.mean(source_feature)
                target_feature_avg = torch.mean(target_feature)
                normalized_source_feature = source_feature - source_feature_avg  # normalize features
                normalized_target_feature = target_feature - target_feature_avg  # normalize features
                feature_difference = normalized_source_feature - normalized_target_feature
                feature_size = feature_difference.size()
                filter = torch.zeros(feature_size).to('cuda')
                feature_distillation = torch.max(feature_difference, filter)
                feature_distillation_loss = torch.mul(feature_distillation, feature_distillation)
                final_feature_distillation_loss.append(torch.mean(feature_distillation_loss))
                del filter
                torch.cuda.empty_cache()  # Release unoccupied memory
            else:
                raise ValueError("Wrong loss function for feature distillation")
    else:
        raise ValueError("Number of source features must equal to number of target features")

    final_feature_distillation_loss = sum(final_feature_distillation_loss)

    return final_feature_distillation_loss


def calculate_roi_distillation_losses(model_source, model_target, images):

    # --- calculate roi-subnet classification and bbox regression distillation loss ---
    # do test on the pre-trained frozen source model to get the soften label
    soften_result, soften_proposal, feature_source, backbone_feature_source, anchor_source, rpn_output_source, feature_proposals = \
        model_source.generate_soften_proposal(images)

    # use soften proposal and soften result to calculate distillation loss
    # 'num_of_distillation_categories' = number of categories for source model including background
    roi_distillation_losses = model_target.calculate_roi_distillation_loss(
        images, soften_proposal, soften_result, cls_preprocess='normalization', cls_loss='l2', bbs_loss='l2', temperature=1)

    return roi_distillation_losses, rpn_output_source, feature_source, backbone_feature_source, soften_result, soften_proposal, feature_proposals




