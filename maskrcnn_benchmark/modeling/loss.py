# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
This file contains specific functions for computing losses on the RPN
file
"""

import torch
from torch.nn import functional as F

from .utils import concat_box_prediction_layers

from ..balanced_positive_negative_sampler import BalancedPositiveNegativeSampler
from ..utils import cat

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist


class RPNLossComputation(object):
    """
    This class computes the RPN loss.
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder,
                 generate_labels_func):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        # self.target_preparator = target_preparator
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.copied_fields = []
        self.generate_labels_func = generate_labels_func
        self.discard_cases = ['not_visibility', 'between_thresholds']

    def match_targets_to_anchors(self, anchor, target, copied_fields=[]):

        # print('rpn | loss.py | match_targets_to_anchors | anchor : {0}'.format(anchor))
        # print('rpn | loss.py | match_targets_to_anchors | target : {0}'.format(target))
        match_quality_matrix = boxlist_iou(target, anchor)  # [num of target box, num of bounding box]
        # print('rpn | loss.py | match_targets_to_anchors | match_quality_matrix size : {0}'.format(match_quality_matrix.size()))

        # value = 0 ~ (M-1) means which gt to match to
        # value = -1 or -2 means no gt to match to, -1 = below_low_threshold, -2 = between_thresholds
        matched_idxs = self.proposal_matcher(match_quality_matrix)  # [num of bounding box]
        # print('rpn | loss.py | match_targets_to_anchors | matched_idxs size : {0}'.format(matched_idxs.size()))

        # RPN doesn't need any fields from target for creating the labels, so clear them all
        target = target.copy_with_fields(copied_fields)

        # get the targets corresponding GT for each anchor
        # need to clamp the indices because we can have a single GT in the image, and matched_idxs can be -2, which goes out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        # print('rpn | loss.py | match_targets_to_anchors | matched_targets : {0}'.format(matched_targets))

        matched_targets.add_field("matched_idxs", matched_idxs)

        return matched_targets, match_quality_matrix

    def prepare_targets(self, anchors, targets):
        labels = []
        regression_targets = []
        overlap_result = []
        matched_result = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            matched_targets, matched_quality_matrix = self.match_targets_to_anchors(anchors_per_image, targets_per_image, self.copied_fields)

            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = self.generate_labels_func(matched_targets)
            labels_per_image = labels_per_image.to(dtype=torch.float32)

            # Background (negative examples)
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD  # match_idxs = -1 means below_low_threshold
            labels_per_image[bg_indices] = 0  # make these anchors' label to be 0

            # discard anchors that go out of the boundaries of the image
            if "not_visibility" in self.discard_cases:
                labels_per_image[~anchors_per_image.get_field("visibility")] = -1  # make these anchors' label to be -1

            # discard indices that are between thresholds
            if "between_thresholds" in self.discard_cases:
                inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1  # make these anchors' label to be -1

            # print('rpn | loss.py | prepare_targets | labels_per_image size : {0}'.format(labels_per_image.size()))

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(matched_targets.bbox, anchors_per_image.bbox)

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)
            overlap_result.append(matched_quality_matrix)
            matched_result.append(matched_idxs)

        return labels, regression_targets, overlap_result, matched_result

    def __call__(self, anchors, objectness, box_regression, targets):
        """
        Arguments:
            anchors (list[BoxList])
            objectness (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor
        """
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        labels, regression_targets, overlap_result, matched_result = self.prepare_targets(anchors, targets)
        # print('rpn | loss.py | call | labels size : {0}'.format(labels[0].size()))
        # print('rpn | loss.py | call | regression_targets size : {0}'.format(regression_targets[0].size()))
        # print('rpn | loss.py | call | overlap_result size : {0}'.format(overlap_result[0].size()))
        # print('rpn | loss.py | call | matched_result size : {0}'.format(matched_result[0].size()))

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        # print('rpn | loss.py | call | sampled_pos_inds size : {0}'.format(sampled_pos_inds[0].size()))
        # print('rpn | loss.py | call | sampled_neg_inds size : {0}'.format(sampled_neg_inds[0].size()))
        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)
        # print('rpn | loss.py | call | sampled_pos_inds size : {0}'.format(sampled_pos_inds.size()))
        # print('rpn | loss.py | call | sampled_pos_inds : {0}'.format(sampled_pos_inds))
        # print('rpn | loss.py | call | sampled_neg_inds size : {0}'.format(sampled_neg_inds.size()))
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        # print('rpn | loss.py | call | sampled_inds size : {0}'.format(sampled_inds.size()))

        sampled_inds_overlap = torch.zeros(sampled_inds.size())
        # print('rpn | loss.py | call | sampled_inds_overlap size : {0}'.format(sampled_inds_overlap.size()))
        count = 0
        for index in sampled_inds:
            label = labels[0][index]  # 1 = positive proposal, 0 = negative proposal, -1 = useless proposal
            if label > 0:
                matched_gt = matched_result[0][index]
                overlap = overlap_result[0][matched_gt][index]
                # overlap = overlap.to('cpu')
                # overlap = float(overlap.data.numpy())
                # print('index = {0}, matched_gt = {1}, overlap = {2}'.format(index, matched_gt, overlap))
            elif label == 0:
                overlap = 0
            else:
                raise ValueError('Something goes wrong at proposal choosing procedure.')
            sampled_inds_overlap[count] = overlap
            count = count + 1
        sampled_inds_overlap = sampled_inds_overlap.to('cuda')
        # print('rpn | loss.py | call | sampled_inds_overlap : {0}'.format(sampled_inds_overlap))

        scaled_weight = -50 * torch.exp(-20 * sampled_inds_overlap)
        scaled_weight = 0.25 + 0.75 * torch.exp(scaled_weight)  # default a = 0.25
        # print('rpn | loss.py | call | scaled_weight : {0}'.format(scaled_weight))
        # final_scaled_weight = scaled_weight.clone()
        # final_scaled_weight[sampled_inds_overlap >= 0.5] = 1
        # print('rpn | loss.py | call | final_scaled_weight : {0}'.format(final_scaled_weight))

        objectness, box_regression = concat_box_prediction_layers(objectness, box_regression)
        objectness = objectness.squeeze()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = smooth_l1_loss(box_regression[sampled_pos_inds], regression_targets[sampled_pos_inds], beta=1.0/9, size_average=False) / (sampled_inds.numel())
        # print('rpn | loss.py | call | box_loss : {0}'.format(box_loss))

        objectness_loss = F.binary_cross_entropy_with_logits(objectness[sampled_inds], labels[sampled_inds], weight=None, size_average=None, reduce=None, reduction='none')
        original_objectness_loss = torch.mean(objectness_loss)
        # print('rpn | loss.py | call | original_objectness_loss : {0}'.format(original_objectness_loss))

        scaled_objectness_loss = objectness_loss * scaled_weight
        scaled_objectness_loss = torch.mean(scaled_objectness_loss)
        # print('rpn | loss.py | call | scaled_objectness_loss : {0}'.format(scaled_objectness_loss))

        return original_objectness_loss, box_loss


# This function should be overwritten in RetinaNet
def generate_rpn_labels(matched_targets):
    matched_idxs = matched_targets.get_field("matched_idxs")
    labels_per_image = matched_idxs >= 0  # if >=0 means having gt, label = 1; else label = 0
    return labels_per_image


def make_rpn_loss_evaluator(cfg, box_coder):
    matcher = Matcher(cfg.MODEL.RPN.FG_IOU_THRESHOLD, cfg.MODEL.RPN.BG_IOU_THRESHOLD, allow_low_quality_matches=True)
    fg_bg_sampler = BalancedPositiveNegativeSampler(cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION)
    loss_evaluator = RPNLossComputation(matcher, fg_bg_sampler, box_coder, generate_rpn_labels)
    return loss_evaluator
