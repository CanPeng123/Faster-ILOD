# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F
import cv2
import numpy

CATEGORIES = ["__background__ ", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
              "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
font = cv2.FONT_HERSHEY_SIMPLEX


def draw_image(input_images, input_targets, input_proposals, index):
    # draw the image, targets and proposals
    output_image_path = "/home/incremental_learning/external_proposals_img/check/%s.jpg" % index
    # draw input image; convert PIL to numpy
    img = cv2.cvtColor(numpy.asarray(input_images), cv2.COLOR_RGB2BGR)
    # draw input proposals
    proposal_bbox = input_proposals.bbox.to("cpu").detach().numpy()
    for i in range(3):
        left = int(proposal_bbox[i][0])
        top = int(proposal_bbox[i][1])
        right = int(proposal_bbox[i][2])
        bottom = int(proposal_bbox[i][3])
        cv2.rectangle(img, (left, top), (right, bottom), (255, 255, 255), 3)  # last two elements: (B, G, R), thickness
    # draw input targets
    gt_bbox = input_targets.bbox
    gt_label = input_targets.get_field("labels")
    for i in range(len(gt_label)):
        left = int(gt_bbox[i][0])
        top = int(gt_bbox[i][1])
        right = int(gt_bbox[i][2])
        bottom = int(gt_bbox[i][3])
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)  # last two elements: (B, G, R), thickness
        cv2.putText(img, '{0}'.format(CATEGORIES[gt_label[i]]), (right, bottom + 5), font, 0.5, (0, 0, 255), 1)
    cv2.imwrite(output_image_path, img)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
        # self.index = 0

    def __call__(self, image, target, proposal=None):

        for t in self.transforms:
            # draw_image(image, target, proposal, self.index)
            # self.index = self.index + 1
            image, target, proposal = t(image, target, proposal)

        return image, target, proposal

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        # self.index = 0

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target, proposal):

        # draw_image(image, target, proposal, self.index)
        # self.index = self.index + 1
        size = self.get_size(image.size)
        image = F.resize(image, size)
        if target is not None:
            target = target.resize(image.size)
        if proposal is not None:
            proposal = proposal.resize(image.size)
        # draw_image(image, target, proposal, self.index)
        # self.index = self.index + 1

        return image, target, proposal


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob
        # self.index = 0

    def __call__(self, image, target, proposal):

        # draw_image(image, target, proposal, self.index)
        # self.index = self.index + 1
        if random.random() < self.prob:
            image = F.hflip(image)
            if target is not None:
                target = target.transpose(0)
            if proposal is not None:
                proposal = proposal.transpose(0)
        # draw_image(image, target, proposal, self.index)
        # self.index = self.index + 1

        return image, target, proposal


class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, target, proposal):
        image = self.color_jitter(image)
        return image, target, proposal


class ToTensor(object):
    def __call__(self, image, target, proposal):
        return F.to_tensor(image), target, proposal


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target, proposal):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target, proposal
