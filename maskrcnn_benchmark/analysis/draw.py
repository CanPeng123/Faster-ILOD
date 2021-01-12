import os
import torch
import torch.distributed as dist
from torch import nn
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
import matplotlib

CONFIDENCE_THRESHOLD = 0.7
CATEGORIES = ["__background__ ", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
              "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
font = cv2.FONT_HERSHEY_SIMPLEX


def display_soften_result(idx, data_loader, soften_result, groundtruth):

    print('--------------------------------------------------------------------------------------')
    print('soften result: {0}'.format(soften_result[0]))
    img_id = data_loader.dataset.get_img_id(idx[0])
    print('id of the input image: {0}'.format(img_id))
    img_info = data_loader.dataset.get_img_info(idx[0])  # get original image width & height
    original_img_height = img_info['height']
    original_img_width = img_info['width']
    print('size of the input image: width={0}, height={1}'.format(original_img_width, original_img_height))
    soften_result = soften_result[0].resize((original_img_width, original_img_height))
    soften_bbox = soften_result.bbox.to("cpu").detach().numpy()
    soften_score = soften_result.get_field("scores").detach().to("cpu").numpy()
    soften_label = soften_result.get_field("labels").detach().to("cpu").numpy()
    print('prediction bounding box result: {0}'.format(soften_bbox))
    print('prediction score: {0}'.format(soften_score))
    print('prediction label : {0}'.format(soften_label))
    input_image_path = "/home/s4401040/nas_home/DATA/VOC2007/JPEGImages/%s.jpg" % img_id
    output_image_path = "/home/s4401040/nas_home/incremental_learning/soften_prediction/%s.jpg" % img_id
    print('input_image_path: {0}'.format(input_image_path))
    print('output_image_path: {0}'.format(output_image_path))
    img = cv2.imread(input_image_path)
    # draw prediction result
    for i in range(len(soften_score)):
        if soften_score[i] > CONFIDENCE_THRESHOLD:
            left = int(soften_bbox[i][0])
            top = int(soften_bbox[i][1])
            right = int(soften_bbox[i][2])
            bottom = int(soften_bbox[i][3])
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)  # last two elements: (B, G, R), thickness
            cv2.putText(img, '{0}: {1}'.format(CATEGORIES[soften_label[i]], soften_score[i]), (right, bottom), font,
                        0.5, (0, 255, 0), 1)
    # draw ground truth
    print('ground truth: {0}'.format(groundtruth))
    groundtruth = groundtruth[0].resize((original_img_width, original_img_height))
    gt_bbox = groundtruth.bbox
    gt_label = groundtruth.get_field("labels")
    print('ground truth bounding box result: {0}'.format(gt_bbox))
    print('ground truth label: {0}'.format(gt_label))
    for i in range(len(gt_label)):
        left = int(gt_bbox[i][0])
        top = int(gt_bbox[i][1])
        right = int(gt_bbox[i][2])
        bottom = int(gt_bbox[i][3])
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)  # last two elements: (B, G, R), thickness
        cv2.putText(img, '{0}'.format(CATEGORIES[gt_label[i]]), (right, bottom + 5), font, 0.5, (0, 0, 255), 1)
    cv2.imwrite(output_image_path, img)
    print('--------------------------------------------------------------------------------------')


def display_soften_proposal(image, idx, data_loader, soften_proposal):

    img_id = data_loader.dataset.get_img_id(idx[0])
    # print('id of the input image: {0}'.format(img_id))
    img_info = data_loader.dataset.get_img_info(idx[0])  # get original image width & height
    original_img_height = img_info['height']
    original_img_width = img_info['width']
    # print('size of the input image: width={0}, height={1}'.format(original_img_width, original_img_height))

    input_image_path = "/home/s4401040/nas_home/DATA/VOC2007/JPEGImages/%s.jpg" % img_id
    output_image_path_1 = "/home/s4401040/nas_home/incremental_learning_ResNet50_C4/figure/soften_proposal/%s_1.jpg" % img_id
    output_image_path_2 = "/home/s4401040/nas_home/incremental_learning_ResNet50_C4/figure/soften_proposal/%s_2.jpg" % img_id
    output_image_path_3 = "/home/s4401040/nas_home/incremental_learning_ResNet50_C4/figure/soften_proposal/%s_3.jpg" % img_id
    output_image_path_4 = "/home/s4401040/nas_home/incremental_learning_ResNet50_C4/figure/soften_proposal/%s_4.jpg" % img_id

    """
    # draw soften_proposal on original image
    img = cv2.imread(input_image_path)
    rescaled_soften_proposal = soften_proposal[0].resize((original_img_width, original_img_height))
    for i in range(len(rescaled_soften_proposal)):
        left = int(rescaled_soften_proposal.bbox[i][0])
        top = int(rescaled_soften_proposal.bbox[i][1])
        right = int(rescaled_soften_proposal.bbox[i][2])
        bottom = int(rescaled_soften_proposal.bbox[i][3])
        # print('on original img | left: {0}, top: {1}, right: {2}, bottom: {3}'.format(left, top, right, bottom))
        cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)  # last two elements: (B, G, R), thickness
    cv2.imwrite(output_image_path_1, img)

    img = cv2.imread(input_image_path)
    rescaled_soften_proposal = soften_proposal[0].resize((original_img_width, original_img_height))
    for i in range(len(rescaled_soften_proposal)):
        left = int(rescaled_soften_proposal.bbox[i][0])
        top = int(rescaled_soften_proposal.bbox[i][1])
        right = int(rescaled_soften_proposal.bbox[i][2])
        bottom = int(rescaled_soften_proposal.bbox[i][3])
        # print('on original img | left: {0}, top: {1}, right: {2}, bottom: {3}'.format(left, top, right, bottom))
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 0), -1)  # last two elements: (B, G, R), thickness
    cv2.imwrite(output_image_path_2, img)
    """
    # draw soften_proposal on processed image
    processed_img_PIL = image.tensors.detach().to("cpu")
    processed_img_PIL = processed_img_PIL[0, :, :, :]
    # print('processed_img_PIL size: {0}'.format(processed_img_PIL.size()))
    processed_img_PIL = transforms.ToPILImage()(processed_img_PIL).convert('RGB')
    processed_img_PIL.save(output_image_path_3)

    processed_img = cv2.imread(output_image_path_3)
    for i in range(len(soften_proposal[0])):
        left = int(soften_proposal[0].bbox[i][0])
        top = int(soften_proposal[0].bbox[i][1])
        right = int(soften_proposal[0].bbox[i][2])
        bottom = int(soften_proposal[0].bbox[i][3])
        # print('on original img | left: {0}, top: {1}, right: {2}, bottom: {3}'.format(left, top, right, bottom))
        cv2.rectangle(processed_img, (left, top), (right, bottom), (255, 0, 0), -1)  # last two elements: (B, G, R), thickness
    cv2.imwrite(output_image_path_4, processed_img)


def draw_feature_mask(feature_mask, idx, data_loader):
    img_id = data_loader.dataset.get_img_id(idx[0])
    # print('id of the input image: {0}'.format(img_id))
    img_info = data_loader.dataset.get_img_info(idx[0])  # get original image width & height
    original_img_height = img_info['height']
    original_img_width = img_info['width']
    # print('size of the input image: width={0}, height={1}'.format(original_img_width, original_img_height))

    output_image_path = "/home/s4401040/nas_home/incremental_learning_ResNet50_C4/figure/feature_mask/%s.jpg" % img_id
    feature_mask_img = feature_mask.detach().to("cpu")
    feature_mask_img = feature_mask_img[0, 0, :, :]
    # print('feature_mask_img size : {0}'.format(feature_mask_img.size()))

    feature_mask_img_PIL = transforms.ToPILImage()(feature_mask_img).convert('L')
    feature_mask_img_PIL.save(output_image_path)










