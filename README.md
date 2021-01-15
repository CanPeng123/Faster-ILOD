# Faster-ILOD
This project hosts the code for implementing the Faster ILOD algorithm for incremental object detection, as presented in our paper:

## Faster ILOD: Incremental Learning for Object Detectors based on Faster RCNN

Can Peng, Kun Zhao and Brian C. Lovell;

In: Pattern Recognition Letters 2020.

arXiv preprint 	arXiv:2003.03901

The full paper is available at: https://arxiv.org/abs/2003.03901.

# Installation

This Faster ILOD implementation is based on maskrcnn-benchmark. 

https://github.com/facebookresearch/maskrcnn-benchmark

Therefore the installation is the same as original maskrcnn-benchmark.

Please check INSTALL.md for installation instructions. You may also want to see the original README.md of maskrcnn-benchmark.

# Training

## The files used to train Faster ILOD models are put inside Faster-ILOD/tools folder.

train_first_step.py: is used to nomally train the first task (standard training). 

train_incremental.py: is used to incrementally train the following tasks (knowledge distillation based training).

## The config settings for the models and datasets are put inside Faster-ILOD/configs folder.

### VOC dataset training

e2e_faster_rcnn_R_50_C4_1x_Source_model.yaml: config and dataset setting for source model (ResNet50) trained on VOC dataset.

e2e_faster_rcnn_R_50_C4_1x_Target_model.yaml: config and dataset setting for target model (ResNet50) trained on VOC dataset.

Training dataset: "voc_2007_train", "voc_2007_val"

Test dataset: "voc_2007_test"

Put the name of old class categories (all previously trained categories) on NAME_OLD_CLASSES.

Put the name of new class categories (categories for current training task) on NAME_NEW_CLASSES.

Put the name of excluded categories (categories not used, VOC totally has 20 categories) on NAME_EXCLUDED_CLASSES.

The code for loading voc dataset to the model is written on Faster-ILOD/tools

### COCO dataset training

e2e_faster_rcnn_R_50_C4_1x_Source_model_COCO.yaml: config and dataset setting for source model (ResNet50) trained on COCO dataset.

e2e_faster_rcnn_R_50_C4_1x_Target_model_COCO.yaml: config and dataset setting for target model (ResNet50) trained on COCO dataset.
