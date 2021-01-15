# Faster-ILOD
This project hosts the code for implementing the Faster ILOD algorithm for incremental object detection, as presented in our paper:

## [Faster ILOD: Incremental Learning for Object Detectors based on Faster RCNN] (https://www.sciencedirect.com/science/article/pii/S0167865520303627)

Can Peng, Kun Zhao and Brian C. Lovell; In: Pattern Recognition Letters 2020.

[arXiv preprint] (https://arxiv.org/abs/2003.03901).

# Installation

This Faster ILOD implementation is based on [maskrcnn-benchmark] (https://github.com/facebookresearch/maskrcnn-benchmark).

Therefore the installation is the same as the original maskrcnn-benchmark.

Please check INSTALL.md for installation instructions. You may also want to see the original [README.md of maskrcnn-benchmark] (https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/README.md).

# Training

## The files used to train Faster ILOD models are under [Faster-ILOD/tools folder] (https://github.com/CanPeng123/Faster-ILOD/tree/main/tools).

**train_first_step.py**: normally train the first task (standard training). 

**train_incremental.py**: incrementally train the following tasks (knowledge distillation based training).

## The config settings for the models and datasets are under [Faster-ILOD/configs folder] (https://github.com/CanPeng123/Faster-ILOD/tree/main/configs).

### VOC dataset training

**e2e_faster_rcnn_R_50_C4_1x_Source_model.yaml**: config and dataset settings for source model (ResNet50) trained on VOC dataset.

**e2e_faster_rcnn_R_50_C4_1x_Target_model.yaml**: config and dataset settings for target model (ResNet50) trained on VOC dataset.

Put the name of old class categories (all previously trained categories) on NAME_OLD_CLASSES.

Put the name of new class categories (categories for current training task) on NAME_NEW_CLASSES.

Put the name of excluded categories (categories not used, since VOC has 20 categories) on NAME_EXCLUDED_CLASSES.

The code for loading VOC dataset to the model is written [here](https://github.com/CanPeng123/Faster-ILOD/blob/main/maskrcnn_benchmark/data/datasets/voc.py).

The path for putting VOC dataset is [here] (https://github.com/CanPeng123/Faster-ILOD/blob/main/maskrcnn_benchmark/config/paths_catalog.py).

### COCO dataset training

**e2e_faster_rcnn_R_50_C4_1x_Source_model_COCO.yaml**: config and dataset settings for source model (ResNet50) trained on COCO dataset.

**e2e_faster_rcnn_R_50_C4_1x_Target_model_COCO.yaml**: config and dataset settings for target model (ResNet50) trained on COCO dataset.

Put the number of classes on NUM_CLASSES. The categories are added in alphabetical orders.

The code for loading COCO dataset to the model is written [here](https://github.com/CanPeng123/Faster-ILOD/blob/main/maskrcnn_benchmark/data/datasets/coco.py).

The path for putting COCO dataset is [here] (https://github.com/CanPeng123/Faster-ILOD/blob/main/maskrcnn_benchmark/config/paths_catalog.py).

## Distillation Loss

The code for calculating feature, RPN, and RCN distillation losses are written [here] https://github.com/CanPeng123/Faster-ILOD/blob/main/maskrcnn_benchmark/distillation/distillation.py.

## Citations

Please consider citing the following paper in your publications if it helps your research. 
@article{peng2020faster,
  title={Faster ILOD: Incremental Learning for Object Detectors based on Faster RCNN},
  author={Peng, Can and Zhao, Kun and Lovell, Brian C},
  journal={Pattern Recognition Letters},
  year={2020}
}

## Acknowledgements
Our Faster ILOD implementation is based on [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). We thanks the authors for making their code public.
