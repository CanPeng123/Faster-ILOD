# Faster-ILOD
This project hosts the code for implementing the Faster ILOD algorithm for incremental object detection, as presented in our paper:

## [Faster ILOD: Incremental Learning for Object Detectors based on Faster RCNN](https://www.sciencedirect.com/science/article/pii/S0167865520303627)

Can Peng, Kun Zhao and Brian C. Lovell; In: Pattern Recognition Letters 2020.

[arXiv preprint](https://arxiv.org/abs/2003.03901).

# Installation

This Faster ILOD implementation is based on [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). Therefore the installation is the same as the original maskrcnn-benchmark.

Please check [INSTALL.md](https://github.com/CanPeng123/Faster-ILOD/blob/main/INSTALL.md) for installation instructions. You may also want to see the original [README.md of maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/README.md).

# Training

The files used to train Faster ILOD models are under [Faster-ILOD/tools](https://github.com/CanPeng123/Faster-ILOD/tree/main/tools) folder.

**train_first_step.py**: normally train the first task (standard training). 

**train_incremental.py**: incrementally train the following tasks (knowledge distillation based training).

The config settings for the models and datasets are under [Faster-ILOD/configs](https://github.com/CanPeng123/Faster-ILOD/tree/main/configs) folder.

### VOC dataset training

**e2e_faster_rcnn_R_50_C4_1x_Source_model.yaml**: config and dataset settings for source model (ResNet50) trained on VOC dataset.

**e2e_faster_rcnn_R_50_C4_1x_Target_model.yaml**: config and dataset settings for target model (ResNet50) trained on VOC dataset.

The code for loading VOC dataset to the model is written on the file [Faster-ILOD/maskrcnn_benchmark/data/datasets/voc.py](https://github.com/CanPeng123/Faster-ILOD/blob/main/maskrcnn_benchmark/data/datasets/voc.py).

1. Please modify the path for putting VOC dataset on the file [Faster-ILOD/maskrcnn_benchmark/config/paths_catalog.py](https://github.com/CanPeng123/Faster-ILOD/blob/main/maskrcnn_benchmark/config/paths_catalog.py).

2. Please modify the setting for the name of old class categories (all previously trained categories) on NAME_OLD_CLASSES on the file [e2e_faster_rcnn_R_50_C4_1x_Target_model.yaml](https://github.com/CanPeng123/Faster-ILOD/blob/main/configs/e2e_faster_rcnn_R_50_C4_1x_Target_model.yaml).

3. Please modify the setting for the name of new class categories (categories for current training task) on NAME_NEW_CLASSES on the file [e2e_faster_rcnn_R_50_C4_1x_Target_model.yaml](https://github.com/CanPeng123/Faster-ILOD/blob/main/configs/e2e_faster_rcnn_R_50_C4_1x_Target_model.yaml).

4. Please modify the setting for the name of excluded categories (categories not used, since VOC has 20 categories) on NAME_EXCLUDED_CLASSES on the file [e2e_faster_rcnn_R_50_C4_1x_Target_model.yaml](https://github.com/CanPeng123/Faster-ILOD/blob/main/configs/e2e_faster_rcnn_R_50_C4_1x_Target_model.yaml).

5. Please modify the number of detecting categories on NUM_CLASSES on the file [e2e_faster_rcnn_R_50_C4_1x_Source_model.yaml](https://github.com/CanPeng123/Faster-ILOD/blob/main/configs/e2e_faster_rcnn_R_50_C4_1x_Source_model.yaml) (number of old categories) and the file [e2e_faster_rcnn_R_50_C4_1x_Target_model.yaml](https://github.com/CanPeng123/Faster-ILOD/blob/main/configs/e2e_faster_rcnn_R_50_C4_1x_Target_model.yaml) (number of old and new categories), repectively. 

### COCO dataset training

**e2e_faster_rcnn_R_50_C4_1x_Source_model_COCO.yaml**: config and dataset settings for source model (ResNet50) trained on COCO dataset.

**e2e_faster_rcnn_R_50_C4_1x_Target_model_COCO.yaml**: config and dataset settings for target model (ResNet50) trained on COCO dataset.

The code for loading COCO dataset to the model is written on the file [Faster-ILOD/maskrcnn_benchmark/data/datasets/coco.py](https://github.com/CanPeng123/Faster-ILOD/blob/main/maskrcnn_benchmark/data/datasets/coco.py).

1. Please modify the path for putting COCO dataset on the file [Faster-ILOD/maskrcnn_benchmark/config/paths_catalog.py](https://github.com/CanPeng123/Faster-ILOD/blob/main/maskrcnn_benchmark/config/paths_catalog.py).

2. The categories for COCO dataset training are added in alphabetical orders. Please modify the number of detecting categories on NUM_CLASSES on the file [e2e_faster_rcnn_R_50_C4_1x_Source_model_COCO.yaml](https://github.com/CanPeng123/Faster-ILOD/blob/main/configs/e2e_faster_rcnn_R_50_C4_1x_Source_model_COCO.yaml) (number of old categories) and the file [e2e_faster_rcnn_R_50_C4_1x_Target_model_COCO.yaml](https://github.com/CanPeng123/Faster-ILOD/blob/main/configs/e2e_faster_rcnn_R_50_C4_1x_Target_model_COCO.yaml) (number of old and new categories), repectively. 

## Distillation Loss

The code for calculating feature, RPN, and RCN distillation losses are written on the file [Faster-ILOD/blob/main/maskrcnn_benchmark/distillation/distillation.py](https://github.com/CanPeng123/Faster-ILOD/blob/main/maskrcnn_benchmark/distillation/distillation.py).

## Citations

Please consider citing the following paper in your publications if it helps your research.

```latexlatex
@article{peng2020faster,
  title={Faster ILOD: Incremental Learning for Object Detectors based on Faster RCNN},  
  author={Peng, Can and Zhao, Kun and Lovell, Brian C},  
  journal={Pattern Recognition Letters},  
  year={2020} 
}
```

## Acknowledgements
Our Faster ILOD implementation is based on [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). We thanks the authors for making their code public.
