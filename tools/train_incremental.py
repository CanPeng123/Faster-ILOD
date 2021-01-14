# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import datetime
import logging
import time
import torch
import torch.distributed as dist
from torch import nn
import numpy as np
import cv2
from PIL import Image

from maskrcnn_benchmark.config import cfg  # import default model configuration: config/defaults.py, config/paths_catalog.py, yaml file
from maskrcnn_benchmark.data import make_data_loader  # import data set
from maskrcnn_benchmark.engine.inference import inference  # inference
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict  # when multiple gpus are used, reduce the loss
from maskrcnn_benchmark.modeling.detector import build_detection_model  # used to create model
from maskrcnn_benchmark.solver import make_lr_scheduler  # learning rate updating strategy
from maskrcnn_benchmark.solver import make_optimizer  # setting the optimizer
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank  # related to multi-gpu training; when usong 1 gpu, get_rank() will return 0
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger  # related to logging model(output training status)
from maskrcnn_benchmark.utils.miscellaneous import mkdir  # related to folder creation
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from tensorboardX import SummaryWriter
from maskrcnn_benchmark.distillation.distillation import calculate_rpn_distillation_loss
from maskrcnn_benchmark.distillation.distillation import calculate_feature_distillation_loss
from maskrcnn_benchmark.distillation.distillation import calculate_roi_distillation_losses

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def do_train(model_source, model_target, data_loader, optimizer, scheduler, checkpointer_source, checkpointer_target,
             device, checkpoint_period, arguments_source, arguments_target, summary_writer):

    # record log information
    logger = logging.getLogger("maskrcnn_benchmark_target_model.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")  # used to record
    max_iter = len(data_loader)  # data loader rewrites the len() function and allows it to return the number of batches (cfg.SOLVER.MAX_ITER)
    start_iter = arguments_target["iteration"]  # 0
    model_target.train()  # set the target model in training mode
    model_source.eval()  # set the source model in inference mode
    start_training_time = time.time()
    end = time.time()
    average_distillation_loss = 0
    average_faster_rcnn_loss = 0

    for iteration, (images, targets, _, idx) in enumerate(data_loader, start_iter):

        data_time = time.time() - end
        iteration = iteration + 1
        arguments_target["iteration"] = iteration
        scheduler.step()  # update the learning rate

        images = images.to(device)   # move images to the device
        targets = [target.to(device) for target in targets]  # move targets (labels) to the device

        loss_dict_target, feature_target, backbone_feature_target, anchor_target, rpn_output_target = model_target(images, targets)
        faster_rcnn_losses = sum(loss for loss in loss_dict_target.values())  # summarise the losses for faster rcnn

        roi_distillation_losses, rpn_output_source, feature_source, backbone_feature_source, soften_result, soften_proposal, feature_proposals \
            = calculate_roi_distillation_losses(model_source, model_target, images)
        # print('roi_distillation_losses: {0}'.format(roi_distillation_losses))
    
        rpn_distillation_losses = calculate_rpn_distillation_loss(rpn_output_source, rpn_output_target, cls_loss='filtered_l2', bbox_loss='l2', bbox_threshold=0.1)
        # print('rpn_distillation_loss: {0}'.format(rpn_distillation_losses))
       
        feature_distillation_losses = calculate_feature_distillation_loss(feature_source, feature_target, loss='normalized_filtered_l1')
        # print('feature_distillation_loss: {0}'.format(feature_distillation_losses))

        distillation_losses = roi_distillation_losses + rpn_distillation_losses + feature_distillation_losses
        # print('distillation_losses: {0}'.format(distillation_losses))

        distillation_dict = {}
        distillation_dict['distillation_loss'] = distillation_losses.clone().detach()
        loss_dict_target.update(distillation_dict)
        # print('loss_dict_target: {0}'.format(loss_dict_target))

        # losses = (faster_rcnn_losses * 1 + distillation_losses * 19)/20
        losses = faster_rcnn_losses + distillation_losses

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict_target)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        if (iteration - 1) > 0:
            average_distillation_loss = (average_distillation_loss * (iteration - 1) + distillation_losses) / iteration
            average_faster_rcnn_loss = (average_faster_rcnn_loss * (iteration - 1) + faster_rcnn_losses) /iteration
        else:
            average_distillation_loss = distillation_losses
            average_faster_rcnn_loss = faster_rcnn_losses

        optimizer.zero_grad()  # clear the gradient cache
        # If mixed precision is not used, this ends up doing nothing, otherwise apply loss scaling for mixed-precision recipe.
        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()  # use back-propagation to update the gradient
        optimizer.step()  # update learning rate

        # time used to do one batch processing
        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)
        # according to time'moving average to calculate how much time needed to finish the training
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        # for every 50 iterations, display the training status
        if iteration % 20 == 0 or iteration == max_iter:
            print('enter logger info')
            logger.info(
                meters.delimiter.join(["eta: {eta}", "iter: {iter}", "{meters}", "lr: {lr:.6f}", "max mem: {memory:.0f}"
                                       ]).format(eta=eta_string, iter=iteration, meters=str(meters),
                                                 lr=optimizer.param_groups[0]["lr"],
                                                 memory=torch.cuda.max_memory_allocated()/1024.0/1024.0))
            # write to tensorboardX
            loss_global_avg = meters.loss.global_avg
            loss_median = meters.loss.median
            # print('loss global average: {0}, loss median: {1}'.format(meters.loss.global_avg, meters.loss.median))
            summary_writer.add_scalar('train_loss_global_avg', loss_global_avg, iteration)
            summary_writer.add_scalar('train_loss_median', loss_median, iteration)
            summary_writer.add_scalar('train_loss_raw', losses_reduced, iteration)
            summary_writer.add_scalar('distillation_losses_raw', distillation_losses, iteration)
            summary_writer.add_scalar('faster_rcnn_losses_raw', faster_rcnn_losses, iteration)
            summary_writer.add_scalar('distillation_losses_avg', average_distillation_loss, iteration)
            summary_writer.add_scalar('faster_rcnn_losses_avg', average_faster_rcnn_loss, iteration)
        # Every time meets the checkpoint_period, save the target model (parameters)
        if iteration % checkpoint_period == 0:
            checkpointer_target.save("model_{:07d}".format(iteration), **arguments_target)
        # When meets the last iteration, save the target model (parameters)
        if iteration == max_iter:
            checkpointer_target.save("model_final", **arguments_target)
    # Display the total used training time
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time/max_iter))


def train(cfg_source, logger_source, cfg_target, logger_target, distributed):

    model_source = build_detection_model(cfg_source)  # create the source model
    model_target = build_detection_model(cfg_target)  # create the target model
    device = torch.device(cfg_source.MODEL.DEVICE)  # default is "cuda"
    model_target.to(device)  # move target model to gpu
    model_source.to(device)  # move source model to gpu
    optimizer = make_optimizer(cfg_target, model_target)  # config optimization strategy
    scheduler = make_lr_scheduler(cfg_target, optimizer)  # config learning rate
    # initialize mixed-precision training
    use_mixed_precision = cfg_target.DTYPE == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model_target, optimizer = amp.initialize(model_target, optimizer, opt_level=amp_opt_level)
    # create a parameter dictionary and initialize the iteration number to 0
    arguments_target = {}
    arguments_target["iteration"] = 0
    arguments_source = {}
    arguments_source["iteration"] = 0
    # path to store the trained parameter value
    output_dir_target = cfg_target.OUTPUT_DIR
    output_dir_source = cfg_source.OUTPUT_DIR
    # create summary writer for tensorboard
    summary_writer = SummaryWriter(log_dir=cfg_target.TENSORBOARD_DIR)
    # when only use 1 gpu, get_rank() returns 0
    save_to_disk = get_rank() == 0
    # create check pointer for source model & load the pre-trained model parameter to source model
    checkpointer_source = DetectronCheckpointer(cfg_source, model_source, optimizer=None, scheduler=None, save_dir=output_dir_source,
                                                save_to_disk=save_to_disk, logger=logger_source)
    extra_checkpoint_data_source = checkpointer_source.load(cfg_source.MODEL.WEIGHT)
    # create check pointer for target model & load the pre-trained model parameter to target model
    checkpointer_target = DetectronCheckpointer(cfg_target, model_target, optimizer=optimizer, scheduler=scheduler, save_dir=output_dir_target,
                                                save_to_disk=save_to_disk, logger=logger_target)
    extra_checkpoint_data_target = checkpointer_target.load(cfg_target.MODEL.WEIGHT)
    # dict updating method to update the parameter dictionary for source model
    arguments_source.update(extra_checkpoint_data_source)
    # dict updating method to update the parameter dictionary for target model
    arguments_target.update(extra_checkpoint_data_target)
    print('start iteration: {0}'.format(arguments_target["iteration"]))
    # load training data
    data_loader = make_data_loader(cfg_target, is_train=True, is_distributed=distributed, start_iter=arguments_target["iteration"])
    print('finish loading data')
    # number of iteration to store parameter value in pth file
    checkpoint_period = cfg_target.SOLVER.CHECKPOINT_PERIOD

    # train the model
    do_train(model_source, model_target, data_loader, optimizer, scheduler, checkpointer_source, checkpointer_target,
             device, checkpoint_period, arguments_source, arguments_target, summary_writer)

    return model_target


def test(cfg_target, model, distributed):

    if distributed:  # whether use multiple gpu to train
        model = model.module
    # Release unoccupied memory
    torch.cuda.empty_cache()
    iou_types = ("bbox",)
    if cfg_target.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg_target.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    # according to number of test dataset decides number of output folder
    output_folders = [None] * len(cfg_target.DATASETS.TEST)
    # create folder to store test result
    # output result is stored in: cfg.OUTPUT_DIR/inference/cfg.DATASETS.TEST
    dataset_names = cfg_target.DATASETS.TEST
    if cfg_target.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg_target.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    # load testing data
    print('loading test data')
    data_loaders_val = make_data_loader(cfg_target, is_train=False, is_distributed=distributed)
    print('finish loading test data')
    # test
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(model,
                  data_loader_val,
                  dataset_name=dataset_name,
                  iou_types=iou_types,
                  box_only=False if cfg_target.MODEL.RETINANET_ON else cfg_target.MODEL.RPN_ONLY,
                  device=cfg_target.MODEL.DEVICE,
                  expected_results=cfg_target.TEST.EXPECTED_RESULTS,
                  expected_results_sigma_tol=cfg_target.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                  output_folder=output_folder,
                  alphabetical_order=cfg_target.TEST.COCO_ALPHABETICAL_ORDER)
        # synchronize function for multiple gpu inference
        synchronize()


def main():
    source_model_config_file = "/home/maskrcnn-benchmark/configs/e2e_faster_rcnn_R_50_C4_1x_Source_model.yaml"
    target_model_config_file = "/home/maskrcnn-benchmark/configs/e2e_faster_rcnn_R_50_C4_1x_Target_model.yaml"
    # source_model_config_file = "/home/maskrcnn-benchmark/configs/e2e_faster_rcnn_R_50_C4_1x_Source_model_COCO.yaml"
    # target_model_config_file = "/home/maskrcnn-benchmark/configs/e2e_faster_rcnn_R_50_C4_1x_Target_model_COCO.yaml"
    local_rank = 0

    # get the number of gpu from the device
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    # distributed: whether using multiple gpus to train
    distributed = num_gpus > 1

    cfg_source = cfg.clone()
    cfg_source.merge_from_file(source_model_config_file)
    cfg_source.freeze()
    cfg_target = cfg.clone()
    cfg_target.merge_from_file(target_model_config_file)
    cfg_target.freeze()

    output_dir_target = cfg_target.OUTPUT_DIR
    if output_dir_target:
        mkdir(output_dir_target)
    output_dir_source = cfg_source.OUTPUT_DIR
    if output_dir_source:
        mkdir(output_dir_source)
    tensorboard_dir = cfg_target.TENSORBOARD_DIR
    if tensorboard_dir:
        mkdir(tensorboard_dir)

    logger_target = setup_logger("maskrcnn_benchmark_target_model", output_dir_target, get_rank())
    logger_target.info("config yaml file for target model: {}".format(target_model_config_file))
    logger_target.info("local rank: {}".format(local_rank))
    logger_target.info("Using {} GPUs".format(num_gpus))
    logger_target.info("Collecting env info (might take some time)")
    logger_target.info("\n" + collect_env_info())
    # open and read the input yaml file, store it on source config_str and display on the screen
    with open(target_model_config_file, "r") as cf:
        target_config_str = "\n" + cf.read()
        logger_target.info(target_config_str)
    logger_target.info("Running with config:\n{}".format(cfg_target))

    logger_source = setup_logger("maskrcnn_benchmark_source_model", output_dir_source, get_rank())
    logger_source.info("config yaml file for target model: {}".format(source_model_config_file))
    logger_source.info("local rank: {}".format(local_rank))
    logger_source.info("Using {} GPUs".format(num_gpus))
    logger_source.info("Collecting env info (might take some time)")
    logger_source.info("\n" + collect_env_info())
    # open and read the input yaml file, store it on source config_str and display on the screen
    with open(source_model_config_file, "r") as cf:
        source_config_str = "\n" + cf.read()
        logger_source.info(source_config_str)
    logger_source.info("Running with config:\n{}".format(cfg_source))

    # start to train the model
    model_target = train(cfg_source, logger_source, cfg_target, logger_target, distributed)
    # start to test the trained target model
    test(cfg_target, model_target, distributed)


if __name__ == "__main__":
    main()

