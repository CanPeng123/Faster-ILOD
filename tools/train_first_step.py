# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg  # import default model configuration: config/defaults.py, config/paths_catalog.py, yaml file
from maskrcnn_benchmark.data import make_data_loader  # import dataset
from maskrcnn_benchmark.solver import make_lr_scheduler  # learning rate updating strategy
from maskrcnn_benchmark.solver import make_optimizer  # setting the optimizer
from maskrcnn_benchmark.engine.inference import inference  # inference
from maskrcnn_benchmark.engine.trainer import do_train  # main logic of model training
from maskrcnn_benchmark.modeling.detector import build_detection_model  # used to create model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank  # related to multi-gpu training; when usong 1 gpu, get_rank() will return 0
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger  # related to logging model(output training status)
from maskrcnn_benchmark.utils.miscellaneous import mkdir  # related to folder creation

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')


def train(cfg, local_rank, distributed):
    # this command called ./maskrcnn_benchmark/modeling/detector/build_detection_model() function
    # this function is used to create the target model structure according to the setting within input yaml file
    # this function will return the desired structure model
    model = build_detection_model(cfg)

    # default is "cuda"
    device = torch.device(cfg.MODEL.DEVICE)
    # move the model to device
    model.to(device)

    # make_optimizer() function capsulate torch.optiom.SGD() function,
    # according to tensor's required_grad properties to generate parameter updating tabel
    optimizer = make_optimizer(cfg, model)

    # according to configuration within yaml file sets the learning rate updating strategy
    scheduler = make_lr_scheduler(cfg, optimizer)

    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    # if multiple gpus are used, parallel processing data
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    # create a parameter dictionary and initialize the iteration number to 0
    arguments = {}
    arguments["iteration"] = 0

    # path to store the trained parameter value
    output_dir = cfg.OUTPUT_DIR

    # when only use 1 gpu, get_rank() returns 0
    save_to_disk = get_rank() == 0

    # create check pointer
    checkpointer = DetectronCheckpointer(cfg, model, optimizer, scheduler, output_dir, save_to_disk)

    # load the pre-trained model parameter to current model
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)

    # dict updating method to update the parameter dictionary
    arguments.update(extra_checkpoint_data)

    # load training data
    # type of data_loader is list, type of its inside elements is torch.utils.data.DataLoader
    # When is_train=True, make sure cfg.DATASETS.TRAIN is a list
    # it has to point to one or multiple annotation files
    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,  # whether using multiple gpus to train
        start_iter=arguments["iteration"],
    )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD  # number of iteration to store parameter value in pth file

    # train the model: call function ./maskrcnn_benchmark/engine/trainer.py do_train() function
    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
    )

    return model


def run_test(cfg, model, distributed):
    if distributed:  # whether use multiple gpu to train
        model = model.module

    torch.cuda.empty_cache()  # TODO check if it helps. Release unoccupied memory

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)

    # according to number of test dataset decides number of output folder
    output_folders = [None] * len(cfg.DATASETS.TEST)

    # create folder to store test result
    # output result is stored in: cfg.OUTPUT_DIR/inference/cfg.DATASETS.TEST
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder

    # load testing data
    data_loaders_val = make_data_loader(
        cfg,
        is_train=False,
        is_distributed=distributed
    )

    # test
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )

        # synchronize function for multiple gpu inference
        synchronize()


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0
    )
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # get the number of gpu from the device
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    # if there is more than 1 gpu, set initialization for distribute training
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))

    # open and read the input yaml file, store it on config_str and display on the screen
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    # strat to train the model
    model = train(cfg, args.local_rank, args.distributed)

    if not args.skip_test:
        # start to test the trained model
        run_test(cfg, model, args.distributed)


if __name__ == "__main__":
    main()

