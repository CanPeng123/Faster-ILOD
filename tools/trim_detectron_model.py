import os
import torch
import argparse
from maskrcnn_benchmark.utils.c2_model_loading import load_c2_format


parser = argparse.ArgumentParser(description="Trim Detection weights and save in PyTorch format.")
parser.add_argument(
    "--pretrained_path",
    default="/home/incremental_learning_ResNet50_C4/coco/multi_step/75_1_1_1_1_1/step_4/model_final.pth",
    help="path to load the pretrained weight",
    type=str,
)
parser.add_argument(
    "--save_path",
    default="/home/incremental_learning_ResNet50_C4/coco/multi_step/75_1_1_1_1_1/step_4/model_trim_optimizer_iteration.pth",
    help="path to save the converted model",
    type=str,
)

args = parser.parse_args()

PRETRAINED_PATH = os.path.expanduser(args.pretrained_path)
print('pretrained model path: {}'.format(PRETRAINED_PATH))

# remove optimizer and iteration information, only remain model parameter and structure information
pretrained_weights = torch.load(PRETRAINED_PATH)['model']
print('pretrained weights: {0}'.format(pretrained_weights))

new_dict = {k: v for k, v in pretrained_weights.items()}

print('new dict: {0}'.format(new_dict))

torch.save(new_dict, args.save_path)
print('saved to {}.'.format(args.save_path))
