import argparse
import os.path as osp
import re
import torch
from collections import OrderedDict


def convert(in_file, out_file, with_det_head=True):
    """This script is used to count ReDet's model size
    As ReResNet contains some unnecessary keys, so we remove them for counting the real model size.
    """
    remove_keys = ['_basisexpansion', 'filter', 'expanded_bias', 'indices', 'bias_expansion']
    # with_det_head: Counting the size of detection head or not
    # if with_det_head==False, the output .pth only contains the backbone
    if not with_det_head:
        remove_keys.extend(['rpn_head', 'bbox_head', 'rbbox_head'])
    checkpoint = torch.load(in_file)
    # remove optimizer
    if 'optimizer' in checkpoint.keys():
        checkpoint.pop('optimizer')

    in_state_dict = checkpoint.pop('state_dict')
    out_state_dict = OrderedDict()
    for key, val in in_state_dict.items():
        flag = 0
        for item in remove_keys:
            if item in key:
                flag = 1
        if flag == 0:
            out_state_dict[key] = val
    checkpoint['state_dict'] = out_state_dict
    torch.save(checkpoint, out_file)


def main():
    parser = argparse.ArgumentParser(description='ReDet model size counter')
    parser.add_argument('input_file', help='input checkpoint file')
    parser.add_argument('output_file', help='output checkpoint file')
    args = parser.parse_args()
    in_file = args.input_file
    out_file = args.output_file
    convert(in_file, out_file)


if __name__ == '__main__':
    main()
