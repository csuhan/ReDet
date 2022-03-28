import argparse
import mmcv
import torch
from collections import OrderedDict
from mmcv.runner import load_checkpoint

from mmcls.models import build_classifier


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert ReResNet to standard pytorch layers')
    parser.add_argument('config', help="config file path")
    parser.add_argument('in_weight', help="input weights of ReResNet")
    parser.add_argument(
        'out_weight', help="output weights of standard pytorch layers")
    args = parser.parse_args()

    return args


def convert_ReResNet_to_pytorch(config, in_weight, out_weight):

    ckpt = torch.load(in_weight)
    old_state_dict = ckpt["state_dict"]

    cfg = mmcv.Config.fromfile(config)
    model = build_classifier(cfg.model)
    _ = load_checkpoint(model, in_weight, map_location='cpu')

    # export to pytorch layers
    backbone_dict = model.backbone.export().state_dict()

    new_state_dict = OrderedDict()
    print("copy detection head of the original model")
    for key in old_state_dict.keys():
        if 'backbone' in key:
            continue
        new_state_dict[key] = old_state_dict[key]
    print("copy converted backbone")
    for key in backbone_dict.keys():
        new_state_dict["backbone." + key] = backbone_dict[key]

    ckpt["state_dict"] = new_state_dict
    print("save converted weights to {}".format(out_weight))
    torch.save(ckpt, out_weight)

if __name__ == '__main__':
    args = parse_args()
    convert_ReResNet_to_pytorch(args.config, args.in_weight, args.out_weight)
