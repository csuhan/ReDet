import argparse
import cv2
import mmcv
import os
import os.path as osp
from mmcv import Config
from mmdet.apis import init_detector, inference_detector, draw_poly_detections
from mmdet.datasets import get_dataset


def save_det_result(config_file, out_dir, checkpoint_file=None, img_dir=None, score_thr=0.2):
    """Visualize results and save to the disk
    """
    cfg = Config.fromfile(config_file)
    data_test = cfg.data.test
    dataset = get_dataset(data_test)
    classnames = [dataset.CLASSES]
    # use checkpoint path in cfg
    if not checkpoint_file:
        checkpoint_file = osp.join(cfg.work_dir, 'latest.pth')
    # use testset in cfg
    if not img_dir:
        img_dir = data_test.img_prefix

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    img_list = os.listdir(img_dir)
    for img_name in img_list:
        img_path = osp.join(img_dir, img_name)
        img_out_path = osp.join(out_dir, img_name)
        result = inference_detector(model, img_path)
        img = draw_poly_detections(img_path,
                                   result,
                                   classnames,
                                   scale=1.0,
                                   threshold=score_thr,
                                   colormap=[(212, 188, 0)])
        print(img_out_path)
        cv2.imwrite(img_out_path, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference demo')
    parser.add_argument('config_file', help='input config file')
    parser.add_argument('out_dir', help='output dir')
    parser.add_argument('img_dir', help='img dir')
    args = parser.parse_args()

    dota_colormap = [
        (54, 67, 244),
        (99, 30, 233),
        (176, 39, 156),
        (183, 58, 103),
        (181, 81, 63),
        (243, 150, 33),
        (212, 188, 0),
        (136, 150, 0),
        (80, 175, 76),
        (74, 195, 139),
        (57, 220, 205),
        (59, 235, 255),
        (0, 152, 255),
        (34, 87, 255),
        (72, 85, 121)]

    hrsc2016_colormap = [(212, 188, 0)]
    text_colormap = [(212, 188, 0)]
    save_det_result(args.config_file, args.out_dir,
                    img_dir=args.img_dir, score_thr=0.5)
