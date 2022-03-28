from e2cnn import gspaces
import math
import e2cnn.nn as enn


def regular_feature_type(gspace: gspaces.GSpace, planes: int, fixparams: bool = False):
    """ build a regular feature map with the specified number of channels"""
    assert gspace.fibergroup.order() > 0

    N = gspace.fibergroup.order()

    if fixparams:
        planes *= math.sqrt(N)

    planes = planes / N
    planes = int(planes)

    return enn.FieldType(gspace, [gspace.regular_repr] * planes)


def trivial_feature_type(gspace: gspaces.GSpace, planes: int, fixparams: bool = False):
    """ build a trivial feature map with the specified number of channels"""

    if fixparams:
        planes *= math.sqrt(gspace.fibergroup.order())

    planes = int(planes)
    return enn.FieldType(gspace, [gspace.trivial_repr] * planes)


FIELD_TYPE = {
    "trivial": trivial_feature_type,
    "regular": regular_feature_type,
}


def conv3x3(gspace, inplanes, out_planes, stride=1, padding=1, dilation=1, bias=False, fixparams=False):
    """3x3 convolution with padding"""
    in_type = FIELD_TYPE['regular'](gspace, inplanes, fixparams=fixparams)
    out_type = FIELD_TYPE['regular'](gspace, out_planes, fixparams=fixparams)
    return enn.R2Conv(in_type, out_type, 3,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3 * r)


def conv1x1(gspace, inplanes, out_planes, stride=1, padding=0, dilation=1, bias=False, fixparams=False):
    """1x1 convolution"""
    in_type = FIELD_TYPE['regular'](gspace, inplanes, fixparams=fixparams)
    out_type = FIELD_TYPE['regular'](gspace, out_planes, fixparams=fixparams)
    return enn.R2Conv(in_type, out_type, 1,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3 * r)


def convnxn(gspace, inplanes, out_planes, kernel_size=3, stride=1, padding=0, groups=1, bias=False, dilation=1, fixparams=False):
    in_type = FIELD_TYPE['regular'](gspace, inplanes, fixparams=fixparams)
    out_type = FIELD_TYPE['regular'](gspace, out_planes, fixparams=fixparams)
    return enn.R2Conv(in_type, out_type, kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      bias=bias,
                      dilation=dilation,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3 * r)


def build_norm_layer(cfg, gspace, num_features, postfix=''):
    in_type = FIELD_TYPE['regular'](gspace, num_features)
    return 'bn' + str(postfix), enn.InnerBatchNorm(in_type)


def ennReLU(gspace, inplanes, inplace=True):
    in_type = FIELD_TYPE['regular'](gspace, inplanes)
    return enn.ReLU(in_type, inplace=inplace)


def ennInterpolate(gspace, inplanes, scale_factor, mode='nearest', align_corners=False):
    in_type = FIELD_TYPE['regular'](gspace, inplanes)
    return enn.R2Upsampling(in_type, scale_factor, mode=mode, align_corners=align_corners)


def ennMaxPool(gspace, inplanes, kernel_size, stride=1, padding=0):
    in_type = FIELD_TYPE['regular'](gspace, inplanes)
    return enn.PointwiseMaxPool(in_type, kernel_size=kernel_size, stride=stride, padding=padding)

