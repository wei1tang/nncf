"""
 Copyright (c) 2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

"""YOLO_v4 Model Defined in Keras."""

from functools import wraps

import math
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras.engine.base_layer import Layer
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from functools import reduce
import keras.layers as layers

from beta.examples.tensorflow.common.object_detection import base_model
from beta.examples.tensorflow.common.object_detection.architecture import factory
from beta.examples.tensorflow.common.object_detection.architecture import keras_utils
from beta.examples.tensorflow.common.object_detection.ops import postprocess_ops
from beta.examples.tensorflow.common.object_detection.evaluation import coco_evaluator
from beta.examples.tensorflow.common.logger import logger
from beta.examples.tensorflow.common.object_detection import losses

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.
    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

class Mish(Layer):
    '''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X_input = Input(input_shape)
        >>> X = Mish()(X_input)
    '''

    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {}
    darknet_conv_kwargs['kernel_initializer'] = keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def DarknetConv2D_BN_Mish(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        Mish())


def resblock_body(x, num_filters, num_blocks, all_narrow=True):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    preconv1 = ZeroPadding2D(((1,0),(1,0)))(x)
    preconv1 = DarknetConv2D_BN_Mish(num_filters, (3,3), strides=(2,2))(preconv1)
    shortconv = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1))(preconv1)
    mainconv = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1))(preconv1)
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Mish(num_filters//2, (1,1)),
                DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (3,3)))(mainconv)
        mainconv = Add()([mainconv,y])
    postconv = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1))(mainconv)
    route = Concatenate()([postconv, shortconv])
    return DarknetConv2D_BN_Mish(num_filters, (1,1))(route)


def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Mish(32, (3,3))(x)
    x = resblock_body(x, 64, 1, False)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x


def make_last_layers(x, num_filters, out_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)
    y = compose(
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D(out_filters, (1,1)))(x)
    return x, y


def yolo4_body(inputs, num_anchors, num_classes):
    """Create YOLO_V4 model CNN body in Keras."""
    darknet = Model(inputs, darknet_body(inputs))

    #19x19 head
    y19 = DarknetConv2D_BN_Leaky(512, (1,1))(darknet.output)
    y19 = DarknetConv2D_BN_Leaky(1024, (3,3))(y19)
    y19 = DarknetConv2D_BN_Leaky(512, (1,1))(y19)
    maxpool1 = MaxPooling2D(pool_size=(13,13), strides=(1,1), padding='same')(y19)
    maxpool2 = MaxPooling2D(pool_size=(9,9), strides=(1,1), padding='same')(y19)
    maxpool3 = MaxPooling2D(pool_size=(5,5), strides=(1,1), padding='same')(y19)
    y19 = Concatenate()([maxpool1, maxpool2, maxpool3, y19])
    y19 = DarknetConv2D_BN_Leaky(512, (1,1))(y19)
    y19 = DarknetConv2D_BN_Leaky(1024, (3,3))(y19)
    y19 = DarknetConv2D_BN_Leaky(512, (1,1))(y19)

    y19_upsample = compose(DarknetConv2D_BN_Leaky(256, (1,1)), UpSampling2D(2))(y19)

    #38x38 head
    y38 = DarknetConv2D_BN_Leaky(256, (1,1))(darknet.layers[204].output)
    y38 = Concatenate()([y38, y19_upsample])
    y38 = DarknetConv2D_BN_Leaky(256, (1,1))(y38)
    y38 = DarknetConv2D_BN_Leaky(512, (3,3))(y38)
    y38 = DarknetConv2D_BN_Leaky(256, (1,1))(y38)
    y38 = DarknetConv2D_BN_Leaky(512, (3,3))(y38)
    y38 = DarknetConv2D_BN_Leaky(256, (1,1))(y38)

    y38_upsample = compose(DarknetConv2D_BN_Leaky(128, (1,1)), UpSampling2D(2))(y38)

    #76x76 head
    y76 = DarknetConv2D_BN_Leaky(128, (1,1))(darknet.layers[131].output)
    y76 = Concatenate()([y76, y38_upsample])
    y76 = DarknetConv2D_BN_Leaky(128, (1,1))(y76)
    y76 = DarknetConv2D_BN_Leaky(256, (3,3))(y76)
    y76 = DarknetConv2D_BN_Leaky(128, (1,1))(y76)
    y76 = DarknetConv2D_BN_Leaky(256, (3,3))(y76)
    y76 = DarknetConv2D_BN_Leaky(128, (1,1))(y76)

    #76x76 output
    y76_output = DarknetConv2D_BN_Leaky(256, (3,3))(y76)
    y76_output = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(y76_output)

    #38x38 output
    y76_downsample = ZeroPadding2D(((1,0),(1,0)))(y76)
    y76_downsample = DarknetConv2D_BN_Leaky(256, (3,3), strides=(2,2))(y76_downsample)
    y38 = Concatenate()([y76_downsample, y38])
    y38 = DarknetConv2D_BN_Leaky(256, (1,1))(y38)
    y38 = DarknetConv2D_BN_Leaky(512, (3,3))(y38)
    y38 = DarknetConv2D_BN_Leaky(256, (1,1))(y38)
    y38 = DarknetConv2D_BN_Leaky(512, (3,3))(y38)
    y38 = DarknetConv2D_BN_Leaky(256, (1,1))(y38)

    y38_output = DarknetConv2D_BN_Leaky(512, (3,3))(y38)
    y38_output = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(y38_output)

    #19x19 output
    y38_downsample = ZeroPadding2D(((1,0),(1,0)))(y38)
    y38_downsample = DarknetConv2D_BN_Leaky(512, (3,3), strides=(2,2))(y38_downsample)
    y19 = Concatenate()([y38_downsample, y19])
    y19 = DarknetConv2D_BN_Leaky(512, (1,1))(y19)
    y19 = DarknetConv2D_BN_Leaky(1024, (3,3))(y19)
    y19 = DarknetConv2D_BN_Leaky(512, (1,1))(y19)
    y19 = DarknetConv2D_BN_Leaky(1024, (3,3))(y19)
    y19 = DarknetConv2D_BN_Leaky(512, (1,1))(y19)

    y19_output = DarknetConv2D_BN_Leaky(1024, (3,3))(y19)
    y19_output = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(y19_output)

    # yolo4_model = Model(inputs, [y19_output, y38_output, y76_output], name='yolov4')

    # return yolo4_model
    return y19_output, y38_output, y76_output


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3] # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[...,::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[...,::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs
    # box_xy shape=(4, 10, 10, 3, 2)  box_wh  shape=(4, 10, 10, 3, 2)  box_confidence  shape=(4, 10, 10, 3, 1) box_class_probs shape=(4, 10, 10, 3, 80)


def yolo_correct_boxes(box_xy, box_wh, image_shape):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    image_shape = K.cast(image_shape, K.dtype(box_yx))

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # boxes shape=(4, 10, 10, 3, 4)
    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
        anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, image_shape)
    batch_size = K.shape(feats)[0]
    boxes = K.reshape(boxes, [batch_size, -1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [batch_size, -1, num_classes])
    return boxes, box_scores


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape):
    """Evaluate YOLO model on given input and return filtered boxes."""
    num_layers = len(yolo_outputs)
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32    # shape=(4, 10, 10, 255)   1:3  (10, 10)
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
            anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=1)    #  shape=(4, 6300, 4)
    box_scores = K.concatenate(box_scores, axis=1)   # shape=(4, 6300, 80)
 
    return boxes, box_scores

def bbox_iou_data(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]
    boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    return inter_area / union_area

# def preprocess_true_boxes(bboxes, train_output_sizes, strides, num_classes, max_bbox_per_scale, anchors):
#     label = [np.zeros((train_output_sizes[i], train_output_sizes[i], 3,
#                        5 + num_classes)) for i in range(3)]
#     bboxes_xywh = [np.zeros((max_bbox_per_scale, 4)) for _ in range(3)]
#     bbox_count = np.zeros((3,))
#     for bbox in bboxes:
#         bbox_coor = bbox[:4]
#         bbox_class_ind = bbox[4]
#         onehot = np.zeros(num_classes, dtype=np.float)
#         onehot[bbox_class_ind] = 1.0
#         bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
#         bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / strides[:, np.newaxis]
#         iou = []
#         for i in range(3):
#             anchors_xywh = np.zeros((3, 4))
#             anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
#             anchors_xywh[:, 2:4] = anchors[i]
#             iou_scale = bbox_iou_data(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
#             iou.append(iou_scale)
#         best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
#         best_detect = int(best_anchor_ind / 3)
#         best_anchor = int(best_anchor_ind % 3)
#         xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)
#         # 防止越界
#         grid_r = label[best_detect].shape[0]
#         grid_c = label[best_detect].shape[1]
#         xind = max(0, xind)
#         yind = max(0, yind)
#         xind = min(xind, grid_r-1)
#         yind = min(yind, grid_c-1)
#         label[best_detect][yind, xind, best_anchor, :] = 0
#         label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
#         label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
#         label[best_detect][yind, xind, best_anchor, 5:] = onehot
#         bbox_ind = int(bbox_count[best_detect] % max_bbox_per_scale)
#         bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
#         bbox_count[best_detect] += 1
#     label_sbbox, label_mbbox, label_lbbox = label
#     sbboxes, mbboxes, lbboxes = bboxes_xywh
#     return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes


def bbox_ciou(boxes1, boxes2):
    '''
    计算ciou = iou - p2/c2 - av
    :param boxes1: (8, 13, 13, 3, 4)   pred_xywh
    :param boxes2: (8, 13, 13, 3, 4)   label_xywh
    :return:
    举例时假设pred_xywh和label_xywh的shape都是(1, 4)
    '''

    # 变成左上角坐标、右下角坐标
    boxes1_x0y0x1y1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                 boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2_x0y0x1y1 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                 boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
    '''
    逐个位置比较boxes1_x0y0x1y1[..., :2]和boxes1_x0y0x1y1[..., 2:]，即逐个位置比较[x0, y0]和[x1, y1]，小的留下。
    比如留下了[x0, y0]
    这一步是为了避免一开始w h 是负数，导致x0y0成了右下角坐标，x1y1成了左上角坐标。
    '''
    boxes1_x0y0x1y1 = tf.concat([tf.minimum(boxes1_x0y0x1y1[..., :2], boxes1_x0y0x1y1[..., 2:]),
                                 tf.maximum(boxes1_x0y0x1y1[..., :2], boxes1_x0y0x1y1[..., 2:])], axis=-1)
    boxes2_x0y0x1y1 = tf.concat([tf.minimum(boxes2_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., 2:]),
                                 tf.maximum(boxes2_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., 2:])], axis=-1)

    # 两个矩形的面积
    boxes1_area = (boxes1_x0y0x1y1[..., 2] - boxes1_x0y0x1y1[..., 0]) * (
                boxes1_x0y0x1y1[..., 3] - boxes1_x0y0x1y1[..., 1])
    boxes2_area = (boxes2_x0y0x1y1[..., 2] - boxes2_x0y0x1y1[..., 0]) * (
                boxes2_x0y0x1y1[..., 3] - boxes2_x0y0x1y1[..., 1])

    # 相交矩形的左上角坐标、右下角坐标，shape 都是 (8, 13, 13, 3, 2)
    left_up = tf.maximum(boxes1_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., :2])
    right_down = tf.minimum(boxes1_x0y0x1y1[..., 2:], boxes2_x0y0x1y1[..., 2:])

    # 相交矩形的面积inter_area。iou
    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / (union_area + K.epsilon())

    # 包围矩形的左上角坐标、右下角坐标，shape 都是 (8, 13, 13, 3, 2)
    enclose_left_up = tf.minimum(boxes1_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., :2])
    enclose_right_down = tf.maximum(boxes1_x0y0x1y1[..., 2:], boxes2_x0y0x1y1[..., 2:])

    # 包围矩形的对角线的平方
    enclose_wh = enclose_right_down - enclose_left_up
    enclose_c2 = K.pow(enclose_wh[..., 0], 2) + K.pow(enclose_wh[..., 1], 2)

    # 两矩形中心点距离的平方
    p2 = K.pow(boxes1[..., 0] - boxes2[..., 0], 2) + K.pow(boxes1[..., 1] - boxes2[..., 1], 2)

    # 增加av。加上除0保护防止nan。
    atan1 = tf.atan(boxes1[..., 2] / (boxes1[..., 3] + K.epsilon()))
    atan2 = tf.atan(boxes2[..., 2] / (boxes2[..., 3] + K.epsilon()))
    v = 4.0 * K.pow(atan1 - atan2, 2) / (math.pi ** 2)
    a = v / (1 - iou + v)

    ciou = iou - 1.0 * p2 / enclose_c2 - 1.0 * a * v
    return ciou


def bbox_iou(boxes1, boxes2):
    '''
    预测框          boxes1 (?, grid_h, grid_w, 3,   1, 4)，神经网络的输出(tx, ty, tw, th)经过了后处理求得的(bx, by, bw, bh)
    图片中所有的gt  boxes2 (?,      1,      1, 1, 150, 4)
    '''
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]  # 所有格子的3个预测框的面积
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]  # 所有ground truth的面积

    # (x, y, w, h)变成(x0, y0, x1, y1)
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    # 所有格子的3个预测框 分别 和  150个ground truth  计算iou。 所以left_up和right_down的shape = (?, grid_h, grid_w, 3, 150, 2)
    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])  # 相交矩形的左上角坐标
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])  # 相交矩形的右下角坐标

    inter_section = tf.maximum(right_down - left_up, 0.0)  # 相交矩形的w和h，是负数时取0     (?, grid_h, grid_w, 3, 150, 2)
    inter_area = inter_section[..., 0] * inter_section[..., 1]  # 相交矩形的面积            (?, grid_h, grid_w, 3, 150)
    union_area = boxes1_area + boxes2_area - inter_area  # union_area      (?, grid_h, grid_w, 3, 150)
    iou = 1.0 * inter_area / union_area  # iou                             (?, grid_h, grid_w, 3, 150)
    return iou

def loss_layer(conv, pred, label, bboxes, stride, num_class, iou_loss_thresh):
    conv_shape = tf.shape(conv)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = stride * output_size
    conv = tf.reshape(conv, (batch_size, output_size, output_size,
                             3, 5 + num_class))
    conv_raw_prob = conv[:, :, :, :, 5:]

    pred_xywh = pred[:, :, :, :, 0:4]
    pred_conf = pred[:, :, :, :, 4:5]

    label_xywh = label[:, :, :, :, 0:4]
    respond_bbox = label[:, :, :, :, 4:5]
    label_prob = label[:, :, :, :, 5:]

    ciou = tf.expand_dims(bbox_ciou(pred_xywh, label_xywh), axis=-1)  # (8, 13, 13, 3, 1)
    input_size = tf.cast(input_size, tf.float32)

    # 每个预测框xxxiou_loss的权重 = 2 - (ground truth的面积/图片面积)
    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    ciou_loss = respond_bbox * bbox_loss_scale * (1 - ciou)  # 1. respond_bbox作为mask，有物体才计算xxxiou_loss

    # 2. respond_bbox作为mask，有物体才计算类别loss
    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    # 3. xxxiou_loss和类别loss比较简单。重要的是conf_loss，是一个focal_loss
    # 分两步：第一步是确定 grid_h * grid_w * 3 个预测框 哪些作为反例；第二步是计算focal_loss。
    expand_pred_xywh = pred_xywh[:, :, :, :, np.newaxis, :]  # 扩展为(?, grid_h, grid_w, 3,   1, 4)
    expand_bboxes = bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :]  # 扩展为(?,      1,      1, 1, 150, 4)
    iou = bbox_iou(expand_pred_xywh, expand_bboxes)  # 所有格子的3个预测框 分别 和  150个ground truth  计算iou。   (?, grid_h, grid_w, 3, 150)
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)  # 与150个ground truth的iou中，保留最大那个iou。  (?, grid_h, grid_w, 3, 1)

    # respond_bgd代表  这个分支输出的 grid_h * grid_w * 3 个预测框是否是 反例（背景）
    # label有物体，respond_bgd是0。 没物体的话：如果和某个gt(共150个)的iou超过iou_loss_thresh，respond_bgd是0；如果和所有gt(最多150个)的iou都小于iou_loss_thresh，respond_bgd是1。
    # respond_bgd是0代表有物体，不是反例；  权重respond_bgd是1代表没有物体，是反例。
    # 有趣的是，模型训练时由于不断更新，对于同一张图片，两次预测的 grid_h * grid_w * 3 个预测框（对于这个分支输出）  是不同的。用的是这些预测框来与gt计算iou来确定哪些预测框是反例。
    # 而不是用固定大小（不固定位置）的先验框。
    respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < iou_loss_thresh, tf.float32)

    # 二值交叉熵损失
    pos_loss = respond_bbox * (0 - K.log(pred_conf + K.epsilon()))
    neg_loss = respond_bgd  * (0 - K.log(1 - pred_conf + K.epsilon()))

    conf_loss = pos_loss + neg_loss
    # 回顾respond_bgd，某个预测框和某个gt的iou超过iou_loss_thresh，不被当作是反例。在参与“预测的置信位 和 真实置信位 的 二值交叉熵”时，这个框也可能不是正例(label里没标这个框是1的话)。这个框有可能不参与置信度loss的计算。
    # 这种框一般是gt框附近的框，或者是gt框所在格子的另外两个框。它既不是正例也不是反例不参与置信度loss的计算。（论文里称之为ignore）

    ciou_loss = tf.reduce_mean(tf.reduce_sum(ciou_loss, axis=[1, 2, 3, 4]))  # 每个样本单独计算自己的ciou_loss，再求平均值
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))  # 每个样本单独计算自己的conf_loss，再求平均值
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))  # 每个样本单独计算自己的prob_loss，再求平均值

    return ciou_loss, conf_loss, prob_loss

def decode(conv_output, anchors, stride, num_class):
    conv_shape       = tf.shape(conv_output)
    batch_size       = conv_shape[0]
    output_size      = conv_shape[1]
    anchor_per_scale = len(anchors)
    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, anchor_per_scale, 5 + num_class))
    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5: ]
    y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
    x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])
    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)
    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
    pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)
    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

def yolo_loss(args, num_classes, iou_loss_thresh, anchors):
    conv_lbbox = args[0]   # (?, ?, ?, 3*(num_classes+5))
    conv_mbbox = args[1]   # (?, ?, ?, 3*(num_classes+5))
    conv_sbbox = args[2]   # (?, ?, ?, 3*(num_classes+5))
    label_sbbox = args[3]   # (?, ?, ?, 3, num_classes+5)
    label_mbbox = args[4]   # (?, ?, ?, 3, num_classes+5)
    label_lbbox = args[5]   # (?, ?, ?, 3, num_classes+5)
    true_sbboxes = args[6]   # (?, 150, 4)
    true_mbboxes = args[7]   # (?, 150, 4)
    true_lbboxes = args[8]   # (?, 150, 4)
    pred_sbbox = decode(conv_sbbox, anchors[0], 8, num_classes)
    pred_mbbox = decode(conv_mbbox, anchors[1], 16, num_classes)
    pred_lbbox = decode(conv_lbbox, anchors[2], 32, num_classes)
    sbbox_ciou_loss, sbbox_conf_loss, sbbox_prob_loss = loss_layer(conv_sbbox, pred_sbbox, label_sbbox, true_sbboxes, 8, num_classes, iou_loss_thresh)
    mbbox_ciou_loss, mbbox_conf_loss, mbbox_prob_loss = loss_layer(conv_mbbox, pred_mbbox, label_mbbox, true_mbboxes, 16, num_classes, iou_loss_thresh)
    lbbox_ciou_loss, lbbox_conf_loss, lbbox_prob_loss = loss_layer(conv_lbbox, pred_lbbox, label_lbbox, true_lbboxes, 32, num_classes, iou_loss_thresh)

    ciou_loss = sbbox_ciou_loss + mbbox_ciou_loss + lbbox_ciou_loss
    conf_loss = sbbox_conf_loss + mbbox_conf_loss + lbbox_conf_loss
    prob_loss = sbbox_prob_loss + mbbox_prob_loss + lbbox_prob_loss

    return ciou_loss, conf_loss, prob_loss


class YoloV4Model(base_model.Model):
    """YoloV4 model function."""

    def _get_class(self):
        class_names = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 
                        'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 
                        'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
                        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 
                        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        return class_names

    def _get_anchors(self):
        anchors_stride_base = np.array([
            [[12, 16], [19, 36], [40, 28]],
            [[36, 75], [76, 55], [72, 146]],
            [[142, 110], [192, 243], [459, 401]]
        ])

        anchors_stride_base = anchors_stride_base.astype(np.float32)
        anchors_stride_base[0] /= 8
        anchors_stride_base[1] /= 16
        anchors_stride_base[2] /= 32
        return anchors_stride_base

    def _get_eval_anchors(self):
        anchors = [12.0, 16.0, 19.0, 36.0, 40.0, 28.0, 36.0, 75.0, 76.0, 55.0, 72.0, 146.0, 142.0, 110.0, 192.0, 243.0, 459.0, 401.0]
        return np.array(anchors).reshape(-1, 2)

    def __init__(self, params):
        super().__init__(params)

        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.eval_anchors = self._get_eval_anchors()
        self.iou_loss_thresh = 0.7
        self.score = 0.5
        self.iou = 0.45
        self.max_bbox_per_scale = 150
        self.num_anchors = len(self.anchors)
        self.num_classes = len(self.class_names)
        self._min_level = params.model_params.architecture.min_level
        self._max_level = params.model_params.architecture.max_level
        self.input_image_shape = K.constant([params.preprocessing.width,  params.preprocessing.height])

        # For eval metrics.
        self._params = params

        self._checkpoint_prefix = 'yolov4/'

        # Loss function.
        self._loss_fn = None
        
        self._keras_model = None

        # Predict function.
        self._generate_detections_fn = postprocess_ops._generate_detections

        # Input layer.
        self._input_layer = tf.keras.layers.Input(
            shape=(None, None, params.input_info.sample_size[-1]),
            name='',
            dtype=tf.float32)

    def build_outputs(self, inputs, is_training):
        
        y19_output, y38_output, y76_output = yolo4_body(inputs, self.num_anchors, self.num_classes)

        model_outputs = {
            'y19_output': y19_output,
            'y38_output': y38_output,
            'y76_output': y76_output,
        }

        return model_outputs

    def build_loss_fn(self):
        if self._keras_model is None:
            raise ValueError('build_loss_fn() must be called after build_model().')


        filter_fn = self.make_filter_trainable_variables_fn()
        trainable_variables = filter_fn(self._keras_model.trainable_variables)

        def _total_loss_fn(labels, outputs):
            _args = [outputs['y19_output'], outputs['y38_output'], 
                        outputs['y76_output'], 
                        tf.squeeze(tf.convert_to_tensor(labels["label_sbbox"])),  # label_sbbox
                        tf.squeeze(tf.convert_to_tensor(labels["label_mbbox"])),  # label_mbbox
                        tf.squeeze(tf.convert_to_tensor(labels["label_lbbox"])),  # label_lbbox
                        tf.squeeze(tf.convert_to_tensor(labels["sbboxes"])),      # true_sbboxes
                        tf.squeeze(tf.convert_to_tensor(labels["mbboxes"])),      # true_mbboxes
                        tf.squeeze(tf.convert_to_tensor(labels["lbboxes"]))       # true_lbboxes
                    ]

            ciou_loss, conf_loss, prob_loss = yolo_loss(_args, self.num_classes, self.iou_loss_thresh, self.anchors)

            model_loss = ciou_loss + conf_loss + prob_loss
            l2_regularization_loss = self.weight_decay_loss(trainable_variables)
            total_loss = model_loss + l2_regularization_loss

            return {
                'total_loss': total_loss,
                'ciou_loss': ciou_loss,
                'conf_loss': conf_loss,
                'prob_loss': prob_loss,
                'model_loss': model_loss,
                'l2_regularization_loss': l2_regularization_loss,
            }

        return _total_loss_fn

    def build_model(self, weights=None, is_training=None):
        if self._keras_model is None:
            with keras_utils.maybe_enter_backend_graph():
                outputs = self.model_outputs(self._input_layer, is_training)

                model = tf.keras.models.Model(inputs=self._input_layer,
                                              outputs=outputs, name='yolov4')
                model.run_eagerly = True
                if is_training:
                    for i in range(len(model.layers)):
                        model.layers[i].trainable = True
                assert model is not None, 'Fail to build tf.keras.Model.'
                self._keras_model = model

            if self._checkpoint_path:
                logger.info('Init backbone')
                init_checkpoint_fn = self.make_restore_checkpoint_fn()
                init_checkpoint_fn(self._keras_model)

            if weights:
                logger.info('Loaded pretrained weights from {}'.format(weights))
                self._keras_model.load_weights(weights)

        return self._keras_model

    def post_processing(self, labels, outputs):
        required_output_fields = ['y19_output', 'y38_output', 'y76_output']

        for field in required_output_fields:
            if field not in outputs:
                raise ValueError('"{}" is missing in outputs, requried {} found {}'.format(
                                 field, required_output_fields, outputs.keys()))
        required_label_fields = ['image_info', 'groundtruths']

        for field in required_label_fields:
            if field not in labels:
                raise ValueError('"{}" is missing in outputs, requried {} found {}'.format(
                                 field, required_label_fields, labels.keys()))

        boxes_, scores_ = yolo_eval([outputs['y19_output'], outputs['y38_output'],
                outputs['y76_output']], self.eval_anchors,
                self.num_classes, self.input_image_shape)

        boxes, scores, classes, valid_detections = self._generate_detections_fn(tf.expand_dims(boxes_, axis=2), boxes_)
        
        # Discards the old output tensors to save memory. The `y19_output`,
        # `y38_output` and `y76_output` are pretty big and could potentiall lead to memory issue.
        outputs = {
            'source_id': labels['groundtruths']['source_id'],
            'image_info': labels['image_info'],
            'num_detections': valid_detections,
            'detection_boxes': boxes,
            'detection_classes': classes,
            'detection_scores': scores,
        }

        return labels, outputs

    def eval_metrics(self):
        annotation_file = self._params.get('val_json_file', None)
        evaluator = coco_evaluator.COCOEvaluator(annotation_file=annotation_file,
                                                 include_mask=False)
        return coco_evaluator.MetricWrapper(evaluator)

