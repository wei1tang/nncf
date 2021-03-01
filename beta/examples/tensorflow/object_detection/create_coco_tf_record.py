# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Convert raw COCO dataset to TFRecord for object_detection.
Example usage:
    python create_coco_tf_record.py --logtostderr \
      --image_dir="${TRAIN_IMAGE_DIR}" \
      --image_info_file="${TRAIN_IMAGE_INFO_FILE}" \
      --object_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --caption_annotations_file="${CAPTION_ANNOTATIONS_FILE}" \
      --output_file_prefix="${OUTPUT_DIR/FILE_PREFIX}" \
      --num_shards=100
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import hashlib
import io
import json
import logging
import multiprocessing
import os
from absl import app
from absl import flags
import numpy as np
import PIL.Image

from pycocotools import mask
from research.object_detection.utils import dataset_util
from research.object_detection.utils import label_map_util

import tensorflow.compat.v1 as tf
flags.DEFINE_boolean(
    'include_masks', False, 'Whether to include instance segmentations masks '
    '(PNG encoded) in the result. default: False.')
flags.DEFINE_string('image_dir', '', 'Directory containing images.')
flags.DEFINE_string(
    'image_info_file', '', 'File containing image information. '
    'Tf Examples in the output files correspond to the image '
    'info entries in this file. If this file is not provided '
    'object_annotations_file is used if present. Otherwise, '
    'caption_annotations_file is used to get image info.')
flags.DEFINE_string(
    'object_annotations_file', '', 'File containing object '
    'annotations - boxes and instance masks.')
flags.DEFINE_string('caption_annotations_file', '', 'File containing image '
                    'captions.')
flags.DEFINE_string('output_file_prefix', '/tmp/train', 'Path to output file')
flags.DEFINE_integer('num_shards', 32, 'Number of shards for output file.')

FLAGS = flags.FLAGS

logger = tf.get_logger()
logger.setLevel(logging.INFO)

def convert_labels_to_80_classes(label):
  # 0..90 --> 0..79
  match = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 
  11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 
  21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 
  31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 
  41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 
  51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 
  61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 
  72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 
  81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79}

  out = match[label]
  return out

def get_anchors():
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

def preprocess_true_boxes(bboxes):
  anchors = get_anchors()
  strides = np.array([8, 16, 32])
  train_output_sizes = 608 // strides
  anchor_per_scale = 3
  num_classes = 80
  max_bbox_per_scale = 150

  label = [
      np.zeros(
          (
              train_output_sizes[i],
              train_output_sizes[i],
              anchor_per_scale,
              5 + num_classes,
          )
      )
      for i in range(3)
  ]
  bboxes_xywh = [np.zeros((max_bbox_per_scale, 4)) for _ in range(3)]
  bbox_count = np.zeros((3,))

  for bbox in bboxes:
      bbox_coor = bbox[:4]
      bbox_class_ind = int(bbox[4])
      bbox_class_ind = convert_labels_to_80_classes(bbox_class_ind)

      onehot = np.zeros(num_classes, dtype=np.float)
      onehot[bbox_class_ind] = 1.0
      uniform_distribution = np.full(
          num_classes, 1.0 / num_classes
      )
      deta = 0.01
      smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

      
      bbox_xywh = np.concatenate(
          [
              (bbox_coor[2:] + bbox_coor[:2]) * 0.5,
              bbox_coor[2:] - bbox_coor[:2],
          ],
          axis=-1,
      )
      bbox_xywh_scaled = (
          1.0 * bbox_xywh[np.newaxis, :] / strides[:, np.newaxis]
      )

      iou = []
      exist_positive = False
      for i in range(3):
          anchors_xywh = np.zeros((anchor_per_scale, 4))
          anchors_xywh[:, 0:2] = (
              np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
          )
          anchors_xywh[:, 2:4] = anchors[i]

          iou_scale = bbox_iou_data(
              bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh
          )
          iou.append(iou_scale)
          iou_mask = iou_scale > 0.3

          if np.any(iou_mask):
              xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(
                  np.int32
              )

              label[i][yind, xind, iou_mask, :] = 0
              label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
              label[i][yind, xind, iou_mask, 4:5] = 1.0
              label[i][yind, xind, iou_mask, 5:] = smooth_onehot

              bbox_ind = int(bbox_count[i] % max_bbox_per_scale)
              bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
              bbox_count[i] += 1

              exist_positive = True

      if not exist_positive:
          best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
          best_detect = int(best_anchor_ind / anchor_per_scale)
          best_anchor = int(best_anchor_ind % anchor_per_scale)
          xind, yind = np.floor(
              bbox_xywh_scaled[best_detect, 0:2]
          ).astype(np.int32)

          label[best_detect][yind, xind, best_anchor, :] = 0
          label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
          label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
          label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

          bbox_ind = int(
              bbox_count[best_detect] % max_bbox_per_scale
          )
          bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
          bbox_count[best_detect] += 1
  label_sbbox, label_mbbox, label_lbbox = label
  sbboxes, mbboxes, lbboxes = bboxes_xywh
  return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

def create_tf_example(image,
                      image_dir,
                      bbox_annotations=None,
                      category_index=None,
                      caption_annotations=None,
                      include_masks=False):
  """Converts image and annotations to a tf.Example proto.
  Args:
    image: dict with keys: [u'license', u'file_name', u'coco_url', u'height',
      u'width', u'date_captured', u'flickr_url', u'id']
    image_dir: directory containing the image files.
    bbox_annotations:
      list of dicts with keys: [u'segmentation', u'area', u'iscrowd',
        u'image_id', u'bbox', u'category_id', u'id'] Notice that bounding box
        coordinates in the official COCO dataset are given as [x, y, width,
        height] tuples using absolute coordinates where x, y represent the
        top-left (0-indexed) corner.  This function converts to the format
        expected by the Tensorflow Object Detection API (which is which is
        [ymin, xmin, ymax, xmax] with coordinates normalized relative to image
        size).
    category_index: a dict containing COCO category information keyed by the
      'id' field of each category.  See the label_map_util.create_category_index
      function.
    caption_annotations:
      list of dict with keys: [u'id', u'image_id', u'str'].
    include_masks: Whether to include instance segmentations masks
      (PNG encoded) in the result. default: False.
  Returns:
    example: The converted tf.Example
    num_annotations_skipped: Number of (invalid) annotations that were ignored.
  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  image_height = image['height']
  image_width = image['width']
  filename = image['file_name']
  image_id = image['id']

  full_path = os.path.join(image_dir, filename)
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  key = hashlib.sha256(encoded_jpg).hexdigest()
  feature_dict = {
      'image/height':
          dataset_util.int64_feature(image_height),
      'image/width':
          dataset_util.int64_feature(image_width),
      'image/filename':
          dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id':
          dataset_util.bytes_feature(str(image_id).encode('utf8')),
      'image/key/sha256':
          dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded':
          dataset_util.bytes_feature(encoded_jpg),
      'image/format':
          dataset_util.bytes_feature('jpeg'.encode('utf8')),
  }

  num_annotations_skipped = 0
  if bbox_annotations:
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    is_crowd = []
    category_names = []
    category_ids = []
    area = []
    encoded_mask_png = []
    bboxes = []
    for object_annotations in bbox_annotations:
      (x, y, width, height) = tuple(object_annotations['bbox'])
      if width <= 0 or height <= 0:
        num_annotations_skipped += 1
        continue
      if x + width > image_width or y + height > image_height:
        num_annotations_skipped += 1
        continue
      xmin.append(float(x) / image_width)
      xmax.append(float(x + width) / image_width)
      ymin.append(float(y) / image_height)
      ymax.append(float(y + height) / image_height)
      is_crowd.append(object_annotations['iscrowd'])
      category_id = int(object_annotations['category_id'])
      category_ids.append(category_id)
      category_names.append(category_index[category_id]['name'].encode('utf8'))
      area.append(object_annotations['area'])
      bboxes.append([float(y) / image_height, float(x) / image_width, float(y + height) / image_height, float(x + width) / image_width, category_id])

      if include_masks:
        run_len_encoding = mask.frPyObjects(object_annotations['segmentation'],
                                            image_height, image_width)
        binary_mask = mask.decode(run_len_encoding)
        if not object_annotations['iscrowd']:
          binary_mask = np.amax(binary_mask, axis=2)
        pil_image = PIL.Image.fromarray(binary_mask)
        output_io = io.BytesIO()
        pil_image.save(output_io, format='PNG')
        encoded_mask_png.append(output_io.getvalue())
    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = preprocess_true_boxes(np.array(bboxes))
    # print("label_sbbox", np.array(label_sbbox).shape)
    # print("label_mbbox", np.array(label_mbbox).shape)
    # print("label_lbbox", np.array(label_lbbox).shape)
    # print("sbboxes", np.array(sbboxes).shape)
    # print("mbboxes", np.array(mbboxes).shape)
    # print("lbboxes", np.array(lbboxes).shape)
    
    label_sbbox = list(label_sbbox.flatten())
    label_mbbox = list(label_mbbox.flatten())
    label_lbbox = list(label_lbbox.flatten())
    sbboxes = list(sbboxes.flatten())
    mbboxes = list(mbboxes.flatten())
    lbboxes = list(lbboxes.flatten())
    feature_dict.update({
        'image/object/bbox/xmin':
            dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax':
            dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin':
            dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax':
            dataset_util.float_list_feature(ymax),
        'image/object/class/text':
            dataset_util.bytes_list_feature(category_names),
        'image/object/class/label':
            dataset_util.int64_list_feature(category_ids),
        'image/object/is_crowd':
            dataset_util.int64_list_feature(is_crowd),
        'image/object/area':
            dataset_util.float_list_feature(area),
        'image/object/label_sbbox':
            dataset_util.float_list_feature(label_sbbox),
        'image/object/label_mbbox':
            dataset_util.float_list_feature(label_mbbox),
        'image/object/label_lbbox':
            dataset_util.float_list_feature(label_lbbox),
        'image/object/sbboxes':
            dataset_util.float_list_feature(sbboxes),
        'image/object/mbboxes':
            dataset_util.float_list_feature(mbboxes),
        'image/object/lbboxes':
            dataset_util.float_list_feature(lbboxes),
    })
    if include_masks:
      feature_dict['image/object/mask'] = (
          dataset_util.bytes_list_feature(encoded_mask_png))
  if caption_annotations:
    captions = []
    for caption_annotation in caption_annotations:
      captions.append(caption_annotation['caption'].encode('utf8'))
    feature_dict.update(
        {'image/caption': dataset_util.bytes_list_feature(captions)})

  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return key, example, num_annotations_skipped


def _pool_create_tf_example(args):
  return create_tf_example(*args)


def _load_object_annotations(object_annotations_file):
  """Loads object annotation JSON file."""
  with tf.gfile.GFile(object_annotations_file, 'r') as fid:
    obj_annotations = json.load(fid)

  images = obj_annotations['images']
  category_index = label_map_util.create_category_index(
      obj_annotations['categories'])

  img_to_obj_annotation = collections.defaultdict(list)
  logging.info('Building bounding box index.')
  for annotation in obj_annotations['annotations']:
    image_id = annotation['image_id']
    img_to_obj_annotation[image_id].append(annotation)

  missing_annotation_count = 0
  for image in images:
    image_id = image['id']
    if image_id not in img_to_obj_annotation:
      missing_annotation_count += 1

  logging.info('%d images are missing bboxes.', missing_annotation_count)

  return img_to_obj_annotation, category_index


def _load_caption_annotations(caption_annotations_file):
  """Loads caption annotation JSON file."""
  with tf.gfile.GFile(caption_annotations_file, 'r') as fid:
    caption_annotations = json.load(fid)

  img_to_caption_annotation = collections.defaultdict(list)
  logging.info('Building caption index.')
  for annotation in caption_annotations['annotations']:
    image_id = annotation['image_id']
    img_to_caption_annotation[image_id].append(annotation)

  missing_annotation_count = 0
  images = caption_annotations['images']
  for image in images:
    image_id = image['id']
    if image_id not in img_to_caption_annotation:
      missing_annotation_count += 1

  logging.info('%d images are missing captions.', missing_annotation_count)

  return img_to_caption_annotation


def _load_images_info(images_info_file):
  with tf.gfile.GFile(images_info_file, 'r') as fid:
    info_dict = json.load(fid)
  return info_dict['images']


def _create_tf_record_from_coco_annotations(images_info_file,
                                            image_dir,
                                            output_path,
                                            num_shards,
                                            object_annotations_file=None,
                                            caption_annotations_file=None,
                                            include_masks=False):
  """Loads COCO annotation json files and converts to tf.Record format.
  Args:
    images_info_file: JSON file containing image info. The number of tf.Examples
      in the output tf Record files is exactly equal to the number of image info
      entries in this file. This can be any of train/val/test annotation json
      files Eg. 'image_info_test-dev2017.json',
      'instance_annotations_train2017.json',
      'caption_annotations_train2017.json', etc.
    image_dir: Directory containing the image files.
    output_path: Path to output tf.Record file.
    num_shards: Number of output files to create.
    object_annotations_file: JSON file containing bounding box annotations.
    caption_annotations_file: JSON file containing caption annotations.
    include_masks: Whether to include instance segmentations masks
      (PNG encoded) in the result. default: False.
  """

  logging.info('writing to output path: %s', output_path)
  writers = [
      tf.python_io.TFRecordWriter(
          output_path + '-%05d-of-%05d.tfrecord' % (i, num_shards))
      for i in range(num_shards)
  ]
  images = _load_images_info(images_info_file)

  img_to_obj_annotation = None
  img_to_caption_annotation = None
  category_index = None
  if object_annotations_file:
    img_to_obj_annotation, category_index = (
        _load_object_annotations(object_annotations_file))
  if caption_annotations_file:
    img_to_caption_annotation = (
        _load_caption_annotations(caption_annotations_file))

  def _get_object_annotation(image_id):
    if img_to_obj_annotation:
      return img_to_obj_annotation[image_id]
    else:
      return None

  def _get_caption_annotation(image_id):
    if img_to_caption_annotation:
      return img_to_caption_annotation[image_id]
    else:
      return None

  pool = multiprocessing.Pool()
  total_num_annotations_skipped = 0
  for idx, (_, tf_example, num_annotations_skipped) in enumerate(
      pool.imap(_pool_create_tf_example,
                [(image, image_dir, _get_object_annotation(image['id']),
                  category_index, _get_caption_annotation(image['id']),
                  include_masks) for image in images])):
    if idx % 100 == 0:
      logging.info('On image %d of %d', idx, len(images))

    total_num_annotations_skipped += num_annotations_skipped
    writers[idx % num_shards].write(tf_example.SerializeToString())

  pool.close()
  pool.join()

  for writer in writers:
    writer.close()

  logging.info('Finished writing, skipped %d annotations.',
               total_num_annotations_skipped)


def main(_):
  assert FLAGS.image_dir, '`image_dir` missing.'
  assert (FLAGS.image_info_file or FLAGS.object_annotations_file or
          FLAGS.caption_annotations_file), ('All annotation files are '
                                            'missing.')
  if FLAGS.image_info_file:
    images_info_file = FLAGS.image_info_file
  elif FLAGS.object_annotations_file:
    images_info_file = FLAGS.object_annotations_file
  else:
    images_info_file = FLAGS.caption_annotations_file

  directory = os.path.dirname(FLAGS.output_file_prefix)
  if not tf.gfile.IsDirectory(directory):
    tf.gfile.MakeDirs(directory)

  _create_tf_record_from_coco_annotations(images_info_file, FLAGS.image_dir,
                                          FLAGS.output_file_prefix,
                                          FLAGS.num_shards,
                                          FLAGS.object_annotations_file,
                                          FLAGS.caption_annotations_file,
                                          FLAGS.include_masks)


if __name__ == '__main__':
  logger = tf.get_logger()
  logger.setLevel(logging.INFO)
  app.run(main)

