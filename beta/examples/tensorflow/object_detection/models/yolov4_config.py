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

from beta.examples.tensorflow.common.sample_config import SampleConfig


YOLOV4_CONFIG = SampleConfig({
    "preprocessing": {
        "match_threshold": 0.5,
        "unmatched_threshold": 0.5,
        "aug_rand_hflip": True,
        "aug_scale_min": 1.0,
        "aug_scale_max": 1.0,
        "skip_crowd_during_training": True,
        "max_num_instances": 100,
        "height" : 320,
        "width" : 320,
    },
    "model_params": {
        "architecture": {
            "min_level": 0,
            "max_level": 0,
            "num_classes": 80
        },
        "anchor": {
            "num_scales": 3,
            "aspect_ratios": [1.0, 2.0, 0.5],
            "anchor_size": 4.0
        },
        "postprocessing": {
            "ignore_thresh" : 0.7,
            "truth_thresh" : 1,
            "scale_x_y" : 1.2,
            "iou_thresh" : 0.213,
            "cls_normalizer" : 1.0,
            "iou_normalizer" : 0.07,
            "beta_nms" : 0.6
        }
    }
})


