# coding=utf-8
# Copyright 2018 Google LLC & Hwalsuk Lee.
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

"""Binary to train and evaluate one GAN configuration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# pylint: disable=unused-import

from absl import app
from absl import flags
from absl import logging

from compare_gan import eval_gan_lib
from compare_gan.metrics import fid_score as fid_score_lib
from compare_gan.metrics import inception_score as inception_score_lib

import dnnlib
import gin
import tensorflow as tf


FLAGS = flags.FLAGS

flags.DEFINE_string("tfhub_url", None, "URL of the pre-trained TFHub model.")
flags.DEFINE_integer(
    "num_eval_averaging_runs", 1,
    "How many times to average FID and IS")

gin.bind_parameter("dataset.name", "imagenet_128")
gin.bind_parameter("eval_z.distribution_fn", tf.random.normal)


def main(unused_argv):
    eval_tasks = [
        inception_score_lib.InceptionScoreTask(),
        fid_score_lib.FIDScoreTask()
    ]
    logging.info("eval_tasks: %s", eval_tasks)

    result_dict = eval_gan_lib.evaluate_tfhub_module(
        module_spec=dnnlib.util.unzip_from_url(FLAGS.tfhub_url),
        eval_tasks=eval_tasks,
        use_tpu=False,
        num_averaging_runs=FLAGS.num_eval_averaging_runs,
        update_bn_accumulators=False,
        use_tags=False,
    )
    logging.info("Evaluation result for checkpoint %s: %s", FLAGS.tfhub_url, result_dict)


if __name__ == "__main__":
    flags.mark_flag_as_required("tfhub_url")
    app.run(main)
