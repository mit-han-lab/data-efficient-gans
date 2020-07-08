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

import numpy as np
from tqdm import tqdm
from PIL import Image

import dnnlib
import tensorflow as tf
import tensorflow_hub as hub


FLAGS = flags.FLAGS

flags.DEFINE_string("tfhub_url", None, "URL of the pre-trained TFHub model.")
flags.DEFINE_multi_integer("classes", [18, 72, 213, 324], "Class indexes.")
flags.DEFINE_integer("batch_size", 8, "Batch size.")
flags.DEFINE_integer("num_batches", 8, "Number of batches.")


def main(unused_argv):
    tfhub_dir = dnnlib.util.unzip_from_url(FLAGS.tfhub_url)
    generator = hub.Module(tfhub_dir, name="gen_module")

    y_in = tf.placeholder(tf.int32, [FLAGS.batch_size])
    z = tf.random.normal([FLAGS.batch_size, 120])
    gen = generator(inputs=dict(z=z, labels=y_in), as_dict=True)["generated"]

    folder = os.path.join('samples', os.path.basename(tfhub_dir))
    try:
        os.makedirs(folder)
    except:
        pass
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for y in FLAGS.classes:
            outputs = []
            for _ in tqdm(range(FLAGS.num_batches)):
                fakes = sess.run(gen, feed_dict={y_in: [y] * FLAGS.batch_size})
                fakes = (fakes * 255).astype(np.uint8)
                outputs.append(np.concatenate(fakes, axis=1))
            outputs = np.concatenate(outputs, axis=0)
            Image.fromarray(outputs).save(os.path.join(folder, '{}.png'.format(y)))


if __name__ == "__main__":
    flags.mark_flag_as_required("tfhub_url")
    app.run(main)
