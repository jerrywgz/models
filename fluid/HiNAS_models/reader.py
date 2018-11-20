# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved
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
"""
CIFAR-10 dataset.
This module will download dataset from
https://www.cs.toronto.edu/~kriz/cifar.html and parse train/test set into
paddle reader creators.
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes,
with 6000 images per class. There are 50000 training images and 10000 test images.
"""

from PIL import Image, ImageEnhance
from PIL import ImageOps
import numpy as np
import image_util as utils
import cPickle
import itertools
import paddle.dataset.common
import tarfile
from absl import flags
import os
import functools
from data_util import GeneratorEnqueuer
import time

FLAGS = flags.FLAGS

flags.DEFINE_boolean("random_flip_left_right", True,
                     "random flip left and right")
flags.DEFINE_boolean("random_flip_up_down", False, "random flip up and down")
#flags.DEFINE_boolean("cutout", True, "cutout")
flags.DEFINE_boolean("standardize_image", True, "standardize input images")
flags.DEFINE_boolean("pad_and_cut_image", True, "pad and cut input images")

__all__ = ['train', 'test']

#URL_PREFIX = 'https://www.cs.toronto.edu/~kriz/'
#CIFAR10_URL = URL_PREFIX + 'cifar-10-python.tar.gz'
#CIFAR10_MD5 = 'c58f30108f718f92721af3b95e74349a'
DATA_DIR = "dataset/ImageNet/"

paddle.dataset.common.DATA_HOME = "dataset/"

image_size = 224
image_depth = 3
THREAD = 8
BUF_SIZE = 102400


def normalized_image(image):
    # Rescale from [0, 255] to [0, 2]
    # Rescale to [-1, 1]
    image = np.transpose(image, (2, 0, 1))
    image = image * 1. / 127.5 - 1.
    return image


def preprocess(samples, image_size, mode, color_jitter, rotate):

    sampler = utils.sampler(100, 0.05, 1.0, 0.75, 1.33, 0.1)
    sample_out = []
    if mode == 'train':
        for batch_position in range(len(samples)):
            sample = samples[batch_position]
            img_path = sample[0]
            label = sample[1]
            bboxes = sample[2:]
            img = Image.open(img_path)
            if [] in bboxes:
                bboxes = [[0., 0., img.size[1], img.size[0]]]
            else:
                bboxes = bboxes[0]
            sample_distorted_bounding_box = utils.generate_samples_imagenet(
                img.size[1], img.size[0], sampler, bboxes)
            bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box
            img = img.crop(distort_bbox)
            if FLAGS.random_flip_left_right and np.random.randint(2):
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if FLAGS.random_flip_up_down and np.random.randint(2):
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img = utils.resize_image(img, image_size, image_size)

            img = np.array(img).astype(np.float32)
            if color_jitter:
                img /= 255.
                img = Image.fromarray(np.uint8(img))
                distorted_image = utils.distort_color(img, batch_position)
                img = np.array(img).astype(np.float32)
                img *= 255.
            img = normalized_image(img)
            sample_out.append((img, label))
    else:
        """
        Validation images do not have bounding boxes, so to crop the image, we first
        resize the image such that the aspect ratio is maintained and the resized
        height and width are both at least 1.15 times `height` and `width`
        respectively.
        """
        scale_factor = 1.15
        for sample in samples:
            img = Image.open(sample[0])
            img_height = float(img.size[0])
            img_width = float(img.size[1])
            max_ratio = max(image_size / img_height, image_size / img_width)
            resize_height = int(img_height * max_ratio * scale_factor)
            resize_width = int(img_height * max_ratio * scale_factor)
            img = utils.resize_image(img, resize_height, resize_width)
            total_crop_height = resize_height - image_size
            crop_top = total_crop_height // 2
            total_crop_width = resize_width - image_size
            crop_left = total_crop_width // 2
            crop_right = crop_left + image_size
            crop_bottom = crop_top + image_size
            img = img.crop((crop_left, crop_top, crop_right, crop_bottom))
            img = np.array(img).astype(np.float32)
            img = normalized_image(img)
            if mode == 'val':
                sample_out.append((img, sample[1]))
            else:
                sample_out.append((img))
    return sample_out


def _reader_creator(file_list,
                    mode,
                    batch_size,
                    devices_num,
                    shuffle=False,
                    color_jitter=False,
                    rotate=False,
                    data_dir=DATA_DIR):
    def reader():
        with open(file_list) as flist:
            full_lines = [line.strip() for line in flist]
            print('data loaded')
            if shuffle:
                np.random.shuffle(full_lines)
            """
            if mode == 'train' and os.getenv('PADDLE_TRAINING_ROLE'):
                # distributed mode if the env var `PADDLE_TRAINING_ROLE` exits
                trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
                trainer_count = int(os.getenv("PADDLE_TRAINERS", "1"))
                per_node_lines = len(full_lines) // trainer_count
                lines = full_lines[trainer_id * per_node_lines:(trainer_id + 1)
                                   * per_node_lines]
                print(
                    "read images from %d, length: %d, lines length: %d, total: %d"
                    % (trainer_id * per_node_lines, per_node_lines, len(lines),
                       len(full_lines)))
            else:
                lines = full_lines
            """
            lines = full_lines
            if mode == 'train' or mode == 'val':
                batch_out = []
                for line in lines:
                    bbox = []
                    data = line.split()
                    img_path = data[0]
                    label = data[1]
                    if len(data) > 2:
                        bbox_num = (len(data) - 2) // 4
                        for i in range(bbox_num):
                            bbox_val = []
                            for j in range(4):
                                bbox_val.append(float(data[2 + j + 4 * i]))
                            bbox.append(bbox_val)
                    img_path = img_path.replace("JPEG", "jpeg")
                    img_path = os.path.join(data_dir + str(mode), img_path)
                    batch_out.append([img_path, int(label), bbox])
                    if len(batch_out) == batch_size * devices_num:
                        batch_out = preprocess(batch_out, image_size, mode,
                                               color_jitter, rotate)
                        for i in range(devices_num):
                            sub_batch_out = []
                            for j in range(batch_size):
                                sub_batch_out.append(batch_out[i * batch_size +
                                                               j])
                            yield sub_batch_out
                            sub_batch_out = []
                        batch_out = []
            elif mode == 'test':
                batch_out = []
                for line in lines:
                    img_path = os.path.join(data_dir, line)
                    batch_out.append((img_path))
                    if len(batch_out) == batch_size:
                        batch_out = preprocess(batch_out, image_size, mode,
                                               color_jitter, rotate)
                        yield batch_out
                        batch_out = []
                if len(batch_out) != 0:
                    batch_out = preprocess(batch_out, image_size, mode,
                                           color_jitter, rotate)
                    yield batch_out

    return reader


def train(batch_size, devices_num, data_dir=DATA_DIR):
    file_list = os.path.join(data_dir, 'train_with_bbox.txt')
    generator = _reader_creator(
        file_list,
        'train',
        batch_size,
        devices_num,
        shuffle=True,
        color_jitter=True,
        rotate=False,
        data_dir=data_dir)

    def infinite_reader():
        while True:
            for data in generator():
                yield data

    def reader():
        try:
            enqueuer = GeneratorEnqueuer(
                infinite_reader(), use_multiprocessing=True)
            enqueuer.start(max_queue_size=24, workers=8)
            generator_output = None
            while True:
                while enqueuer.is_running():
                    if not enqueuer.queue.empty():
                        generator_output = enqueuer.queue.get()
                        break
                    else:
                        time.sleep(0.02)
                yield generator_output
                generator_output = None
        finally:
            if enqueuer is not None:
                enqueuer.stop()

    return reader


def val(batch_size, devices_num, data_dir=DATA_DIR):
    file_list = os.path.join(data_dir, 'val.txt')
    return _reader_creator(
        file_list,
        'val',
        batch_size,
        devices_num,
        shuffle=False,
        data_dir=data_dir)


def test(batch_size, devices_num, data_dir=DATA_DIR):
    file_list = os.path.join(data_dir, 'val.txt')
    return _reader_creator(
        file_list,
        'test',
        batch_size,
        devices_num,
        shuffle=False,
        data_dir=data_dir)
