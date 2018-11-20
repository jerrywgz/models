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

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from absl import flags
import paddle.fluid as fluid
import build.layers as layers
import build.ops as _ops

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_stages", 3, "number of stages")
flags.DEFINE_integer("num_cells", 7, "number of cells per stage")
flags.DEFINE_integer("width", 96, "network width")

flags.DEFINE_integer("ratio", 6, "compression ratio")

flags.DEFINE_float("dropout_rate_path", 0.4, "dropout rate for cell path")
flags.DEFINE_float("dropout_rate_fin", 0.5, "dropout rate for finishing layer")

num_classes = 1000

ops = [
    _ops.conv_1x1,
    _ops.conv_3x3,
    _ops.conv_5x5,
    _ops.dilated_3x3,
    _ops.conv_1x3_3x1,
    _ops.conv_1x5_5x1,
    _ops.maxpool_3x3,
    _ops.maxpool_5x5,
    _ops.avgpool_3x3,
    _ops.avgpool_5x5,
]


def factorized_reduction(inputs, filters, stride):
    assert filters % 2 == 0, (
        'Need even number of filters when using this factorized reduction.')
    if stride == (1, 1):
        # with tf.variable_scope("fred_conv_bn0"):
        x = layers.conv(inputs, filters, stride)
        x = fluid.layers.batch_norm(x)
        return x

    # with tf.variable_scope("fred_path1"):
    path1 = layers.avgpool_valid(inputs, (1, 1), stride)
    path1 = layers.conv(path1, int(filters / 2), (1, 1))

    pad_arr = [0, 0, 0, 0, 0, 1, 0, 1]
    path2 = fluid.layers.pad(inputs, pad_arr)
    path2 = fluid.layers.slice(
        path2,
        axes=[2, 3],
        starts=[1, 1],
        ends=[path2.shape[2], path2.shape[3]])

    # with tf.variable_scope("fred_path2"):
    path2 = layers.avgpool_valid(path2, (1, 1), stride)
    path2 = layers.conv(path2, int(filters / 2), (1, 1))

    print("path2 shape: {}".format(path2.shape))
    final_path = fluid.layers.concat(input=[path1, path2], axis=1)
    final_path = fluid.layers.batch_norm(final_path)
    return final_path


def net(inputs, output, tokens):
    adjvec = tokens[1]
    tokens = tokens[0]
    print("tokens: " + str(tokens))
    print("adjvec: " + str(adjvec))

    num_nodes = len(tokens) // 2

    def slice(vec):
        mat = np.zeros([num_nodes, num_nodes])
        pos = lambda i: i * (i - 1) // 2
        for i in range(1, num_nodes):
            mat[0:i, i] = vec[pos(i):pos(i + 1)]
        return mat

    normal_to, reduce_to = np.split(tokens, 2)
    normal_ad, reduce_ad = map(slice, np.split(adjvec, 2))
    print("AAA1: input.shape=" + str(inputs.shape))
    # with tf.variable_scope("0.initial_conv"):
    x = layers.conv(inputs, FLAGS.width, (3, 3), (2, 2))
    print("AAA2: after initial_conv:" + str(x.shape))
    # with tf.variable_scope("0.initial_bn"):
    x = layers.batch_norm(x)
    stem_idx = [0, 1]
    for c in stem_idx:
        # with tf.variable_scope("%d.stem_cell" % c):
        # NOTE:1 - using cell achevie 75% TOP1 ACC.
        #x = cell(x, reduce_to, reduce_ad, 0., downsample=True) 
        # NOTE: 2 - using factorized_reduction achevie 72.34% TOP1 ACC.
        x = factorized_reduction(x, FLAGS.width * (2**(c + 1)), (2, 2))
        print("AAA2: stem c=" + str(c) + "shape=" + str(x.shape))
    pre_activation_idx = [1]
    reduction_idx = [
        i * FLAGS.num_cells + 1 for i in range(1, FLAGS.num_stages)
    ]
    aux_head_idx = [(FLAGS.num_stages - 1) * FLAGS.num_cells]

    num_cells = FLAGS.num_stages * FLAGS.num_cells
    for c in range(1, num_cells + 1):
        dropout_rate = c / num_cells * FLAGS.dropout_rate_path
        if c in pre_activation_idx:
            # with tf.variable_scope("%d.normal_cell" % c):
            x = cell(x, normal_to, normal_ad, dropout_rate, pre_activation=True)
        elif c in reduction_idx:
            # with tf.variable_scope("%d.reduction_cell" % c):
            x = cell(x, reduce_to, reduce_ad, dropout_rate, downsample=True)
            print("AAA4: reduction cell=" + str(c) + "shape=" + str(x.shape))
        else:
            # with tf.variable_scope("%d.normal_cell" % c):
            x = cell(x, normal_to, normal_ad, dropout_rate)
        if c in aux_head_idx:
            # with tf.variable_scope("aux_head"):
            aux_loss = aux_head(x, output)
            print("AAA4: aux_loss=" + str(c) + "shape=" + str(aux_loss.shape))

    # with tf.variable_scope("%d.global_average_pooling" % (num_cells + 1)):
    #print("main:" + str(x.shape))
    x = layers.bn_relu(x)
    x = layers.global_avgpool(x)
    print("AAA5: after global_pool: " + str(x.shape))
    x = layers.dropout(x, dropout_rate)
    logits = layers.fully_connected(x, num_classes)

    cost = fluid.layers.softmax_with_cross_entropy(logits=logits, label=output)
    avg_cost = fluid.layers.mean(cost) + 0.4 * aux_loss
    accuracy = fluid.layers.accuracy(input=logits, label=output)

    return avg_cost, accuracy


def aux_head(inputs, output):
    print("aux_input: " + str(inputs.shape))

    # x = layers.avgpool(inputs, (5, 5), (3, 3), padding="valid")
    x = layers.avgpool_valid(inputs, (5, 5), (3, 3))
    print("aux:" + str(x.shape))

    x = layers.conv(x, 128, (1, 1))
    print("aux:" + str(x.shape))
    x = layers.bn_relu(x)
    shape = x.shape
    print("aux:" + str(shape))
    # x = layers.conv(x, 768, (4, 4), padding="valid")
    x = layers.conv(x, 768, (shape[2], shape[2]), auto_pad=False)

    # x = tf.squeeze(x, axis=[2, 3])
    x = fluid.layers.squeeze(x, [2, 3])
    print("12:" + str(x.shape))
    x = layers.bn_relu(x)
    logits = layers.fully_connected(x, num_classes)

    cost = fluid.layers.softmax_with_cross_entropy(logits=logits, label=output)
    return fluid.layers.mean(cost)


def cell(inputs,
         tokens,
         adjmat,
         dropout_rate,
         pre_activation=False,
         downsample=False):
    filters = int(inputs.shape[1])
    d = filters // FLAGS.ratio

    if pre_activation:
        inputs = layers.bn_relu(inputs)

    num_nodes, tensors = len(adjmat), []
    for n in range(num_nodes):
        func = ops[tokens[n]]
        idx, = np.nonzero(adjmat[:, n])
        # with tf.variable_scope("%d.%s" % (n, func.__name__)):
        if len(idx) == 0:
            x = inputs if pre_activation else layers.bn_relu(inputs)

            x = layers.conv(x, d, (1, 1))
            x = layers.bn_relu(x)
            x = func(x, downsample)
        else:
            # x = tf.add_n([tensors[i] for i in idx])
            tensor_list = [tensors[i] for i in idx]
            x = tensor_list[0]
            for i in range(1, len(tensor_list)):
                x = fluid.layers.elementwise_add(x, tensor_list[i])

            x = layers.bn_relu(x)
            x = func(x)
        x = layers.dropout(x, dropout_rate)
        tensors.append(x)

    free_ends, = np.where(~adjmat.any(axis=1))
    tensors = [tensors[i] for i in free_ends]
    filters = filters * 2 if downsample else filters
    # with tf.variable_scope("%d.add" % num_nodes):
    #     x = tf.concat(tensors, axis=1)
    x = fluid.layers.concat(tensors, axis=1)
    x = layers.conv(x, filters, (1, 1))

    #print("cell: %s -> %s" % (inputs.shape, x.shape))
    return x
