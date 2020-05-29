#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import six
import time
import subprocess
import distutils.util
import numpy as np
import sys
import paddle.fluid as fluid
from paddle.fluid import core
import multiprocessing as mp
import glob
import json


def print_arguments(args):
    """Print argparse's arguments.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        parser.add_argument("name", default="Jonh", type=str, help="User name.")
        args = parser.parse_args()
        print_arguments(args)

    :param args: Input argparse.Namespace for printing.
    :type args: argparse.Namespace
    """
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(six.iteritems(vars(args))):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def add_arguments(argname, type, default, help, argparser, **kwargs):
    """Add argparse's argument.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        add_argument("name", str, "Jonh", "User name.", parser)
        args = parser.parse_args()
    """
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs)


def fmt_time():
    """ get formatted time for now
    """
    now_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    return now_str


def recall_topk_ori(fea, lab, k):
    fea = np.array(fea)
    fea = fea.reshape(fea.shape[0], -1)
    n = np.sqrt(np.sum(fea**2, 1)).reshape(-1, 1)
    fea = fea / n
    a = np.sum(fea**2, 1).reshape(-1, 1)
    b = a.T
    ab = np.dot(fea, fea.T)
    d = a + b - 2 * ab
    d = d + np.eye(len(fea)) * 1e8
    sorted_index = np.argsort(d, 1)
    res = 0
    for i in range(len(fea)):
        for j in range(k):
            pred = lab[sorted_index[i][j]]
            if lab[i] == pred:
                res += 1.0
                break
    res = res / len(fea)
    return res


def func(param):
    sharedlist, s, e = param
    fea, a, b = sharedlist
    ab = np.dot(fea[s:e], fea.T)
    d = a[s:e] + b - 2 * ab
    for i in range(e - s):
        d[i][s + i] += 1e8
    sorted_index = np.argsort(d, 1)[:, :10]
    return sorted_index


def recall_topk_parallel(fea, lab, k):
    fea = np.array(fea)
    fea = fea.reshape(fea.shape[0], -1)
    n = np.sqrt(np.sum(fea**2, 1)).reshape(-1, 1)
    fea = fea / n
    a = np.sum(fea**2, 1).reshape(-1, 1)
    b = a.T
    sharedlist = mp.Manager().list()
    sharedlist.append(fea)
    sharedlist.append(a)
    sharedlist.append(b)

    N = 100
    L = fea.shape[0] / N
    params = []
    for i in range(N):
        if i == N - 1:
            s, e = int(i * L), int(fea.shape[0])
        else:
            s, e = int(i * L), int((i + 1) * L)
        params.append([sharedlist, s, e])

    pool = mp.Pool(processes=4)
    sorted_index_list = pool.map(func, params)
    pool.close()
    pool.join()
    sorted_index = np.vstack(sorted_index_list)

    res = 0
    for i in range(len(fea)):
        for j in range(k):
            pred = lab[sorted_index[i][j]]
            if lab[i] == pred:
                res += 1.0
                break
    res = res / len(fea)
    return res


def recall_topk(fea, lab, k=1):
    if fea.shape[0] < 20:
        return recall_topk_ori(fea, lab, k)
    else:
        return recall_topk_parallel(fea, lab, k)


def get_gpu_num():
    visibledevice = os.getenv('CUDA_VISIBLE_DEVICES')
    if visibledevice:
        devicenum = len(visibledevice.split(','))
    else:
        devicenum = subprocess.check_output(
            [str.encode('nvidia-smi'), str.encode('-L')]).decode('utf-8').count(
                '\n')
    return devicenum

def check_cuda(use_cuda, err = \
    "\nYou can not set use_cuda = True in the model because you are using paddlepaddle-cpu.\n \
    Please: 1. Install paddlepaddle-gpu to run your models on GPU or 2. Set use_cuda = False to run models on CPU.\n"
                                                                                                                     ):
    try:
        if use_cuda == True and fluid.is_compiled_with_cuda() == False:
            print(err)
            sys.exit(1)
    except Exception as e:
        pass


def cosine_distance(fea, seq_id, thresh, k):
    fea = fea.reshape(fea.shape[0], -1)
    a = np.linalg.norm(fea, axis=1).reshape(-1, 1)

    d = 1 - np.dot(fea, fea.T) / (a * a.T)
    d = d + np.eye(len(fea)) * 1e8
    sorted_index = np.argsort(d, 1)
    matched_index = []
    k = min(len(fea), k)
    for i in range(len(fea)):
        matched_index.append([])
        matched = 0
        for j in range(k):
            # avoid matched index with same seq_id
            if seq_id[i] == seq_id[sorted_index[i, j]]: continue
            if d[i, sorted_index[i, j]] < thresh:
                matched_index[i].append(sorted_index[i, j])
                matched = 1
        if not matched:
            matched_index[i].append(-1)
    return matched_index


def post_process(results, groups, labels, seq_id, thresh, k):
    group_num = np.max(groups) + 1
    res_group = [[] for i in range(group_num)]
    res_final = [[] for i in range(group_num)]
    im_id = 0
    for res, l, s, g in zip(results, labels, seq_id, groups):
        res_group[g].append([res, l, s, im_id])
        im_id += 1

    for group_id, res_pg in enumerate(res_group):
        #res_label = {}
        res_list = []
        seq_list = []
        if len(res_pg) == 0: continue
        for res in res_pg:
            #if res[1] not in res_label:
            #    res_label[res[1]] = [[res[0]], [res[2]]]
            #else:
            #    res_label[res[1]][0].append(res[0])
            #    res_label[res[1]][1].append(res[2])
            res_list.append(res[0])
            seq_list.append(res[2])
        matched_index = cosine_distance(np.array(res_list), seq_list, thresh, k)
        for i, m_idxs in enumerate(matched_index):
            for m_i in m_idxs:
                if m_i == -1:
                    res_final[group_id].append({i})
                    continue
                if {i, m_i} not in res_final[group_id]:
                    res_final[group_id].append({i, m_i})

    return res_final


def save_result(res_final, path, output_path):
    data_list = os.path.join(path, 'val')
    anno_path = os.path.join(data_list, 'json')
    anno_files = glob.glob(os.path.join(anno_path, '*.json'))
    for i, anno_file in enumerate(anno_files):
        anno = json.load(open(anno_file))
        signs = anno['signs']
        res = res_final[i]
        match_list = []
        for j, match_pair in enumerate(res):
            match_pair = list(match_pair)
            sign_id = signs[match_pair[0]]['sign_id']
            match_list.append({'sign_id': sign_id})
            if len(match_pair) > 1:
                match_sign_id = signs[match_pair[1]]['sign_id']
            else:
                match_sign_id = " "
            match_list[j].update({'match_sign_id': match_sign_id})
        anno.update({'match': match_list})
        file_name = os.path.split(anno_file)[1]
        result_file = os.path.join(output_path, file_name)
        with open(result_file, 'w') as fp:
            json.dump(anno, fp)
        fp.close()
