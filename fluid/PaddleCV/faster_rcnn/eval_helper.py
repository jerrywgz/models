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

import os
import numpy as np
import paddle.fluid as fluid
import math
import box_utils
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from config import cfg
import pycocotools.mask as mask_util
import six
from colormap import colormap


def box_decoder(target_box, prior_box, prior_box_var):
    proposals = np.zeros_like(target_box, dtype=np.float32)
    prior_box_loc = np.zeros_like(prior_box, dtype=np.float32)
    prior_box_loc[:, 0] = prior_box[:, 2] - prior_box[:, 0] + 1.
    prior_box_loc[:, 1] = prior_box[:, 3] - prior_box[:, 1] + 1.
    prior_box_loc[:, 2] = (prior_box[:, 2] + prior_box[:, 0]) / 2
    prior_box_loc[:, 3] = (prior_box[:, 3] + prior_box[:, 1]) / 2
    pred_bbox = np.zeros_like(target_box, dtype=np.float32)
    for i in range(prior_box.shape[0]):
        dw = np.minimum(prior_box_var[2] * target_box[i, 2::4], cfg.bbox_clip)
        dh = np.minimum(prior_box_var[3] * target_box[i, 3::4], cfg.bbox_clip)
        pred_bbox[i, 0::4] = prior_box_var[0] * target_box[
            i, 0::4] * prior_box_loc[i, 0] + prior_box_loc[i, 2]
        pred_bbox[i, 1::4] = prior_box_var[1] * target_box[
            i, 1::4] * prior_box_loc[i, 1] + prior_box_loc[i, 3]
        pred_bbox[i, 2::4] = np.exp(dw) * prior_box_loc[i, 0]
        pred_bbox[i, 3::4] = np.exp(dh) * prior_box_loc[i, 1]
    proposals[:, 0::4] = pred_bbox[:, 0::4] - pred_bbox[:, 2::4] / 2
    proposals[:, 1::4] = pred_bbox[:, 1::4] - pred_bbox[:, 3::4] / 2
    proposals[:, 2::4] = pred_bbox[:, 0::4] + pred_bbox[:, 2::4] / 2 - 1
    proposals[:, 3::4] = pred_bbox[:, 1::4] + pred_bbox[:, 3::4] / 2 - 1

    return proposals


def clip_tiled_boxes(boxes, im_shape):
    """Clip boxes to image boundaries. im_shape is [height, width] and boxes
    has shape (N, 4 * num_tiled_boxes)."""
    assert boxes.shape[1] % 4 == 0, \
        'boxes.shape[1] is {:d}, but must be divisible by 4.'.format(
        boxes.shape[1]
    )
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def get_nmsed_box(rpn_rois, confs, locs, class_nums, im_info):
    lod = rpn_rois.lod()[0]
    rpn_rois_v = np.array(rpn_rois)
    variance_v = np.array(cfg.bbox_reg_weights)
    confs_v = np.array(confs)
    locs_v = np.array(locs)
    rois = box_decoder(locs_v, rpn_rois_v, variance_v)
    im_results = [[] for _ in range(len(lod) - 1)]
    new_lod = [0]
    for i in range(len(lod) - 1):
        start = lod[i]
        end = lod[i + 1]
        if start == end:
            continue
        rois_n = rois[start:end, :]
        rois_n = rois_n / im_info[i][2]
        rois_n = clip_tiled_boxes(rois_n, im_info[i][:2])

        cls_boxes = [[] for _ in range(class_nums)]
        scores_n = confs_v[start:end, :]
        for j in range(1, class_nums):
            inds = np.where(scores_n[:, j] > cfg.TEST.score_thresh)[0]
            scores_j = scores_n[inds, j]
            rois_j = rois_n[inds, j * 4:(j + 1) * 4]
            dets_j = np.hstack((rois_j, scores_j[:, np.newaxis])).astype(
                np.float32, copy=False)
            keep = box_utils.nms(dets_j, cfg.TEST.nms_thresh)
            nms_dets = dets_j[keep, :]
            #add labels
            #cat_id = numId_to_catId_map[j]
            label = np.array([j for _ in range(len(keep))])
            nms_dets = np.hstack((nms_dets, label[:, np.newaxis])).astype(
                np.float32, copy=False)
            cls_boxes[j] = nms_dets
    # Limit to max_per_image detections **over all classes**
        image_scores = np.hstack(
            [cls_boxes[j][:, -2] for j in range(1, class_nums)])
        if len(image_scores) > cfg.TEST.detections_per_im:
            image_thresh = np.sort(image_scores)[-cfg.TEST.detections_per_im]
            for j in range(1, class_nums):
                keep = np.where(cls_boxes[j][:, -2] >= image_thresh)[0]
                cls_boxes[j] = cls_boxes[j][keep, :]

        im_results_n = np.vstack([cls_boxes[j] for j in range(1, class_nums)])
        im_results[i] = im_results_n
        new_lod.append(len(im_results_n) + new_lod[-1])
        boxes = im_results_n[:, :-2]
        scores = im_results_n[:, -2]
        labels = im_results_n[:, -1]
    im_results = np.vstack([im_results[k] for k in range(len(lod) - 1)])
    return new_lod, im_results


def get_dt_res(batch_size, lod, nmsed_out, data, numId_to_catId_map):
    dts_res = []
    nmsed_out_v = np.array(nmsed_out)
    assert (len(lod) == batch_size + 1), \
      "Error Lod Tensor offset dimension. Lod({}) vs. batch_size({})"\
                    .format(len(lod), batch_size)
    k = 0
    for i in range(batch_size):
        dt_num_this_img = lod[i + 1] - lod[i]
        image_id = int(data[i][-1])
        image_width = int(data[i][1][1])
        image_height = int(data[i][1][2])
        for j in range(dt_num_this_img):
            dt = nmsed_out_v[k]
            k = k + 1
            xmin, ymin, xmax, ymax, score, num_id = dt.tolist()
            category_id = numId_to_catId_map[num_id]
            w = xmax - xmin + 1
            h = ymax - ymin + 1
            bbox = [xmin, ymin, w, h]
            dt_res = {
                'image_id': image_id,
                'category_id': category_id,
                'bbox': bbox,
                'score': score
            }
            dts_res.append(dt_res)
    return dts_res


def get_segms_res(batch_size, lod, segms_out, data, numId_to_catId_map):
    segms_res = []
    segms_out_v = np.array(segms_out)
    for i in range(batch_size):
        dt_num_this_img = lod[i + 1] - lod[i]
        image_id = int(data[i][-1])
        for j in range(dt_num_this_img):
            dt = segms_out_v[k]
            score, num_id, segm = dt.tolist()
            cat_id = numId_to_catId_map[num_id]
            if six.PY3:
                if 'counts' in segm:
                    segm['counts'] = rle['counts'].decode("utf8")
            segm_res = {
                'image_id': image_id,
                'category_id': cat_id,
                'segmentation': segm,
                'score': score
            }
            segms_res.append(segm_res)
    return segms_res


def draw_bounding_box_on_image(image_path, nms_out, draw_threshold, label_list,
                               numId_to_catId_map):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size

    for dt in nms_out:
        xmin, ymin, xmax, ymax, score, num_id = dt.tolist()
        category_id = numId_to_catId_map[num_id]
        if score < draw_threshold:
            continue
        bbox = dt[:4]
        xmin, ymin, xmax, ymax = bbox
        draw.line(
            [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
             (xmin, ymin)],
            width=4,
            fill='red')
        if image.mode == 'RGB':
            draw.text((xmin, ymin), label_list[int(category_id)], (255, 255, 0))
    image_name = image_path.split('/')[-1]
    print("image with bbox drawed saved as {}".format(image_name))
    image.save(image_name)


def draw_mask_on_image(image_path, segms_out, draw_threshold):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    mask_color_id = 0
    w_ratio = .4
    for dt in nms_out:
        score, num_id, segm = dt.tolist()
        if score < draw_threshold:
            continue
        mask = mask_util.decode(segm) * 255
        color_list = colormap(rgb=True)
        color_mask = color_list[mask_color_id % len(color_list), 0:3]
        mask_color_id += 1
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio * 255
        image.paste(color_mask, mask=Image.fromarray(mask))
    image_name = image_path.split('/')[-1] + '_mask'
    print("image with bbox drawed saved as {}".format(image_name))
    image.save(image_name)


def segm_results(im_result, masks, im_info):
    class_num = cfg.class_num
    M = cfg.resolution
    scale = (M + 2.0) / M
    lod = masks.lod()[0]
    masks_v = np.array(masks)
    boxes = im_results[:, :-2]
    labels = im_results[:, -1]
    segms_results = [[] for _ in range(len(lod) - 1)]
    for i in range(len(lod) - 1):
        im_results_n = im_results[lod[i]:lod[i + 1]]
        cls_segms = []
        masks_n = masks_v[lod[i]:lod[i + 1]]
        boxes_n = boxes[lod[i]:lod[i + 1]]
        labels_n = labels[lod[i]:lod[i + 1]]
        im_h = im_info[i][0]
        im_w = im_info[i][0]
        boxes_n = box_utils.expand_boxes(boxes_n, scale)
        boxes_n = boxes_n.astype(np.int32)
        padded_mask = np.zeros((M + 2, M + 2), dtype=np.float32)

        for j in range(len(im_results_n)):
            class_id = labels_n[j]
            padded_mask[1:-1, 1:-1] = masks_n[j, class_id, :, :]

            ref_box = boxes_n[j, :]
            w = ref_box[2] - ref_box[0] + 1
            h = ref_box[3] - ref_box[1] + 1
            w = np.maximum(w, 1)
            h = np.maximum(h, 1)

            mask = cv2.resize(padded_mask, (w, h))
            mask = np.array(mask > cfg.MRCNN_THRESH_BINARIZE, dtype=np.uint8)
            im_mask = np.zeros((im_h, im_w), dtype=np.uint8)

            x_0 = max(ref_box[0], 0)
            x_1 = min(ref_box[2] + 1, im_w)
            y_0 = max(ref_box[1], 0)
            y_1 = min(ref_box[3] + 1, im_h)

            im_mask[y_0:y_1, x_0:x_1] = mask[(y_0 - ref_box[1]):(y_1 - ref_box[
                1]), (x_0 - ref_box[0]):(x_1 - ref_box[0])]

            rle = mask_util.encode(
                np.array(
                    im_mask[:, :, np.newaxis], order='F'))[0]
            cls_segms.append(rle)
        segms_results[i] = np.array(cls_segms)[:, np.newaxis]

    segms_results = np.vstack([segms_results[k] for k in range(len(lod) - 1)])
    im_results = np.hstack([im_results, segms_results])
    return im_results[:, -3]
