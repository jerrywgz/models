from PIL import Image, ImageEnhance, ImageDraw
from PIL import ImageFile
import numpy as np
import random
import math


class sampler():
    def __init__(self, max_trial, min_scale, max_scale, min_aspect_ratio,
                 max_aspect_ratio, min_overlap):
        #self.max_sample = max_sample
        self.max_trial = max_trial
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_overlap = min_overlap


class BBox():
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


def IsEmpty(src_bbox):
    return src_bbox.xmin > src_bbox.xmax or src_bbox.ymin > src_bbox.ymax


def bbox_area(src_bbox):
    width = src_bbox.xmax - src_bbox.xmin
    height = src_bbox.ymax - src_bbox.ymin
    return width * height


def bbox_overlap(sample_bbox, object_bbox):
    if IsEmpty(sample_bbox) or IsEmpty(object_bbox):
        return 0
    intersect_xmin = max(sample_bbox.xmin, object_bbox.xmin)
    intersect_ymin = max(sample_bbox.ymin, object_bbox.ymin)
    intersect_xmax = min(sample_bbox.xmax, object_bbox.xmax)
    intersect_ymax = min(sample_bbox.ymax, object_bbox.ymax)
    intersect_size = (intersect_xmax - intersect_xmin) * (
        intersect_ymax - intersect_ymin)
    return intersect_size


def satisfy_sample_constraint(sampler, sample_bbox, bboxes):
    kMinArea = 1.0
    if bbox_area(sample_bbox) < kMinArea:
        return False
    for i in range(len(bboxes)):
        object_bbox = BBox(bboxes[i][0], bboxes[i][1], bboxes[i][2],
                           bboxes[i][3])
        object_area = bbox_area(object_bbox)
        if object_area < kMinArea:
            continue
        overlap = bbox_overlap(sample_bbox, object_bbox)
        if overlap / object_area >= sampler.min_overlap:
            return True
    return False


def generate_sample(origin_w, origin_h, sampler):
    scale = np.random.uniform(sampler.min_scale, sampler.max_scale)
    aspect_ratio = np.random.uniform(sampler.min_aspect_ratio,
                                     sampler.max_aspect_ratio)
    min_area = sampler.min_scale * origin_w * origin_h
    max_area = sampler.max_scale * origin_w * origin_h
    height = int((min_area * aspect_ratio)**0.5)
    max_height = int((max_area / aspect_ratio)**0.5)
    if max_height * aspect_ratio > origin_w:
        kEps = 1e-7
        max_height = int((origin_w + 0.5 - kEps) / aspect_ratio)
    if max_height > origin_h:
        max_height = origin_h
    if height >= max_height:
        height = max_height
    if height < max_height:
        height = np.random.randint(height, max_height + 1)
    width = int(height * aspect_ratio)
    area = height * width
    if area < min_area:
        height += 1
        width = int(height * aspect_ratio)
        area = height * width
    if area > max_area:
        height -= 1
        width = int(height * aspect_ratio)
        area = height * width
    xmin, ymin = 0, 0
    if height < origin_h:
        ymin = np.random.randint(origin_h - height)
    if width < origin_w:
        xmin = np.random.randint(origin_w - width)

    xmax = xmin + width
    ymax = ymin + height
    sampled_bbox = BBox(xmin, ymin, xmax, ymax)
    return sampled_bbox


def generate_samples_imagenet(origin_w, origin_h, sampler, bboxes):
    sample_result = BBox(0., 0., origin_w, origin_h)
    for i in range(sampler.max_trial):
        sample_bbox = generate_sample(origin_w, origin_h, sampler)
        if satisfy_sample_constraint(sampler, sample_bbox, bboxes):
            sample_result = sample_bbox
            break
    bbox_begin = [sample_bbox.ymin, sample_bbox.xmin]
    bbox_size = [sample_bbox.ymax-sample_bbox.ymin, \
                 sample_bbox.xmax-sample_bbox.xmin]
    bndbox = (int(sample_bbox.xmin), int(sample_bbox.ymin), \
              int(sample_bbox.xmax), int(sample_bbox.ymax))
    return (bbox_begin, bbox_size, bndbox)


def resize_image(image, resized_w, resized_h):
    return image.resize((resized_w, resized_h), Image.BILINEAR)


def random_hsv_in_yiq(image,
                      lower_saturation,
                      upper_saturation,
                      max_delta_hue=0,
                      scale_value=1):
    delta_hue = np.random.uniform(-max_delta_hue, max_delta_hue)
    scale_saturation = np.random.uniform(lower_saturation, upper_saturation)
    img_hsv = np.array(image.convert('HSV'))
    img_hsv[:, :, 0] = img_hsv[:, :, 0] + delta_hue
    image = Image.fromarray(img_hsv, mode='HSV').convert('RGB')
    image = ImageEnhance.Color(image).enhance(scale_saturation)
    return image


def random_contrast(image, lower, upper=None):
    if upper is None:
        delta = np.random.uniform(-lower, lower)
    else:
        delta = np.random.uniform(lower, upper)
    return ImageEnhance.Contrast(image).enhance(delta)


def random_brightness(image, lower, upper=None):
    if upper is None:
        delta = np.random.uniform(-lower, lower)
    else:
        delta = np.random.uniform(lower, upper)
    return ImageEnhance.Brightness(image).enhance(delta)


def clip_by_value(image, lower, upper):
    img = np.array(image)
    np.clip(img, lower, upper)
    image = Image.fromarray(img)
    return image


def distort_color(image, batch_position=0):
    def distort_fn_0(image=image):
        image = random_brightness(image, 32. / 255.)
        image = random_hsv_in_yiq(
            image,
            lower_saturation=0.5,
            upper_saturation=1.5,
            max_delta_hue=0.2 * math.pi)
        image = random_contrast(image, lower=0.5, upper=1.5)
        return image

    def distort_fn_1(image=image):
        image = random_brightness(image, 32. / 255.)
        image = random_contrast(image, lower=0.5, upper=1.5)
        image = random_hsv_in_yiq(
            image,
            lower_saturation=0.5,
            upper_saturation=1.5,
            max_delta_hue=0.2 * math.pi)
        return image

    if batch_position % 2 == 0:
        image = distort_fn_0(image)
    else:
        image = distort_fn_1(image)
    image = clip_by_value(image, 0., 1.)
    return image
