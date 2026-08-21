"""
Synchronized Data Augmentations for Dual-Stream YOLOv11.
Ensures that geometric transformations (Mosaic, Letterbox, RandomFlip, Affine)
are applied identically to both Visible (RGB) and Infrared (IR) images.
"""

import random
import cv2
import numpy as np
import torch


def synchronized_letterbox(img1, img2, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """
    Resize and pad both RGB and IR image identically while meeting stride-multiple constraints.
    """
    shape1 = img1.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape1[0], new_shape[1] / shape1[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape1[1] * r)), int(round(shape1[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape1[1], new_shape[0] / shape1[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape1[::-1] != new_unpad:  # resize
        img1 = cv2.resize(img1, new_unpad, interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img1 = cv2.copyMakeBorder(img1, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    img2 = cv2.copyMakeBorder(img2, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img1, img2, ratio, (dw, dh)


def synchronized_random_flip(img1, img2, bboxes, p_lr=0.5, p_ud=0.0):
    """
    Randomly flip both RGB and IR image horizontally/vertically using the same random decision.
    bboxes: numpy array [[x1, y1, x2, y2], ...] normalized or absolute.
    """
    # Horizontal flip
    if random.random() < p_lr:
        img1 = np.fliplr(img1)
        img2 = np.fliplr(img2)
        if len(bboxes) > 0:
            bboxes[:, [0, 2]] = 1.0 - bboxes[:, [2, 0]]

    # Vertical flip
    if random.random() < p_ud:
        img1 = np.flipud(img1)
        img2 = np.flipud(img2)
        if len(bboxes) > 0:
            bboxes[:, [1, 3]] = 1.0 - bboxes[:, [3, 1]]

    return np.ascontiguousarray(img1), np.ascontiguousarray(img2), bboxes


def synchronized_random_affine(img1, img2, bboxes, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, border=(0, 0)):
    """
    Apply synchronized random affine transformation to both image modalities and coordinates.
    """
    height = img1.shape[0] + border[0] * 2
    width = img1.shape[1] + border[1] * 2

    # Rotation and Scale
    C = np.eye(3)
    C[0, 2] = -img1.shape[1] / 2
    C[1, 2] = -img1.shape[0] / 2

    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height

    # Combined matrix
    M = T @ R @ C

    # Warp both RGB and IR images with identical matrix
    img1 = cv2.warpAffine(img1, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
    img2 = cv2.warpAffine(img2, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Transform bounding boxes
    if len(bboxes) > 0:
        n = len(bboxes)
        xy = np.ones((n * 4, 3))
        xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T
        xy = xy[:, :2].reshape(n, 8)

        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        bboxes = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # Clip boxes
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clip(0, width)
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clip(0, height)

    return img1, img2, bboxes
