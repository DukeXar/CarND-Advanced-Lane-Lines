import collections
import glob
import logging
import os
import pickle

import cv2
import numpy as np

from pipeline import Processor


def load_calibration_images(path):
    filenames = glob.glob(os.path.join(path, '*.jpg'))
    result = []
    for fname in filenames:
        result.append(cv2.imread(fname, cv2.IMREAD_COLOR))
    return result


def calibrate_camera(images, grid_shape=(9, 6)):
    """
    Calibrates the camera using provided images.
    Assumes that all images are of the same shape.
    :param grid_shape: expected grid shape
    :param images: list of color images
    :return: (mtx, dist, rvecs, tvecs, imgpoints, good_images)
    """

    grayed = []
    for img in images:
        grayed.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    objp = np.zeros((grid_shape[0] * grid_shape[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid_shape[0], 0:grid_shape[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    # shape is like (720, 1280), convert it into (1280, 720)
    # image_size must be in (w, h) order
    image_size = grayed[0].shape[::-1]

    good_images = []
    for idx, img in enumerate(grayed):
        ret, corners = cv2.findChessboardCorners(img, grid_shape, None)
        if not ret:
            logging.error('Could not find chessboard corners for image idx={}'.format(idx))
            continue
        objpoints.append(objp)
        imgpoints.append(corners)
        good_images.append(images[idx])

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    assert ret

    return mtx, dist, rvecs, tvecs, imgpoints, good_images


CalibrationConfig = collections.namedtuple('CalibrationConfig', ['mtx', 'dist'])


def load_camera_calibration(calibration_images_dir='./camera_cal',
                            calibration_filename='./camera_calibration.p'):
    """
    Loads cached or recreates camera calibration.
    """
    try:
        with open(calibration_filename, 'rb') as fh:
            return pickle.load(fh)

    except Exception as e:
        logging.error('Could not load {}: {}'.format(calibration_filename, e))

    mtx, dist, _, _, _, _ = calibrate_camera(load_calibration_images(calibration_images_dir))
    cal = CalibrationConfig(mtx=mtx, dist=dist)

    try:
        with open(calibration_filename, 'wb') as fh:
            pickle.dump(cal, fh)
    except Exception as e:
        logging.error('Could not store {}: {}'.format(calibration_filename, e))

    return cal


class CameraCalibration(Processor):
    def __init__(self, config):
        super().__init__()
        self._config = config

    def apply(self, image):
        return cv2.undistort(image, self._config.mtx, self._config.dist, None, self._config.mtx)

    def dump_input_frame(self, image):
        return image

    def dump_output_frame(self, image):
        return self.output


PerspectiveWarpConfig = collections.namedtuple('PerspectiveWarpConfig', ['src', 'dst'])


def draw_warp_shape(image, shape, draw_center=False):
    """
    Draws projection trapezoid.
    :param image: image to draw on
    :param shape: shape of the trapezoid [bottom-left, bottom-right, top-left, top-right]
    :param draw_center: display trapezoid vertical center line
    """
    pts = np.array([
        shape[0], shape[1], shape[3], shape[2],
    ], np.int32)
    pts = pts.reshape((-1, 1, 2))

    cv2.polylines(image, [pts], True, (255, 0, 0))

    if draw_center:
        pts2 = np.array([
            (shape[1] - shape[0]) / 2 + shape[0],
            (shape[3] - shape[2]) / 2 + shape[2]
        ], np.int32)
        pts2 = pts2.reshape((-1, 1, 2))
        cv2.polylines(image, [pts2], True, (255, 0, 0))


def ensure_color(image):
    """
    Ensures that image is not binary.
    :param image: source image with one or multiple channels
    :return: either source image or grayscaled one.
    """
    if len(image.shape) < 3 or image.shape[2] < 3:
        return cv2.cvtColor(image * 255., cv2.COLOR_GRAY2RGB)
    return image


class PerspectiveWarp(Processor):
    """
    Applies perspective warp transformation from source to destination points.
    """
    def __init__(self, src, dst):
        super().__init__()
        self._src = src
        self._dst = dst
        self._m = cv2.getPerspectiveTransform(src, dst)

    def apply(self, image):
        return cv2.warpPerspective(image, self._m, (image.shape[1], image.shape[0]))

    def dump_input_frame(self, image):
        augm = ensure_color(image).copy()
        draw_warp_shape(augm, self._src, draw_center=True)
        return augm

    def dump_output_frame(self, image):
        augm = self.apply(ensure_color(image)).copy()
        draw_warp_shape(augm, self._dst, draw_center=True)
        return augm


class ScaleBinaryToGrayscale(Processor):
    def __init__(self):
        super().__init__()

    def apply(self, image):
        return ensure_color(image)

    def dump_input_frame(self, image):
        return self.apply(image)

    def dump_output_frame(self, image):
        return self.apply(image)
