import collections
import glob
import logging
import os
import pickle

import cv2
import numpy as np


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


class Processor(object):
    def __init__(self):
        self._output = None

    def process(self, image):
        self._output = self.apply(image)

    def apply(self, image):
        raise NotImplementedError()

    def dump_input_frame(self, image):
        raise NotImplementedError()

    def dump_output_frame(self, image):
        raise NotImplementedError()

    @property
    def output(self):
        return self._output


class CameraCalibration(Processor):
    def __init__(self, config):
        super().__init__()
        self._config = config

    def apply(self, image):
        return cv2.undistort(image, self._config.mtx, self._config.dist, None, self._config.mtx)

    def dump_input_frame(self, image):
        return image

    def dump_output_frame(self, image):
        return self.apply(image)


PerspectiveWarpConfig = collections.namedtuple('PerspectiveWarpConfig', ['src', 'dst'])


def draw_warp_shape(image, shape, draw_center=False):
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
    if len(image.shape) < 3 or image.shape[2] < 3:
        return cv2.cvtColor(image * 255., cv2.COLOR_GRAY2RGB)
    return image


class PerspectiveWarp(Processor):
    def __init__(self, src, dst):
        super().__init__()
        self._src = src
        self._dst = dst
        self._m = cv2.getPerspectiveTransform(src, dst)

    def apply(self, image):
        return cv2.warpPerspective(image, self._m, (image.shape[1], image.shape[0]))

    def dump_input_frame(self, image):
        augm = ensure_color(image)
        draw_warp_shape(augm, self._src, draw_center=True)
        return augm

    def dump_output_frame(self, image):
        augm = self.apply(ensure_color(image))
        draw_warp_shape(augm, self._dst, draw_center=True)
        return augm


class BinaryThreshold(Processor):
    def __init__(self):
        super().__init__()
        self._s_thresh = (170, 255)
        self._sobel_x_thresh = (20, 100)

    def _color_threshold(self, hls):
        s_channel = hls[:, :, 2]
        binary = np.zeros_like(s_channel)
        binary[(s_channel > self._s_thresh[0]) & (s_channel <= self._s_thresh[1])] = 1
        return binary

    def _sobel_threshold(self, hls):
        l_channel = hls[:, :, 1]
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= self._sobel_x_thresh[0]) & (scaled_sobel <= self._sobel_x_thresh[1])] = 1
        return sxbinary

    def apply(self, image):
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float32)
        binary_color = self._color_threshold(hls)
        binary_sobel = self._sobel_threshold(hls)
        result = np.zeros_like(binary_color)
        result[(binary_color == 1) | (binary_sobel == 1)] = 1
        return result

    def dump_input_frame(self, image):
        return image.copy()

    def dump_output_frame(self, image):
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float32)
        binary_color = self._color_threshold(hls)
        binary_sobel = self._sobel_threshold(hls)

        result = np.dstack((np.zeros_like(binary_sobel), binary_sobel, binary_color))
        return result * 255.


class ScaleBinaryToGrayscale(Processor):
    def __init__(self):
        super().__init__()

    def apply(self, image):
        return ensure_color(image)

    def dump_input_frame(self, image):
        return self.apply(image)

    def dump_output_frame(self, image):
        return self.apply(image)


def find_initial_centroids_lane(image, window_width, left, right):
    window = np.ones(window_width)  # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template
    # Sum quarter bottom of image to get slice, could use a different ratio
    sum_height = int(3 * image.shape[0] / 4)
    a_sum = np.sum(image[sum_height:, left:right], axis=0)
    center_x = np.argmax(np.convolve(window, a_sum)) - window_width / 2 + left

    return center_x


def find_window_centroids_lane(image, window_width, window_height, margin, center_x, left, right):
    window = np.ones(window_width)  # Create our window template that we will use for convolutions

    window_centroids = []

    image_height = image.shape[0]

    # Go through each layer looking for max pixel locations
    for level in range(1, int(image_height / window_height)):
        top = int(image_height - (level + 1) * window_height)
        bottom = int(image_height - level * window_height)

        # convolve the window into the vertical slice of the image
        image_layer = np.sum(image[top:bottom, :], axis=0)

        conv_signal = np.convolve(window, image_layer)

        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window,
        # not center of window
        offset = window_width / 2
        min_index = int(max(center_x + offset - margin, 0))
        max_index = int(min(center_x + offset + margin, image.shape[1]))
        center_x = np.argmax(conv_signal[min_index:max_index]) + min_index - offset

        # Add what we found for that layer
        window_centroids.append((center_x, top + window_height / 2))

    return window_centroids


def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
           max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    return output


def draw_centroids(centroids, window_width, window_height, image):
    l_points = np.zeros_like(image)
    r_points = np.zeros_like(image)

    if centroids:
        for level in range(0, len(centroids)):
            # TODO: use y instead of level
            l_mask = window_mask(window_width, window_height, image, centroids[level][0][0], level)
            r_mask = window_mask(window_width, window_height, image, centroids[level][1][0], level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 1) | ((l_mask == 1))] = 1
            r_points[(r_points == 1) | ((r_mask == 1))] = 1

        # Draw the results
        # add both left and right window pixels together
        template = np.array(r_points + l_points, np.uint8)
        zero_channel = np.zeros_like(template)
        # make window pixels green
        template = np.array(cv2.merge((zero_channel, template * 255, zero_channel)), np.uint8)

        # making the original road pixels 3 color channels
        warpage = np.array(cv2.merge((image, image, image)), np.uint8)

        # overlay the orignal road image with window results
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)

    else:
        output = np.array(cv2.merge((image, image, image)), np.uint8)

    return output


class LaneSearchCentroids(Processor):
    def __init__(self):
        super().__init__()
        self._window_width = 50
        self._window_height = 80  # Break image into 9 vertical layers since image height is 720
        self._search_margin = 100

    def apply(self, image):
        middle = int(image.shape[1] / 2)
        l_center = find_initial_centroids_lane(image, self._window_width, 0, middle)
        r_center = find_initial_centroids_lane(image, self._window_width, middle, image.shape[1])
        l_centroids = find_window_centroids_lane(image, self._window_width, self._window_height,
                                                 self._search_margin, l_center, 0, middle)
        r_centroids = find_window_centroids_lane(image, self._window_width, self._window_height,
                                                 self._search_margin, r_center, middle, image.shape[1])
        return list(zip([(l_center, image.shape[0])] + l_centroids, [(r_center, image.shape[0])] + r_centroids))

    def dump_input_frame(self, image):
        return image

    def dump_output_frame(self, image):
        centroids = self.apply(image)
        return draw_centroids(centroids, self._window_width, self._window_height, image * 255)


class DisplayLaneSearchCentroids(Processor):
    def __init__(self, image_source_warped):
        super().__init__()
        self._image_source = image_source_warped
        self._window_width = 50
        self._window_height = 80

    def apply(self, centroids):
        image = self._image_source.output
        return draw_centroids(centroids, self._window_width, self._window_height, image * 255)

    def dump_input_frame(self, centroids):
        image = self._image_source.output
        return image

    def dump_output_frame(self, centroids):
        return self.apply(centroids)


class SingleLaneSearch(object):
    def __init__(self, window_width, window_height, search_margin, left, right):
        self._window_width = window_width
        self._window_height = window_height
        self._search_margin = search_margin
        self._left = left
        self._right = right
        self._current_fit = None

    def apply(self, image):
        center_x = find_initial_centroids_lane(image, self._window_width, self._left, self._right)
        center_y = image.shape[0] - self._window_height / 2
        centroids = [(center_x, center_y)] + find_window_centroids_lane(image, self._window_width, self._window_height,
                                                                        self._search_margin, center_x,
                                                                        self._left, self._right)
        fit = np.polyfit([item[1] for item in centroids], [item[0] for item in centroids], 2)
        self._current_fit = fit

    @property
    def current_fit(self):
        return self._current_fit


def get_search_points(fit, ploty, search_margin):
    fitx = fit[0] * ploty ** 2 + fit[1] * ploty + fit[2]
    line_window1 = np.array([np.transpose(np.vstack([fitx - search_margin, ploty]))])
    line_window2 = np.array([np.flipud(np.transpose(np.vstack([fitx + search_margin, ploty])))])
    pts = np.hstack((line_window1, line_window2))
    return pts


def draw_fitted_lanes_warped(image, l_fit, r_fit, search_margin, left_color=(0, 255, 0), right_color=(0, 255, 0)):
    out_img = np.dstack((image, image, image)) * 255
    window_img = np.zeros_like(out_img)

    ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])

    left_line_pts = get_search_points(l_fit, ploty, search_margin)
    cv2.fillPoly(window_img, np.int_([left_line_pts]), left_color)

    right_line_pts = get_search_points(r_fit, ploty, search_margin)
    cv2.fillPoly(window_img, np.int_([right_line_pts]), right_color)

    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    return result


class LaneSearchFitted(Processor):
    def __init__(self, search_margin, window_width, window_height, image_width, image_height):
        super().__init__()
        self._window_width = window_width
        self._window_height = window_height
        self._search_margin = search_margin
        self._image_height = image_height
        self._image_width = image_width

        middle = int(self._image_width / 2)
        self._l_lane = SingleLaneSearch(self._window_width, self._window_height, self._search_margin,
                                        0, middle)
        self._r_lane = SingleLaneSearch(self._window_width, self._window_height, self._search_margin,
                                        middle + 1, self._image_width)

    def apply(self, image):
        assert image.shape[0:2] == (self._image_height, self._image_width), \
            "Image dimensions must match: {} != {}".format(image.shape[0:1], (self._image_height, self._image_width))

        self._l_lane.apply(image)
        self._r_lane.apply(image)

        return self._l_lane.current_fit, self._r_lane.current_fit

    def dump_input_frame(self, image):
        return image

    def dump_output_frame(self, image):
        left_fit, right_fit = self.apply(image)
        return draw_fitted_lanes_warped(image, left_fit, right_fit, self._search_margin)

    @property
    def search_margin(self):
        return self._search_margin


class DisplayLaneSearchFitted(Processor):
    def __init__(self, image_source_warped, search_margin):
        super().__init__()
        self._image_source = image_source_warped
        self._search_margin = search_margin

    def apply(self, fits):
        l_fit, r_fit = fits
        image = self._image_source.output
        return draw_fitted_lanes_warped(image, l_fit, r_fit, self._search_margin)

    def dump_input_frame(self, centroids):
        image = self._image_source.output
        return image

    def dump_output_frame(self, fits):
        return self.apply(fits)


class DisplayLaneSearchFittedUnwarped(Processor):
    def __init__(self, image_source, src, dst):
        super().__init__()
        self._image_source = image_source
        self._minv = cv2.getPerspectiveTransform(dst, src)

    def apply(self, fits):
        l_fit, r_fit = fits
        image = self._image_source.output

        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        l_fitx = l_fit[0] * ploty ** 2 + l_fit[1] * ploty + l_fit[2]
        r_fitx = r_fit[0] * ploty ** 2 + r_fit[1] * ploty + r_fit[2]

        pts_left = np.array([np.transpose(np.vstack([l_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([r_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        warp = np.zeros_like(image).astype(np.uint8)
        cv2.fillPoly(warp, np.int_([pts]), (0, 255, 0))

        unwarped = cv2.warpPerspective(warp, self._minv, (image.shape[1], image.shape[0]))
        result = cv2.addWeighted(image.copy(), 1, unwarped, 0.3, 0)

        return result

    def dump_input_frame(self, centroids):
        image = self._image_source.output
        return image

    def dump_output_frame(self, fits):
        return self.apply(fits)
