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
        #self._sobel_x_thresh = (20, 100)
        #self._sobel_x_thresh = (20, 70)

    def _sobel(self, channel, x_thresh, ksize=5):
        sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=ksize)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= x_thresh[0]) & (scaled_sobel <= x_thresh[1])] = 1
        return sxbinary

    def _sobel_threshold(self, hls):
        s_channel = hls[:, :, 2]
        return self._sobel(s_channel, (20, 70))

    def _get_masks_old(self, image):
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float32)

        binary_color = np.zeros_like(hls[:, :, 2])
        channel = hls[:, :, 2]
        binary_color[(channel > 90) & (channel <= 255)] = 1

        binary_color2 = np.zeros_like(hls[:, :, 2])
        channel = image.astype(np.float32)[:, :, 0]
        binary_color2[(channel > 190) & (channel <= 255) & (binary_color == 1)] = 1

        binary_sobel = self._sobel_threshold(hls)

        return [np.zeros_like(binary_color2), binary_color2, binary_sobel]

    def _get_masks(self, image):
        yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0)
        cl1 = clahe.apply(yuv[:, :, 0])
        yuv[:, :, 0] = cl1
        image = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float32)

        binary_color2 = np.zeros_like(hls[:, :, 2])
        channel = image.astype(np.float32)[:, :, 0]
        binary_color2[(channel > 210) & (channel <= 255)] = 1

        #binary_sobel = self._sobel(hls[:, :, 2], (25, 100))
        binary_sobel = np.zeros_like(hls[:, :, 2])

        yellow_line = np.zeros_like(hls[:, :, 0])
        yellow_line[(hls[:, :, 0] >= 18) & (hls[:, :, 0] <= 28) & (hls[:, :, 2] > 45)] = 1
        #yellow_line = self._sobel(yellow_line, (40, 100))

        #white_line = np.zeros_like(hls[:, :, 0])
        #white_line[(image[:, :, 0] > 195) & (image[:, :, 1] > 195) & (image[:, :, 2] > 50)] = 1

        white_line = np.zeros_like(hls[:, :, 0])
        white_line[(image[:, :, 0] > 220) & (image[:, :, 1] > 220) & (image[:, :, 2] > 200)] = 1

        return [white_line, yellow_line, binary_sobel]


    def apply(self, image):
        masks = self._get_masks(image)
        result = np.zeros_like(masks[0])
        result[(masks[0] == 1) | (masks[1] == 1) | (masks[2] == 1)] = 1
        return result

    def dump_input_frame(self, image):
        return image.copy()

    def dump_output_frame(self, image):
        masks = self._get_masks(image)
        result = np.dstack(masks)
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


def find_initial_centroids_lane(image, window_width, left, right, height_k=0.5):
    window = np.ones(window_width)  # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template
    # Sum quarter bottom of image to get slice, could use a different ratio
    sum_height = int(height_k * image.shape[0])
    a_sum = np.sum(image[sum_height:, left:right], axis=0)
    convolved = np.convolve(window, a_sum)
    max_idx = np.argmax(convolved)

    if convolved[max_idx]:
        center_x = max_idx - window_width / 2 + left
        return center_x

    return None

    #histogram = np.sum(image[image.shape[0] / 2:, :], axis=0)
    #midpoint = np.int(histogram.shape[0] / 2)
    #leftx_base = np.argmax(histogram[:midpoint])
    #rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    #return center_x


def find_window_centroids_lane(image, window_width, window_height, margin, center_x):
    window = np.ones(window_width)  # Create our window template that we will use for convolutions

    window_centroids = []

    image_height = image.shape[0]

    # Go through each layer looking for max pixel locations
    for level in range(0, int(image_height / window_height)):
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

        argmax_idx = np.argmax(conv_signal[min_index:max_index])

        # Skip empty centroids
        if conv_signal[min_index:max_index][argmax_idx]:
            center_x = argmax_idx + min_index - offset
            window_centroids.append((center_x, top + window_height / 2))

    return window_centroids


def window_mask(width, height, img_ref, center_x, center_y):
    output = np.zeros_like(img_ref)
    output[max(0, int(center_y - height / 2)):min(int(center_y + height / 2), img_ref.shape[0]),
           max(0, int(center_x - width / 2)):min(int(center_x + width / 2), img_ref.shape[1])] = 1
    return output


def draw_centroids(centroids, window_width, window_height, image):
    l_points = np.zeros_like(image)
    r_points = np.zeros_like(image)

    if centroids:
        l_centroids, r_centroids = centroids

        for centroid in l_centroids:
            mask = window_mask(window_width, window_height, image, centroid[0], centroid[1])
            l_points[(l_points == 1) | (mask == 1)] = 1

        for centroid in r_centroids:
            mask = window_mask(window_width, window_height, image, centroid[0], centroid[1])
            r_points[(r_points == 1) | (mask == 1)] = 1

        # Draw the results
        # add both left and right window pixels together
        template = np.array(r_points + l_points, np.uint8)
        zero_channel = np.zeros_like(template)
        # make window pixels green
        template = np.array(cv2.merge((zero_channel, template * 255, zero_channel)), np.uint8)

        # making the original road pixels 3 color channels
        warpage = np.array(cv2.merge((image, image, image)), np.uint8)

        # overlay the orignal road image with window results
        output = cv2.addWeighted(warpage, 0.5, template, 0.5, 0.0)

    else:
        output = np.array(cv2.merge((image, image, image)), np.uint8)

    return output


class LaneSearchCentroids(Processor):
    def __init__(self, window_width, window_height, search_margin):
        super().__init__()
        # TODO: What about using non-linear grid?
        self._window_width = window_width
        self._window_height = window_height
        self._search_margin = search_margin

    def apply(self, image):
        middle = int(image.shape[1] / 2)
        l_center = (repeatedly_find_initial_centroids_lane(image, self._window_width, 0, middle), image.shape[0])
        l_centroids = find_window_centroids_lane(image, self._window_width, self._window_height,
                                                 self._search_margin, l_center[0])

        r_center = (repeatedly_find_initial_centroids_lane(image, self._window_width, middle, image.shape[1]), image.shape[0])
        r_centroids = find_window_centroids_lane(image, self._window_width, self._window_height,
                                                 self._search_margin, r_center[0])

        return l_centroids, r_centroids

    def dump_input_frame(self, image):
        return image

    def dump_output_frame(self, image):
        centroids = self.apply(image)
        return draw_centroids(centroids, self._window_width, self._window_height, image * 255)

    @property
    def window_width(self): return self._window_width

    @property
    def window_height(self): return self._window_height


class DisplayLaneSearchCentroids(Processor):
    def __init__(self, image_source_warped, window_width, window_height):
        super().__init__()
        self._image_source = image_source_warped
        self._window_width = window_width
        self._window_height = window_height

    def apply(self, centroids):
        image = self._image_source.output
        return draw_centroids(centroids, self._window_width, self._window_height, image * 255)

    def dump_input_frame(self, centroids):
        image = self._image_source.output
        return image

    def dump_output_frame(self, centroids):
        return self.apply(centroids)


def repeatedly_find_initial_centroids_lane(image, window_width, left, right):
    center_x = None

    for idx in range(2, 5):
        center_x = find_initial_centroids_lane(image, window_width, left, right,
                                               height_k=idx * 0.1)
        if center_x is not None:
            break

    if center_x is None:
        center_x = left + (right - left) / 2

    return center_x


class LaneFunc(object):
    def apply(self, ploty):
        raise NotImplementedError()

    def load(self, points):
        raise NotImplementedError()


class QuadraticLaneFunc(LaneFunc):
    def __init__(self):
        self._fit = None

    def apply(self, ploty):
        return self._fit[0] * ploty ** 2 + self._fit[1] * ploty + self._fit[2]

    def load(self, points):
        self._fit = np.polyfit([item[1] for item in points], [item[0] for item in points], 2)


import scipy.interpolate


class SplineLaneFunc(LaneFunc):
    def __init__(self):
        self._spline = None

    def apply(self, ploty):
        return self._spline(ploty)

    def load(self, points):
        self._spline = scipy.interpolate.UnivariateSpline([item[1] for item in points],
                                                          [item[0] for item in points])


class SingleLaneSearch(object):
    def __init__(self, window_width, window_height, search_margin, left, right):
        self._window_width = window_width
        self._window_height = window_height
        self._search_margin = search_margin
        self._left = left
        self._right = right
        self._current_centroids = []
        self._current_lane_func = QuadraticLaneFunc()

    def apply(self, image):
        center_x = repeatedly_find_initial_centroids_lane(image, self._window_width, self._left, self._right)

        centroids = find_window_centroids_lane(image, self._window_width, self._window_height,
                                               self._search_margin, center_x)
        if centroids:
            self._current_lane_func.load(centroids)
            self._current_centroids = centroids

    @property
    def current_lane_func(self):
        return self._current_lane_func

    @property
    def current_centroids(self):
        return self._current_centroids


def get_search_points(func, ploty, search_margin):
    fitx = func.apply(ploty)
    line_window1 = np.array([np.transpose(np.vstack([fitx - search_margin, ploty]))])
    line_window2 = np.array([np.flipud(np.transpose(np.vstack([fitx + search_margin, ploty])))])
    pts = np.hstack((line_window1, line_window2))
    return pts


def draw_fitted_lanes_warped(image, l_func, r_func, search_margin, left_color=(0, 255, 0), right_color=(0, 255, 0)):
    out_img = np.dstack((image, image, image)) * 255
    window_img = np.zeros_like(out_img)

    ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])

    left_line_pts = get_search_points(l_func, ploty, search_margin)
    cv2.fillPoly(window_img, np.int_([left_line_pts]), left_color)

    right_line_pts = get_search_points(r_func, ploty, search_margin)
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

        return self._l_lane.current_lane_func, self._r_lane.current_lane_func

    def dump_input_frame(self, image):
        return image

    def dump_output_frame(self, image):
        left_func, right_func = self.apply(image)
        return draw_fitted_lanes_warped(image, left_func, right_func, self._search_margin)

    @property
    def search_margin(self):
        return self._search_margin


class DisplayLaneSearchFitted(Processor):
    def __init__(self, image_source_warped, search_margin):
        super().__init__()
        self._image_source = image_source_warped
        self._search_margin = search_margin

    def apply(self, lane_funcs):
        image = self._image_source.output
        return draw_fitted_lanes_warped(image, lane_funcs[0], lane_funcs[1], self._search_margin)

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

    def apply(self, lane_funcs):
        image = self._image_source.output

        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        l_fitx = lane_funcs[0].apply(ploty)
        r_fitx = lane_funcs[1].apply(ploty)

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
