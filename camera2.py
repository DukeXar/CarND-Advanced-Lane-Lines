import cv2
import numpy as np
from camera import Processor, QuadraticLaneFunc, draw_fitted_lanes_warped, sobel


# class PolyLaneFunc(LaneFunc):
#     def __init__(self, order):
#         self._fit = None
#         self._order = order
#
#     def apply(self, ploty):
#         res = 0
#         for idx, f in enumerate(self._fit):
#             res += f * ploty ** (self._order - idx)
#         return res
#
#     def load(self, points):
#         self._fit = np.polyfit([item[1] for item in points], [item[0] for item in points], self._order)
#
#     @property
#     def loaded(self): return self._fit is not None
#
#     @property
#     def fit(self): return self._fit



def find_initial_centroids(image, left, right, height_k=0.25):
    scan_height = int(image.shape[0] * height_k)
    histogram = np.sum(image[scan_height:, left:right], axis=0)
    max_idx = np.argmax(histogram)
    base = max_idx + left
    return base


def find_centroids_and_points(image, window_height, search_margin, center_x):
    threshold_pixels = 50

    height = image.shape[0]
    nwindows = int(height / window_height)

    nonzero = image.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    center_x_curr = center_x

    indices = []
    centers = []
    for win_idx in range(nwindows):
        win_y_bottom = height - win_idx * window_height
        win_y_top = win_y_bottom - window_height
        win_x_left = center_x_curr - int(search_margin / 2)
        win_x_right = center_x_curr + int(search_margin / 2)

        good_inds = ((nonzero_y >= win_y_top) & (nonzero_y < win_y_bottom) &
                     (nonzero_x >= win_x_left) & (nonzero_x < win_x_right)).nonzero()[0]

        centers.append((center_x_curr, win_y_top + window_height / 2))
        indices.append(good_inds)

        if len(good_inds) > threshold_pixels:
            center_x_curr = int(np.mean(nonzero_x[good_inds]))

    all_indices = np.concatenate(indices)

    all_x = nonzero_x[all_indices]
    all_y = nonzero_y[all_indices]

    all_points = [(x, y) for x, y in zip(all_x, all_y)]

    return centers, all_points


def find_centroids_and_points_nonlinear(image, window_height, search_margin, center_x):
    threshold_pixels = 50

    height = image.shape[0]

    nonzero = image.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    center_x_curr = center_x

    indices = []
    centers = []

    win_y_bottom = height
    window_height /= 20.
    while win_y_bottom - window_height > 0:
        win_y_top = win_y_bottom - window_height
        win_x_left = center_x_curr - int(search_margin / 2)
        win_x_right = center_x_curr + int(search_margin / 2)

        good_inds = ((nonzero_y >= win_y_top) & (nonzero_y < win_y_bottom) &
                     (nonzero_x >= win_x_left) & (nonzero_x < win_x_right)).nonzero()[0]

        indices.append(good_inds)

        if len(good_inds) > 1:
            centers.append((center_x_curr, win_y_top + window_height / 2))

        if len(good_inds) > threshold_pixels:
            center_x_curr = int(np.mean(nonzero_x[good_inds]))

        win_y_bottom -= window_height
        window_height *= 1.1

    all_indices = np.concatenate(indices)

    all_x = nonzero_x[all_indices]
    all_y = nonzero_y[all_indices]

    all_points = [(x, y) for x, y in zip(all_x, all_y)]

    return centers, all_points



class SingleLaneSearch(object):
    def __init__(self, window_height, search_margin, left, right, to_right, m_per_pix):
        self._window_height = window_height
        self._search_margin = search_margin
        self._left = left
        self._right = right
        self._current_centroids = []
        self._current_points = []
        self._current_lane_func = QuadraticLaneFunc()
        self._scaled_lane_func = QuadraticLaneFunc()
        self._to_right = to_right
        self._m_per_pix = m_per_pix

        self._length_y = 0

    def search(self, image):
        center_x = find_initial_centroids(image, self._left, self._right)
        centroids, points = find_centroids_and_points(image, self._window_height, self._search_margin, center_x)

        if centroids:
            self._current_lane_func.load(points)
            self._scaled_lane_func.load([(x * self._m_per_pix[0], y * self._m_per_pix[1]) for x, y in points])
            self._current_centroids = centroids
            self._current_points = points
            self._length_y = image.shape[0] - min((y for x, y in points))

    @property
    def current_lane_func(self):
        return self._current_lane_func

    @property
    def scaled_lane_func(self):
        return self._scaled_lane_func

    @property
    def current_centroids(self):
        return self._current_centroids

    @property
    def current_length_y(self):
        return self._length_y


def draw_centroids(lr_centroids, window_height, search_margin, out_image):
    if lr_centroids:
        l_centroids, r_centroids = lr_centroids

        for centroid in l_centroids:
            win_y_bottom = int(centroid[1] - window_height / 2)
            win_y_top = int(centroid[1] + window_height / 2)
            win_x_left = int(centroid[0] - search_margin / 2)
            win_x_right = int(centroid[0] + search_margin / 2)
            cv2.rectangle(out_image, (win_x_left, win_y_bottom), (win_x_right, win_y_top), (0, 255, 0), 2)

        for centroid in r_centroids:
            win_y_bottom = int(centroid[1] - window_height / 2)
            win_y_top = int(centroid[1] + window_height / 2)
            win_x_left = int(centroid[0] - search_margin / 2)
            win_x_right = int(centroid[0] + search_margin / 2)
            cv2.rectangle(out_image, (win_x_left, win_y_bottom), (win_x_right, win_y_top), (0, 0, 255), 2)


class LaneSearchFitted(Processor):
    def __init__(self, search_margin, window_height, image_width, image_height, m_per_pix):
        super().__init__()
        self._window_height = window_height
        self._search_margin = search_margin
        self._image_height = image_height
        self._image_width = image_width
        self._m_per_pix = m_per_pix

        middle = int(self._image_width / 2)
        self._l_lane = SingleLaneSearch(self._window_height, self._search_margin,
                                        0, middle, to_right=False, m_per_pix=m_per_pix)
        self._r_lane = SingleLaneSearch(self._window_height, self._search_margin,
                                        middle + 1, self._image_width, to_right=True, m_per_pix=m_per_pix)

        self._last_result = None

    def apply(self, image):
        assert image.shape[0:2] == (self._image_height, self._image_width), \
            "Image dimensions must match: {} != {}".format(image.shape[0:1], (self._image_height, self._image_width))

        self._l_lane.search(image)
        self._r_lane.search(image)

        l_curve_rad = self._l_lane.scaled_lane_func.get_curvative(self._image_height * self._m_per_pix[1])
        r_curve_rad = self._r_lane.scaled_lane_func.get_curvative(self._image_height * self._m_per_pix[1])

        print('OOOOO: {} vs {}'.format(l_curve_rad, r_curve_rad))
        print('XXXXX: {} vs {}'.format(self._l_lane.current_length_y, self._r_lane.current_length_y))

        l_x = self._l_lane.current_lane_func.apply(self._image_height)
        r_x = self._r_lane.current_lane_func.apply(self._image_height)

        # Select one that was recognized best, and adjust second line accordingly
        if self._l_lane.current_length_y > self._r_lane.current_length_y:
            l_lane_func = self._l_lane.current_lane_func
            r_lane_func = self._l_lane.current_lane_func.shift(r_x - l_x)
        else:
            l_lane_func = self._r_lane.current_lane_func.shift(l_x - r_x)
            r_lane_func = self._r_lane.current_lane_func

        self._last_result = (l_lane_func, r_lane_func)
        return self._last_result

    def dump_input_frame(self, image):
        return image

    def dump_output_frame(self, image):
        left_func, right_func = self._last_result
        result = draw_fitted_lanes_warped(image, left_func, right_func, self._search_margin)
        draw_centroids([self._l_lane.current_centroids, self._r_lane.current_centroids],
                       self._window_height, self._search_margin, result)
        return result

    @property
    def search_margin(self):
        return self._search_margin


