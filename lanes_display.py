import cv2
import numpy as np

from pipeline import Processor


def draw_fitted_lanes_warped(image, l_func, r_func, search_margin, left_color=(0, 255, 0), right_color=(0, 255, 0)):
    """
    Draws fitted lanes with search margin, overlayed on a warped image.
    Returns a new image.
    """
    out_img = np.dstack((image, image, image)) * 255
    window_img = np.zeros_like(out_img)

    ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])

    if l_func.loaded:
        left_line_pts = get_lane_search_points(l_func, ploty, search_margin)
        cv2.fillPoly(window_img, np.int_([left_line_pts]), left_color)

    if r_func.loaded:
        right_line_pts = get_lane_search_points(r_func, ploty, search_margin)
        cv2.fillPoly(window_img, np.int_([right_line_pts]), right_color)

    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    return result


def get_lane_search_points(func, ploty, search_margin):
    """
    Returns points to display fitted lane search margin.
    """
    fitx = func.apply(ploty)
    line_window1 = np.array([np.transpose(np.vstack([fitx - search_margin, ploty]))])
    line_window2 = np.array([np.flipud(np.transpose(np.vstack([fitx + search_margin, ploty])))])
    pts = np.hstack((line_window1, line_window2))
    return pts


class DisplayLaneSearchFittedUnwarped(Processor):
    """
    This really displays the final image. It takes input image of the pipeline (undistorted) as a side channel
    in `image_source`, and then displays `items`, which are lane functions and lane parameters, on the image.
    """
    def __init__(self, image_source, src, dst):
        super().__init__()
        self._image_source = image_source
        self._minv = cv2.getPerspectiveTransform(dst, src)

    def apply(self, items):
        l_func, r_func, curv, car_shift_m = items

        image = self._image_source.output
        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])

        # Draw lane line on a warped image, then unwarp it and overlay on the input image.
        warp = np.zeros_like(image).astype(np.uint8)

        if not l_func.loaded or not r_func.loaded:
            error = 'No lane found'
        else:
            l_fitx = l_func.apply(ploty)
            r_fitx = r_func.apply(ploty)
            pts_left = np.array([np.transpose(np.vstack([l_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([r_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))
            cv2.fillPoly(warp, np.int_([pts]), (0, 255, 0))
            error = ''

        unwarped = cv2.warpPerspective(warp, self._minv, (image.shape[1], image.shape[0]))
        result = cv2.addWeighted(image.copy(), 1, unwarped, 0.3, 0)

        # Display measurements.
        cv2.rectangle(result, (0, 0), (image.shape[1], 100), (0, 0, 0), -1)

        if not error:
            text = 'Curvative radius: {:.1f}m, car shift: {:.1f}m'.format(curv, car_shift_m)
            cv2.putText(result, text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(result, error, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2, cv2.LINE_AA)
        return result

    def dump_input_frame(self, items):
        return self._image_source.output

    def dump_output_frame(self, items):
        return self.output


class DisplayLaneSearchFitted(Processor):
    def __init__(self, image_source_warped, search_margin):
        super().__init__()
        self._image_source = image_source_warped
        self._search_margin = search_margin

    def apply(self, items):
        l_func, r_func, curv, car_shift_m = items
        image = self._image_source.output
        return draw_fitted_lanes_warped(image, l_func[0], r_func[1], self._search_margin)

    def dump_input_frame(self, centroids):
        image = self._image_source.output
        return image

    def dump_output_frame(self, fits):
        return self.apply(fits)
