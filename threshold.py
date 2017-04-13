import cv2
import numpy as np

from pipeline import Processor


class BinaryThreshold(Processor):
    """
    Applies binary thresholding using sobel operators and color tresholding.
    Sobel operators allow to select directed lines, color tresholding anded with
    those lines select just those which look like lane marking.
    """
    def __init__(self):
        super().__init__()

    def _mag_thresh(self, sobelx, sobely, channel, mag_thresh=(0, 255)):
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag) / 255.
        gradmag = (gradmag / scale_factor).astype(np.uint8)
        binary_output = np.zeros_like(channel)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
        return binary_output

    def _dir_threshold(self, sobelx, sobely, channel, thresh=(0, np.pi / 2)):
        # Take the absolute value of the gradient direction,
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output = np.zeros_like(channel)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
        return binary_output

    def _grad_threshold(self, sobel, channel, thresh=(0, 255)):
        sobel = np.absolute(sobel)
        sobel = (255 * sobel / np.max(sobel)).astype(np.uint8)
        binary_output = np.zeros_like(channel)
        binary_output[(sobel >= thresh[0]) & (sobel <= thresh[1])] = 1
        return binary_output

    def _sobel(self, channel, mag_thresh=(0, 255), dir_thresh=(0, np.pi / 2), grad_thresh_x=(7, 200),
               grad_thresh_y=(7, 200), ksize=3):
        sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=ksize)
        sobely = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=ksize)
        binary_output = np.zeros_like(channel)
        binary_mag = self._mag_thresh(sobelx, sobely, binary_output, mag_thresh)
        binary_dir = self._dir_threshold(sobelx, sobely, binary_output, dir_thresh)
        binary_grad_x = self._grad_threshold(sobelx, binary_output, thresh=grad_thresh_x)
        binary_grad_y = self._grad_threshold(sobely, binary_output, thresh=grad_thresh_y)
        binary_output[((binary_grad_x == 1) & (binary_grad_y == 1)) | ((binary_mag == 1) & (binary_dir == 1))] = 1
        return binary_output

    def _get_masks(self, image):
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS_FULL).astype(np.float32)

        # Treshold directed lines on saturation channel (2)
        binary_sobel = self._sobel(hls[:, :, 2],
                                   mag_thresh=(5, 255), dir_thresh=(0.7, 1.1),
                                   grad_thresh_x=(2, 200), grad_thresh_y=(2, 200),
                                   ksize=21)

        # Select yellow lines (look at hue channel mainly)
        yellow_line = np.zeros_like(hls[:, :, 0])
        yellow_line[(hls[:, :, 0] >= 25) & (hls[:, :, 0] <= 50) & (hls[:, :, 2] > 60) & (binary_sobel == 1)] = 1

        # Select white lines (looking at all rgb channels)
        white_line = np.zeros_like(hls[:, :, 0])
        white_line[(image[:, :, 0] > 190) & (image[:, :, 1] > 190) & (image[:, :, 2] > 190) & (binary_sobel == 1)] = 1

        # For processing, only white and yellow lines are used, binary sobel mask would be shown as red channel in
        # dump_output_frame call.
        return [white_line, yellow_line, binary_sobel]

    def apply(self, image):
        white_line, yellow_line, _ = self._get_masks(image)
        result = np.zeros_like(white_line)
        result[(white_line == 1) | (yellow_line == 1)] = 1
        return result

    def dump_input_frame(self, image):
        return image.copy()

    def dump_output_frame(self, image):
        masks = self._get_masks(image)
        result = np.dstack(masks)
        return result * 255.