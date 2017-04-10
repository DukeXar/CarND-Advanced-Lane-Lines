#!/usr/bin/env python3

import argparse

import numpy as np
from moviepy.editor import VideoFileClip
import cv2

import camera


def draw_warp_shape(image, src):
    pts = np.array([
        src[0], src[1], src[3], src[2],
    ], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(image, [pts], True, (255, 0, 0))


def store_calibrated_frame(clip, camera_calibration_config, perspective_warp_config, time, filename):
    perspective_warp = camera.PerspectiveWarp(perspective_warp_config.src, perspective_warp_config.dst)
    calibration = camera.CameraCalibration(camera_calibration_config)

    def process(image):
        #draw_warp_shape(image, perspective_warp_config.src)
        result = perspective_warp.apply(calibration.apply(image))
        draw_warp_shape(result, perspective_warp_config.dst)
        return result

    calibrated_clip = clip.fl_image(process)
    calibrated_clip.save_frame(filename, t=time)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('clip_file', type=str, help='Clip file')
    parser.add_argument('target', type=str, help='Target frame file')
    parser.add_argument('--time', type=float, help='Timestamp of the frame')

    args = parser.parse_args()

    camera_calibration_config = camera.load_camera_calibration()
    clip = VideoFileClip(args.clip_file)

    offset = 100

    perspective_warp_config = camera.PerspectiveWarpConfig(src=np.float32([
        [335, 660], [1086, 660], [608, 448], [682, 448]
    ]), dst=np.float32([
        [offset, clip.h], [clip.w - offset, clip.h], [offset, 0], [clip.w - offset, 0]
    ]))

    store_calibrated_frame(clip, camera_calibration_config,
                           perspective_warp_config, args.time, args.target)


if __name__ == '__main__':
    main()