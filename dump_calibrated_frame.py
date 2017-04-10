#!/usr/bin/env python3

import argparse

from moviepy.editor import VideoFileClip

import camera


def store_calibrated_frame(clip, camera_calibration_config, time, filename):
    calibration = camera.CameraCalibration(camera_calibration_config)
    calibrated_clip = clip.fl_image(calibration.apply)
    calibrated_clip.save_frame(filename, t=time)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('clip_file', type=str, help='Clip file')
    parser.add_argument('target', type=str, help='Target frame file')
    parser.add_argument('--time', type=float, help='Timestamp of the frame')

    args = parser.parse_args()

    camera_calibration_config = camera.load_camera_calibration()
    clip = VideoFileClip(args.clip_file)
    store_calibrated_frame(clip, camera_calibration_config, args.time, args.target)


if __name__ == '__main__':
    main()