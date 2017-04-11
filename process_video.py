#!/usr/bin/env python3

import collections
from moviepy.editor import VideoFileClip, clips_array

import camera
import cv2
import os
import pickle
import logging
import argparse
import numpy as np


class ProcessPipeline(object):
    def __init__(self, camera_calibration_config, perspective_warp_config):
        self._camera_calibration = camera.CameraCalibration(camera_calibration_config)
        self._thresholding = camera.BinaryThreshold()
        self._perspective_warp = camera.PerspectiveWarp(perspective_warp_config.src, perspective_warp_config.dst)

        show_centroids = False
        if show_centroids:
            self._lane_search = camera.LaneSearchCentroids(search_margin=100, window_width=50, window_height=80)
            self._display_lanes = camera.DisplayLaneSearchCentroids(self._perspective_warp,
                                                                    window_width=self._lane_search.window_width,
                                                                    window_height=self._lane_search.window_height)
        else:
            self._lane_search = camera.LaneSearchFitted(search_margin=100, window_width=50, window_height=80,
                                                        image_height=720, image_width=1280)
            self._display_lanes = camera.DisplayLaneSearchFittedUnwarped(self._camera_calibration,
                                                                         perspective_warp_config.src,
                                                                         perspective_warp_config.dst)

        self._stages = collections.OrderedDict([
            ('cam_calibration', self._camera_calibration),
            ('thresholding', self._thresholding),
            ('perspective_warp', self._perspective_warp),
            ('lane_search', self._lane_search),
            ('display_lanes', self._display_lanes),
            #('grayscaled', camera.ScaleBinaryToGrayscale())
        ])

    def process_frame(self, image):
        frame = image
        for _, stage in self._stages.items():
            stage.process(frame)
            frame = stage.output
        return frame

    def dump_stages(self, image):
        result = collections.OrderedDict()

        frame = image
        for name, stage in self._stages.items():
            result[name + '_in'] = stage.dump_input_frame(frame)
            result[name + '_out'] = stage.dump_output_frame(frame)
            stage.process(frame)
            frame = stage.output

        return result


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('clip_file', type=str, help='Clip file')
    parser.add_argument('target', type=str, help='Target frame file')
    parser.add_argument('--time', type=float, help='Timestamp of the frame')
    parser.add_argument('--time-from', type=float, help='Timestamp to start from')
    parser.add_argument('--time-to', type=float, help='Timestamp to end')

    args = parser.parse_args()

    clip = VideoFileClip(args.clip_file)

    offset = 400

    # TODO: In theory, if camera is centered and rotated properly, coordinates should be symmetrical

    # obtained on second 20
    perspective_warp_config = camera.PerspectiveWarpConfig(src=np.float32([
        # Bottom line  left(x, y), right(x, y)
        [315, 680], [1025, 680],
        # Top line left(x, y), right(x, y)
        [601, 448], [689, 448]
    ]), dst=np.float32([
        [offset, clip.h], [clip.w - offset, clip.h], [offset, 0], [clip.w - offset, 0]
    ]))


    camera_calibration_config = camera.load_camera_calibration()

    process_pipeline = ProcessPipeline(camera_calibration_config, perspective_warp_config)

    if args.time is not None:
        frame = clip.get_frame(t=args.time)
        stages_dump = process_pipeline.dump_stages(frame)

        for name, image in stages_dump.items():
            # Use [:,:,::-1] to flip BGR to RGB
            if (len(image.shape) < 3) or (image.shape[2] != 3):
                cv2.imwrite(args.target + name + '.jpg', image * 255.0)
            else:
                cv2.imwrite(args.target + name + '.jpg', image[:, :, ::-1])
    elif args.time_from is not None and args.time_to is not None:
        clip = clip.subclip(args.time_from, args.time_to)
        processed_clip = clip.fl_image(process_pipeline.process_frame)
        processed_clip.write_videofile(args.target)
    else:
        processed_clip = clip.fl_image(process_pipeline.process_frame)
        processed_clip.write_videofile(args.target)

    #combined_clip = clips_array([[clip], [processed_clip]])
    #combined_clip.write_videofile('./project_video_undistorted.mp4')


if __name__ == '__main__':
    main()