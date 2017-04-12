#!/usr/bin/env python3

import argparse
import collections
import os

import cv2
import numpy as np
from moviepy.editor import VideoFileClip, clips_array

import camera
import camera2


class ProcessPipeline(object):
    def __init__(self, camera_calibration_config, perspective_warp_config):
        self._camera_calibration = camera.CameraCalibration(camera_calibration_config)
        self._thresholding = camera.BinaryThreshold()
        self._perspective_warp = camera.PerspectiveWarp(perspective_warp_config.src, perspective_warp_config.dst)

        version_2 = True
        if version_2:
            self._lane_search = camera2.LaneSearchFitted(search_margin=150, window_height=80,
                                                         image_height=720, image_width=1280,
                                                         m_per_pix=(3.7 / 700, 30 / 720))
            self._display_lanes = camera.DisplayLaneSearchFittedUnwarped(self._camera_calibration,
                                                                         perspective_warp_config.src,
                                                                         perspective_warp_config.dst)
        else:
            self._lane_search = camera.LaneSearchFitted(search_margin=100, window_width=50, window_height=80,
                                                        image_height=720, image_width=1280)
            self._display_lanes = camera.DisplayLaneSearchFittedUnwarped(self._camera_calibration,
                                                                         perspective_warp_config.src,
                                                                         perspective_warp_config.dst)

        self._stages = collections.OrderedDict([
            ('1.cam_calibration', self._camera_calibration),
            ('2.thresholding', self._thresholding),
            ('3.perspective_warp', self._perspective_warp),
            ('4.lane_search', self._lane_search),
            ('5.display_lanes', self._display_lanes),
            #('grayscaled', camera.ScaleBinaryToGrayscale())
        ])

    def process_frame(self, image, limit=-1):
        frame = image
        idx = 0
        for _, stage in self._stages.items():
            stage.process(frame)
            frame = stage.output
            idx += 1
            if limit > 0 and idx >= limit:
                break
        return frame

    def dump_stages(self, image):
        result = collections.OrderedDict()

        frame = image
        for name, stage in self._stages.items():
            result[name + '_in'] = stage.dump_input_frame(frame)
            stage.process(frame)
            result[name + '_out'] = stage.dump_output_frame(frame)
            frame = stage.output

        return result


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--clip', type=str, help='Clip file to load')
    parser.add_argument('--image', type=str, help='Image file to load')
    parser.add_argument('--dump-stages', action='store_true', help='Dump stages')
    parser.add_argument('target', type=str, help='Target frame file')
    parser.add_argument('--time', type=float, help='Timestamp of the frame')
    parser.add_argument('--time-from', type=float, help='Timestamp to start from')
    parser.add_argument('--time-to', type=float, help='Timestamp to end')
    parser.add_argument('--combine', action='store_true', help='Set to produce clip with several stages combined')

    args = parser.parse_args()

    offset_x = 400
    offset_y = 0
    width = 1280
    height = 720

    # obtained on second 20
    perspective_warp_config = camera.PerspectiveWarpConfig(src=np.float32([
        # Bottom line  left(x, y), right(x, y)
        [315, 680], [1025, 680],
        # Top line left(x, y), right(x, y)
        [601, 448], [689, 448]
    ]), dst=np.float32([
        [offset_x, height - offset_y], [width - offset_x, height - offset_y], [offset_x, offset_y], [width - offset_x, offset_y]
    ]))

    # obtained on second 20
    perspective_warp_config = camera.PerspectiveWarpConfig(src=np.float32([
        # Bottom line  left(x, y), right(x, y)
        [252, 690], [1056, 690],
        # Top line left(x, y), right(x, y)
        [601, 448], [689, 448]
    ]), dst=np.float32([
        [offset_x, height - offset_y], [width - offset_x, height - offset_y], [offset_x, offset_y], [width - offset_x, offset_y]
    ]))

    camera_calibration_config = camera.load_camera_calibration()

    process_pipeline = ProcessPipeline(camera_calibration_config, perspective_warp_config)

    def flip_colors(image):
        return image[:, :, ::-1]

    def flip_process_frame(frame, limit=-1):
        return flip_colors(process_pipeline.process_frame(flip_colors(frame), limit))

    if args.clip:
        clip = VideoFileClip(args.clip)
        if args.time is not None:
            frame = flip_colors(clip.get_frame(t=args.time))
            stages_dump = process_pipeline.dump_stages(frame)

            for name, image in stages_dump.items():
                if (len(image.shape) < 3) or (image.shape[2] != 3):
                    cv2.imwrite(args.target + '.' + name + '.jpg', image * 255.0)
                else:
                    cv2.imwrite(args.target + '.' + name + '.jpg', image)

        else:
            if args.time_from is not None and args.time_to is not None:
                clip = clip.subclip(args.time_from, args.time_to)

            if not args.combine:
                processed_clip = clip.fl_image(flip_process_frame)
                processed_clip.write_videofile(args.target)
            else:
                # TODO: can have side-effects as pipeline is same
                thresholded = clip.fl_image(lambda f:
                                            flip_colors(camera.ensure_color(
                                                process_pipeline.process_frame(flip_colors(f), limit=2)))
                                            )
                warp = clip.fl_image(lambda f:
                                     flip_colors(camera.ensure_color(
                                         process_pipeline.process_frame(flip_colors(f), limit=3)))
                                     )
                complete = clip.fl_image(lambda f: flip_process_frame(f, limit=-1))
                combined_clip = clips_array([[thresholded, warp], [complete, complete]])
                combined_clip.write_videofile(args.target)

    else:
        frame = cv2.imread(args.image)

        fname, ext = os.path.splitext(args.target)

        if args.dump_stages:
            stages_dump = process_pipeline.dump_stages(frame)

            for name, image in stages_dump.items():
                if (len(image.shape) < 3) or (image.shape[2] != 3):
                    cv2.imwrite(fname + '.' + name + ext, image * 255.0)
                else:
                    cv2.imwrite(fname + '.' + name + ext, image)

        processed = process_pipeline.process_frame(frame)
        cv2.imwrite(args.target, processed)


if __name__ == '__main__':
    main()