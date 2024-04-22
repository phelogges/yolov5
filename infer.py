import argparse
import os
import platform
import sys
from pathlib import Path

import torch
from ultralytics.data import augment

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    scale_segments,
    strip_optimizer,
)
from utils.segment.general import masks2segments, process_mask, process_mask_native
from utils.torch_utils import select_device, smart_inference_mode

from datetime import datetime

from utils.augmentations import letterbox

import numpy as np


class Yolov5Segment:
    def __init__(self,
                 checkpoint: str,
                 class_file: str,
                 saved_dir: str,
                 device: int = -1,
                 imgsz=(640, 640),
                 conf_thres=0.25,
                 iou_thres=0.45,
                 max_det=1000,
                 agnostic_nms=False,
                 augment=False,
                 retina_masks=False,
                 save_img=True,
                 save_txt=True):
        self.device = torch.device("cpu" if device == -1 else "cuda:" + str(device))
        self.model = DetectMultiBackend(checkpoint, self.device, class_file)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.img_size = check_img_size(imgsz, s=self.stride)

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.retina_masks = retina_masks
        self.save_img = save_img
        self.save_txt = save_txt
        self.saved_dir = saved_dir
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir, exist_ok=True)

        self.TIME_FORMAT = '%Y-%m-%d-%H%M%S.%f'

    def _format_current_time(self) -> str:
        """
        格式化当前时间戳，格式为'%Y-%m-%d-%H:%M:%S.%f'
        :return: str，格式化后的时间戳
        """
        return datetime.now().strftime(self.TIME_FORMAT)[:-3]

    def infer(self,
              bgr: np.ndarray):
        timestamp = self._format_current_time()
        save_path = os.path.join(self.saved_dir, timestamp)  # with suffix, added latter

        dt = Profile(device=self.device)
        seen = 0

        with dt[0]:
            im = letterbox(bgr, self.img_size, stride=self.stride, auto=self.pt)[0]
            im = im.transpose((2, 0, 1))[::-1]
            im = np.ascontiguousarray(im)

            im = torch.from_numpy(im).to(self.model.device)
            im = im.half() if self.model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]

        with dt[1]:
            pred, proto = self.model(im, augment=self.augment)[:2]

        with dt[2]:
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres,
                                       None, self.agnostic_nms, max_det=self.max_det,
                                       nm=32)

        im0 = bgr.copy()

        det = pred[0]

        seen += 1
        s = ""
        s += "%gx%g " % im.shape[2:]  # print string
        annotator = Annotator(im0, 3, example=str(self.names))
        if len(det):
            if self.retina_masks:
                # scale bbox first the crop masks
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                masks = process_mask_native(proto[0], det[:, 6:], det[:, :4], im0.shape[:2])  # HWC
            else:
                masks = process_mask(proto[0], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

            # save result
            result = [det[:, :6], masks]

            # Segments
            if self.save_txt:
                segments = [
                    scale_segments(im0.shape if self.retina_masks else im.shape[2:], x, im0.shape, normalize=True)
                    for x in reversed(masks2segments(masks))
                ]
            s = ""
            # Print results
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Mask plotting
            annotator.masks(
                masks,
                colors=[colors(x, True) for x in det[:, 5]],
                im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(self.device).permute(2, 0, 1).flip(
                    0).contiguous()
                       / 255
                if self.retina_masks
                else im[0],
            )

            # Write results
            for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                if self.save_txt:  # Write to file
                    seg = segments[j].reshape(-1)  # (n,2) to (n*2)
                    line = (cls, *seg, conf)  # label format
                    with open(f"{save_path}.txt", "a") as f:
                        f.write(("%g " * len(line)).rstrip() % line + "\n")

                if self.save_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = f"{self.names[c]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    # annotator.draw.polygon(segments[j], outline=colors(c, True), width=3)
                # if save_crop:
                #     save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

            im0 = annotator.result()
            if self.save_img:
                cv2.imwrite(f"{save_path}.jpg", im0)
        else:
            result = None

        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        # Print results
        t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
        LOGGER.info(
            f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *self.img_size)}" % t)

        return result


if __name__ == "__main__":
    import argparse


    def arg_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument("ckpt", type=str, help="Model checkpoint")
        parser.add_argument("img", type=str, help="Input image")
        parser.add_argument("class_file", type=str, help="Class file")
        parser.add_argument("saved_dir", type=str, help="Saved directory", default="./tmp")
        parser.add_argument("-d", "--device", type=int, help="GPU device, -1 for use CPU", default=0)
        parser.add_argument("-m", "--mode", type=int, default=1,
                            help="0 for Detection and 1 for Segmentation", choices=[0, 1])

        return parser.parse_args()

    args = arg_parser()

    bgr = cv2.imread(args.img)

    if args.mode == 1:
        segment = Yolov5Segment(args.ckpt, args.class_file, args.saved_dir, args.device)
    else:
        raise ValueError("Current only support segmentation task")

    result = segment.infer(bgr)

    print(result)
