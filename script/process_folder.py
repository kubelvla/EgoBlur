# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from functools import lru_cache
from typing import List

import cv2
import numpy as np
import torch
import torchvision


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--face_model_path",
        required=False,
        type=str,
        default=None,
        help="Absolute EgoBlur face model file path",
    )

    parser.add_argument(
        "--face_model_score_threshold",
        required=False,
        type=float,
        default=0.9,
        help="Face model score threshold to filter out low confidence detections",
    )

    parser.add_argument(
        "--lp_model_path",
        required=False,
        type=str,
        default=None,
        help="Absolute EgoBlur license plate model file path",
    )

    parser.add_argument(
        "--lp_model_score_threshold",
        required=False,
        type=float,
        default=0.9,
        help="License plate model score threshold to filter out low confidence detections",
    )

    parser.add_argument(
        "--nms_iou_threshold",
        required=False,
        type=float,
        default=0.3,
        help="NMS iou threshold to filter out low confidence overlapping boxes",
    )

    parser.add_argument(
        "--scale_factor_detections",
        required=False,
        type=float,
        default=1,
        help="Scale detections by the given factor to allow blurring more area, 1.15 would mean 15% scaling",
    )
    
    parser.add_argument(
        "--maximum_detection_size",
        required=False,
        type=float,
        default=float('inf'),
        help="Discard false-positive detections that are obviously too large",
    )


    parser.add_argument("--input_folder", type=str, required=True, help="Path to input folder containing images")
    
    parser.add_argument("--output_folder", type=str, required=True, help="Path to output folder for anonymized images")

    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> argparse.Namespace:
    """
    parameter args: parsed arguments
    Run some basic checks on the input arguments
    """
    # input args value checks
    if not 0.0 <= args.face_model_score_threshold <= 1.0:
        raise ValueError(
            f"Invalid face_model_score_threshold {args.face_model_score_threshold}"
        )
    if not 0.0 <= args.lp_model_score_threshold <= 1.0:
        raise ValueError(
            f"Invalid lp_model_score_threshold {args.lp_model_score_threshold}"
        )
    if not 0.0 <= args.nms_iou_threshold <= 1.0:
        raise ValueError(f"Invalid nms_iou_threshold {args.nms_iou_threshold}")
    if not 0 <= args.scale_factor_detections:
        raise ValueError(
            f"Invalid scale_factor_detections {args.scale_factor_detections}"
        )

    # input/output paths checks
    if args.face_model_path is None and args.lp_model_path is None:
        raise ValueError(
            "Please provide either face_model_path or lp_model_path or both"
        )

    return args


@lru_cache
def get_device() -> str:
    """
    Return the device type
    """
    return (
        "cpu"
        if not torch.cuda.is_available()
        else f"cuda:{torch.cuda.current_device()}"
    )


def read_image(image_path: str) -> np.ndarray:
    """
    parameter image_path: absolute path to an image
    Return an image in BGR format
    """
    bgr_image = cv2.imread(image_path)
    if len(bgr_image.shape) == 2:
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)
    return bgr_image


def write_image(image: np.ndarray, image_path: str) -> None:
    """
    parameter image: np.ndarray in BGR format
    parameter image_path: absolute path where we want to save the visualized image
    """
    cv2.imwrite(image_path, image)


def get_image_tensor(bgr_image: np.ndarray) -> torch.Tensor:
    """
    parameter bgr_image: image on which we want to make detections

    Return the image tensor
    """
    bgr_image_transposed = np.transpose(bgr_image, (2, 0, 1))
    image_tensor = torch.from_numpy(bgr_image_transposed).to(get_device())

    return image_tensor


def get_detections(
    detector: torch.jit._script.RecursiveScriptModule,
    image_tensor: torch.Tensor,
    model_score_threshold: float,
    nms_iou_threshold: float,
) -> List[List[float]]:
    """
    parameter detector: Torchscript module to perform detections
    parameter image_tensor: image tensor on which we want to make detections
    parameter model_score_threshold: model score threshold to filter out low confidence detection
    parameter nms_iou_threshold: NMS iou threshold to filter out low confidence overlapping boxes

    Returns the list of detections
    """
    with torch.no_grad():
        detections = detector(image_tensor)

    boxes, _, scores, _ = detections  # returns boxes, labels, scores, dims

    nms_keep_idx = torchvision.ops.nms(boxes, scores, nms_iou_threshold)
    boxes = boxes[nms_keep_idx]
    scores = scores[nms_keep_idx]

    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()

    score_keep_idx = np.where(scores > model_score_threshold)[0]
    boxes = boxes[score_keep_idx]
    return boxes.tolist()


def scale_box(
    box: List[List[float]], max_width: int, max_height: int, scale: float
) -> List[List[float]]:
    """
    parameter box: detection box in format (x1, y1, x2, y2)
    parameter scale: scaling factor

    Returns a scaled bbox as (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    w = x2 - x1
    h = y2 - y1

    xc = x1 + w / 2
    yc = y1 + h / 2

    w = scale * w
    h = scale * h

    x1 = max(xc - w / 2, 0)
    y1 = max(yc - h / 2, 0)

    x2 = min(xc + w / 2, max_width)
    y2 = min(yc + h / 2, max_height)

    return [x1, y1, x2, y2]


def visualize(
    image: np.ndarray,
    detections: List[List[float]],
    scale_factor_detections: float,
    maximum_detection_size
) -> np.ndarray:
    """
    parameter image: image on which we want to make detections
    parameter detections: list of bounding boxes in format [x1, y1, x2, y2]
    parameter scale_factor_detections: scale detections by the given factor to allow blurring more area, 1.15 would mean 15% scaling

    Visualize the input image with the detections and save the output image at the given path
    """
    image_fg = image.copy()
    mask_shape = (image.shape[0], image.shape[1], 1)
    mask = np.full(mask_shape, 0, dtype=np.uint8)

    for box in detections:
        if scale_factor_detections != 1.0:
            box = scale_box(
                box, image.shape[1], image.shape[0], scale_factor_detections
            )
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        w = x2 - x1
        h = y2 - y1
        
        if w >= maximum_detection_size or h >= maximum_detection_size:
            continue
 
        ksize = (image.shape[0] // 2, image.shape[1] // 2)
        image_fg[y1:y2, x1:x2] = cv2.blur(image_fg[y1:y2, x1:x2], ksize)
        cv2.ellipse(mask, (((x1 + x2) // 2, (y1 + y2) // 2), (w, h), 0), 255, -1)

    inverse_mask = cv2.bitwise_not(mask)
    image_bg = cv2.bitwise_and(image, image, mask=inverse_mask)
    image_fg = cv2.bitwise_and(image_fg, image_fg, mask=mask)
    image = cv2.add(image_bg, image_fg)

    return image


def anonymize_image(
    input_image_path: str,
    face_detector: torch.jit._script.RecursiveScriptModule,
    lp_detector: torch.jit._script.RecursiveScriptModule,
    face_model_score_threshold: float,
    lp_model_score_threshold: float,
    nms_iou_threshold: float,
    scale_factor_detections: float,
    maximum_detection_size: float
):
    """
    parameter input_image_path: absolute path to the input image
    parameter face_detector: face detector model to perform face detections
    parameter lp_detector: face detector model to perform face detections
    parameter face_model_score_threshold: face model score threshold to filter out low confidence detection
    parameter lp_model_score_threshold: license plate model score threshold to filter out low confidence detection
    parameter nms_iou_threshold: NMS iou threshold
    parameter output_image_path: absolute path where the visualized image will be saved
    parameter scale_factor_detections: scale detections by the given factor to allow blurring more area

    Perform detections on the input image and returns it
    """
    bgr_image = read_image(input_image_path)
    image = bgr_image.copy()

    image_tensor = get_image_tensor(bgr_image)
    image_tensor_copy = image_tensor.clone()
    detections = []
    # get face detections
    if face_detector is not None:
        detections.extend(
            get_detections(
                face_detector,
                image_tensor,
                face_model_score_threshold,
                nms_iou_threshold,
            )
        )

    # get license plate detections
    if lp_detector is not None:
        detections.extend(
            get_detections(
                lp_detector,
                image_tensor_copy,
                lp_model_score_threshold,
                nms_iou_threshold,
            )
        )
    image = visualize(
        image,
        detections,
        scale_factor_detections,
        maximum_detection_size
    )
    
    return image


def process_folder(input_folder, output_folder, face_detector, lp_detector, args):
    create_output_directory(output_folder)
    file_counter = 0
    total_files = len(os.listdir(input_folder))
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        if os.path.isfile(input_path) and filename.lower().endswith(('png', 'jpg', 'jpeg')):
            file_counter += 1
            print(f"Processing file {file_counter}/{total_files}")
            anonymized_image =  anonymize_image(input_path, 
                                face_detector, 
                                lp_detector, 
                                args.face_model_score_threshold, 
                                args.lp_model_score_threshold,
                                args.nms_iou_threshold,
                                args.scale_factor_detections,
                                args.maximum_detection_size)
            write_image(anonymized_image, output_path)
            

def create_output_directory(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == "__main__":
    args = validate_inputs(parse_args())
    if args.face_model_path is not None:
        face_detector = torch.jit.load(args.face_model_path, map_location="cpu").to(
            get_device()
        )
        face_detector.eval()
    else:
        face_detector = None

    if args.lp_model_path is not None:
        lp_detector = torch.jit.load(args.lp_model_path, map_location="cpu").to(
            get_device()
        )
        lp_detector.eval()
    else:
        lp_detector = None

    if args.input_folder is not None:
        process_folder(args.input_folder, args.output_folder, face_detector, lp_detector, args)
      
        

