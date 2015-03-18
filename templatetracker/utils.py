import cv2
import numpy as np

from menpo.image import Image


def is_gray(image):
    r"""
    Returns True if the image has one channel per pixel.
    """
    return image.ndim < 3


def width_height_divided_by(image, divisor):
    r"""
    Returns the height and width of the image divided by a value.
    """
    h, w = image.shape[:2]
    return h/divisor, w/divisor


def draw_bounding_box(image, bounding_box, color=(0, 0, 255), thickness=2):
    r"""
    Draws a bounding box on the image using opencv's rectangle function.
    """
    p0, _, p2, _ = bounding_box
    p0 = np.require(p0, dtype=np.int)
    p2 = np.require(p2, dtype=np.int)
    cv2.rectangle(image, tuple(p0[::-1]), tuple(p2[::-1]),
                  color=color, thickness=thickness)


def draw_landmarks(image, landmarks):
    r"""
    Draws landmarks on the image using opencv's circle function
    """
    for l in landmarks:
        cv2.circle(image, (int(l[1]), int(l[0])),
                   radius=1, color=(0, 255, 0), thickness=2)


def crop_image(image, bounding_box, safe_margin=0.5):
    x, y, w, h = bounding_box
    bb_range = np.array([x+w, y+h]) - np.array([x, y])
    bb_margin = bb_range * safe_margin
    extended_bb = bounding_box + np.hstack([-bb_margin, bb_margin])

    e_x, e_y, e_w, e_h = np.require(extended_bb, dtype=int)
    extended_xy = np.array([e_x, e_y])

    cropped_image = image[e_y:e_y+e_h, e_x:e_x+e_w]

    xy_difference = np.array([x, y]) - extended_xy
    cropped_bb = (xy_difference[0], xy_difference[1], xy_difference[0]+w,
                  xy_difference[1]+h)

    return cropped_image, cropped_bb, extended_xy


def preprocess_frame(frame):
    # resize
    frame = cv2.resize(frame, (640, 360))
    # bgr to rgb
    frame = frame[..., ::-1]
    # pixel values from 0 to 1
    frame = np.require(frame, dtype=np.double)
    frame /= 255
    # roll channel axis to the front
    frame = np.rollaxis(frame, -1)
    # build and return menpo image
    return Image(frame)