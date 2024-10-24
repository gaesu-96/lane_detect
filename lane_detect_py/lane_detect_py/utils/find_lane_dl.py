from lane_detect_py.DL.inference import inference
from lane_detect_py.utils.hough_transform import get_radian
import cv2
import numpy as np

def find_lane(frame, prev_center_x):
    output, frame = inference(frame) # new frame's shape is h * w, 1 chennel
    masked_output = ROI(output)
    danger_signal = danger_region(output)

    mask_condition1 = (masked_output[:, :, 0] == 255) & (masked_output[:, :, 1] == 0) & (masked_output[:, :, 2] == 255)
    frame[mask_condition1] = [255, 0, 0]

    mask_condition2 = (masked_output[:, :, 0] == 255) & (masked_output[:, :, 1] == 0) & (masked_output[:, :, 2] == 0)
    frame[mask_condition2] = (0, 255, 0) # lane

    mask_condition3 = (masked_output[:, :, 0] == 255) & (masked_output[:, :, 1] == 255) & (masked_output[:, :, 2] == 0)
    frame[mask_condition3] = (0, 0, 255) # road

    output_y = frame[frame.shape[0] // 2 + 200, :,:]
    road_idx = np.where((output_y[:,0] == 0) & (output_y[:,1] == 0) & (output_y[:,2] == 255))[0]
    lane = np.where((output_y[:,0] == 0) & (output_y[:,1] == 255) & (output_y[:,2] == 0))[0]

    if lane.size > 0:
        left_edge = lane[0]
        right_edge = lane[-1]
    else:
        left_edge = 2000
        right_edge = 2000

    final_idx = road_idx[(road_idx > left_edge) & (road_idx < right_edge)]

    if final_idx.size > 0:
        center_x = (final_idx[0] + final_idx[-1]) // 2

    else:
        center_x = prev_center_x

    if center_x is None:
        center_x = frame.shape[1] // 2

    cv2.circle(frame, (center_x, frame.shape[0] // 2 + 200), 5, (0, 255, 0), 2)
    cv2.line(frame, (frame.shape[1] // 2, 0), (frame.shape[1] // 2, frame.shape[0] - 1), (0, 0, 255), 1)
    if danger_signal:
        rad = 5.
    else:
        rad = get_radian(frame, center_x, frame.shape[0] // 2 + 200)



    return frame, center_x, rad

def danger_region(frame):
    h, w = frame.shape[:2]

    danger = np.array([
        [w // 2 - 150, h - 1],
        [w // 2 - 150, h - 130],
        [w // 2, h - 130],
        [w // 2, h - 1]
    ], dtype = np.int32)

    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, [danger], (255, 255, 255))
    danger_image = cv2.bitwise_and(frame, mask)

    cv2.imshow('test', danger_image)
    green_mask = (danger_image[:, :, 0] == 255) & (danger_image[:, :, 1] == 0) & (danger_image[:, :, 2] == 0)


    if green_mask.any():
        danger_signal = True
    else:
        danger_signal = False

    return danger_signal

def ROI(image):
    h, w = image.shape[:2]

    mask = np.zeros_like(image)

    rect = np.array([
        [w * 0.02, h * 0.97],
        [w * 0.95, h * 0.97],
        [w * 0.75, h * 0.4],
        [w * 0.4, h * 0.4]
    ], dtype = np.int32)

    cv2.fillPoly(mask, [rect], (255, 255, 255))

    masked_img = cv2.bitwise_and(image, mask)

    return masked_img