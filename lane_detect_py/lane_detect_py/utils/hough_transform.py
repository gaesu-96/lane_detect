import cv2
import numpy as np

def hough(image, prev_center_x = None):
    masked_image = ROI(image)
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)

    edge = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edge, 1, np.pi / 180, 110, minLineLength = 40, maxLineGap = 10)
    right_points = []
    left_points = []
    horizon_points = []

    right_center_x = None
    left_center_x = None

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            if (-40 < angle < 40 or -140 > angle):
                continue

            if (y1 > y2):
                if (x1 >= image.shape[1] // 2):
                    right_points.append((x1, y1))
                    right_points.append((x2, y2))
                elif (x1 < image.shape[1] // 2):
                    left_points.append((x1, y1))
                    left_points.append((x2, y2))
            else:
                if (x2 >= image.shape[1] // 2):
                    right_points.append((x1, y1))
                    right_points.append((x2, y2))
                elif (x2 < image.shape[1] // 2):
                    left_points.append((x1, y1))
                    left_points.append((x2, y2))

    right_points = np.array(right_points)
    left_points = np.array(left_points)

    if len(right_points) > 0:
        right_center_x = fit_lines(right_points, image)
        right_curve = False
    else:
        right_curve = True
    if len(left_points) > 0:
        left_center_x = fit_lines(left_points, image)
        left_curve = False
    else: left_curve = True

    if (right_center_x is not None and left_center_x is not None):
        center_x = (right_center_x + left_center_x) // 2
    else:
        center_x = prev_center_x

    cv2.circle(image, (center_x, image.shape[0] // 2 + 50), 5, (0, 255, 0), 2)
    cv2.line(image, (image.shape[1] // 2, image.shape[0] - 1), (image.shape[1] // 2, 0), (0, 0, 255), 1)

    if center_x is None:
        center_x = image.shape[1] // 2

    rad = get_radian(image, center_x, image.shape[0] // 2 + 50)

    return image, rad, center_x, right_curve, left_curve

def get_radian(image, x, y):
    origin_center_x, origin_center_y = image.shape[1] // 2, image.shape[0] - 1

    rad = np.arctan2((x - origin_center_x), (y - origin_center_y))

    return rad

def fit_lines(points, image):
    h, w = image.shape[:2]
    y_target = h // 2 + 10

    points = np.array(points)

    fit_line = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)

    vx, vy, x, y = fit_line[0], fit_line[1], fit_line[2], fit_line[3]

    if vy != 0:
        center_x = int(x + (y_target - y) * (vx / (vy + 0e9)))
    else:
        center_x = int(image.shape[1] // 2)

    x1 = int(x - 1000 * vx)
    y1 = int(y - 1000 * vy)
    x2 = int(x + 1000 * vx)
    y2 = int(y + 1000 * vy)

    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)

    return center_x

def ROI(image):
    h, w = image.shape[:2]

    mask = np.zeros_like(image)

    rect = np.array([
        [w * 0.02, h * 0.85],
        [w * 0.87, h * 0.85],
        [w * 0.7, h * 0.4],
        [w * 0.4, h * 0.4]
    ], dtype = np.int32)

    cv2.fillPoly(mask, [rect], (255, 255, 255))

    masked_img = cv2.bitwise_and(image, mask)

    cv2.imshow('test', masked_img)
    cv2.waitKey(1)

    return masked_img

def find_center():
    return None