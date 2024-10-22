import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge
from lane_detect_py.utils.hough_transform import hough

from std_msgs.msg import Float32
from std_msgs.msg import Bool
class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('ImabeSubscriber')

        self.subscriber = self.create_subscription(
            Image,
            'camera/image_raw',
            self.subs_callback,
            10
        )

        self.point_publisher = self.create_publisher(
            Float32,
            '/Point',
            10
        )

        self.curve_signal = self.create_publisher(
            Bool,
            '/curve_signal',
            10
        )

        self.cv_bridge = CvBridge()
        self.center = None
        self.right_cnt = 0
        self.left_cnt = 0

    def subs_callback(self, msg):
        image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')

        line_image, self.rad, self.center, self.right_curve, self.left_curve = hough(image, self.center)
        if (self.right_curve):
            self.right_cnt += 1
        else: self.right_cnt = 0

        if (self.left_curve):
            self.left_cnt += 1
        else: self.left_cnt = 0

        signal = Bool()

        if self.right_cnt > 10 or self.left_cnt > 10:
            signal.data = True
            self.curve_signal.publish(signal)
        else:
            signal.data = False
            self.curve_signal.publish(signal)


        cv2.imshow('Image', line_image)
        cv2.waitKey(1)
        self.center_publish()

    def center_publish(self):
        msg = Float32()

        msg.data = self.rad * -1

        self.point_publisher.publish(msg)
        self.get_logger().info(f'center coordinate: {msg.data}')


def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()

    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()