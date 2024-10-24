import rclpy as rp
from rclpy.node import Node
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge
from lane_detect_py.utils.find_lane_dl import find_lane

from std_msgs.msg import Float32
from std_msgs.msg import Bool

import signal
import sys

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

        self.fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.out = cv2.VideoWriter('DL_output1.avi', self.fourcc, 20.0, (1280, 720))

    def subs_callback(self, msg):
        image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
        output, self.center, self.rad = find_lane(image, self.center)

        cv2.imshow('Image', output)
        cv2.waitKey(1)
        self.out.write(output)
        self.center_publish()

    def center_publish(self):
        msg = Float32()

        msg.data = self.rad * -1

        self.point_publisher.publish(msg)
        self.get_logger().info(f'center coordinate: {msg.data}')

    def destroy_node(self):
        self.out.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rp.init(args=args)
    image_subscriber = ImageSubscriber()

    rp.spin(image_subscriber)
    image_subscriber.destroy_node()
    rp.shutdown()

if __name__ == '__main__':
    main()