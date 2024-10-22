import rclpy as rp
from rclpy.node import Node

from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import Twist

import signal
import sys

class Manipulate(Node):
    def __init__(self):
        super().__init__('Manipulate')

        self.center_subscriber = self.create_subscription(
            Float32,
            '/Point',
            self.center_callback,
            10
        )

        self.rad_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.curve_signal = self.create_subscription(
            Bool,
            '/curve_signal',
            self.curve_callback,
            10
        )

        self.Kp = 0.05
        self.Ki = 0.0
        self.Kd = 0.0

        self.prev_error = 0.
        self.integral = 0.01
        self.prev_time = self.get_clock().now()
        self.start_time = self.get_clock().now()
        self.signal = False

        signal.signal(signal.SIGINT, self.signal_handler)

    def curve_callback(self, msg):
        self.signal = msg.data


    def PID_Controll(self, error):
        current_time = self.get_clock().now()

        dt = (current_time - self.prev_time).nanoseconds / 1e9

        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        control_signal = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        self.prev_error = error
        self.prev_time = current_time
        
        return control_signal

    def center_callback(self, msg):
        rad = msg.data
        signal = self.PID_Controll(rad)

        twist_msg = Twist()
        twist_msg.linear.x = 0.5
        twist_msg.angular.z = signal

        if self.signal:
            twist_msg.linear.x = 1.
            self.Kp = 0.01
            self.Ki = 0.03
            self.get_logger().info('curving!')
        else:
            if (self.get_clock().now() - self.start_time).nanoseconds / 1e9 > 20:
                twist_msg.linear.x = 2.
                self.Kp = 0.005
                self.get_logger().info('not curving')

        self.rad_publisher.publish(twist_msg)

    def signal_handler(self, sig, frame):
        stop_msg = Twist()
        stop_msg.linear.x = 0.
        stop_msg.angular.z = 0.

        self.rad_publisher.publish(stop_msg)

        rp.shutdown()
        sys.exit(0)

def main(args=None):
    rp.init(args=args)
    node = Manipulate()
    rp.spin(node)
    node.destroy_node()
    rp.shutdown()

if __name__ == '__main__':
    main()