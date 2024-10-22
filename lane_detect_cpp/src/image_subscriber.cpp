// #include "rclcpp/rclcpp.hpp"
// #include "sensor_msgs/msg/image.hpp"
// #include "image_transport/image_transport.hpp"

// class ImageSubscriber : public rclcpp::Node, public std::enable_shared_from_this<ImageSubscriber> {
// public:
//     ImageSubscriber() : Node("image_subscriber") {
//         // shared_from_this() 호출을 지연시키기 위해 별도의 초기화 메서드 사용
//         auto self = shared_from_this(); // shared_from_this() 호출
//         image_transport::ImageTransport it(self); // 이곳에서 사용할 수 있도록
//         sub_ = it.subscribe("/camera/image_raw", 10, &ImageSubscriber::imageCallback, this);
//     }

// private:
//     void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
//         RCLCPP_INFO(this->get_logger(), "running!");
//     }

//     image_transport::Subscriber sub_;
// };

// int main(int argc, char **argv) {
//     rclcpp::init(argc, argv);
//     auto node = std::make_shared<ImageSubscriber>();
//     rclcpp::spin(node);
//     rclcpp::shutdown();
//     return 0;
// }
