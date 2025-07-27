#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <fstream>
#include <iomanip>
#include <string>

class OdomLoggerNode : public rclcpp::Node
{
public:
    OdomLoggerNode() : Node("odom_logger_node")
    {
        // Declare and get parameters
        this->declare_parameter<std::string>("wheel_odom_topic", "/wheel_odom");
        this->declare_parameter<std::string>("lidar_odom_topic", "/lidar_odom");
        this->declare_parameter<std::string>("wheel_odom_file", "wheel_odom.txt");
        this->declare_parameter<std::string>("lidar_odom_file", "lidar_odom.txt");
        
        std::string wheel_topic = this->get_parameter("wheel_odom_topic").as_string();
        std::string lidar_topic = this->get_parameter("lidar_odom_topic").as_string();
        std::string wheel_file = this->get_parameter("wheel_odom_file").as_string();
        std::string lidar_file = this->get_parameter("lidar_odom_file").as_string();
        
        // Open output files
        wheel_file_.open(wheel_file);
        lidar_file_.open(lidar_file);
        
        if (!wheel_file_.is_open() || !lidar_file_.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open output files!");
            rclcpp::shutdown();
            return;
        }
        
        // Write headers
        wheel_file_ << "# timestamp x y theta_in_radians\n";
        lidar_file_ << "# timestamp x y theta_in_radians\n";
        
        // Set precision for floating point output
        wheel_file_ << std::fixed << std::setprecision(6);
        lidar_file_ << std::fixed << std::setprecision(6);
        
        // Create message filter subscribers
        wheel_sub_.subscribe(this, wheel_topic);
        lidar_sub_.subscribe(this, lidar_topic);
        
        // Create synchronizer with ExactTime policy
        sync_ = std::make_shared<Synchronizer>(MySyncPolicy(10), wheel_sub_, lidar_sub_);
        sync_->registerCallback(std::bind(&OdomLoggerNode::syncCallback, this, 
                                         std::placeholders::_1, std::placeholders::_2));
        
        RCLCPP_INFO(this->get_logger(), "OdomLoggerNode initialized");
        RCLCPP_INFO(this->get_logger(), "Subscribing to wheel odometry: %s", wheel_topic.c_str());
        RCLCPP_INFO(this->get_logger(), "Subscribing to lidar odometry: %s", lidar_topic.c_str());
        RCLCPP_INFO(this->get_logger(), "Writing wheel data to: %s", wheel_file.c_str());
        RCLCPP_INFO(this->get_logger(), "Writing lidar data to: %s", lidar_file.c_str());
        RCLCPP_INFO(this->get_logger(), "Using ExactTime synchronization policy");
    }
    
    ~OdomLoggerNode()
    {
        if (wheel_file_.is_open()) {
            wheel_file_.close();
        }
        if (lidar_file_.is_open()) {
            lidar_file_.close();
        }
        RCLCPP_INFO(this->get_logger(), "OdomLoggerNode shutting down, files closed");
    }
    
private:
    // Type definitions for message filters
    typedef message_filters::sync_policies::ExactTime<nav_msgs::msg::Odometry, 
                                                      nav_msgs::msg::Odometry> MySyncPolicy;
    typedef message_filters::Synchronizer<MySyncPolicy> Synchronizer;
    
    // Message filter subscribers
    message_filters::Subscriber<nav_msgs::msg::Odometry> wheel_sub_;
    message_filters::Subscriber<nav_msgs::msg::Odometry> lidar_sub_;
    
    // Synchronizer
    std::shared_ptr<Synchronizer> sync_;
    
    // Output file streams
    std::ofstream wheel_file_;
    std::ofstream lidar_file_;
    
    // Message counter
    size_t msg_count_ = 0;
    
    void syncCallback(const nav_msgs::msg::Odometry::ConstSharedPtr& wheel_msg,
                      const nav_msgs::msg::Odometry::ConstSharedPtr& lidar_msg)
    {
        // Process wheel odometry
        processAndWriteOdom(wheel_msg, wheel_file_, "wheel");
        
        // Process lidar odometry
        processAndWriteOdom(lidar_msg, lidar_file_, "lidar");
        
        msg_count_++;
        
        // Log progress every 100 messages
        if (msg_count_ % 100 == 0) {
            RCLCPP_INFO(this->get_logger(), "Processed %zu synchronized message pairs", msg_count_);
        }
    }
    
    void processAndWriteOdom(const nav_msgs::msg::Odometry::ConstSharedPtr& msg,
                            std::ofstream& file,
                            const std::string& odom_type)
    {
        // Extract timestamp
        double timestamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
        
        // Extract position
        double x = msg->pose.pose.position.x;
        double y = msg->pose.pose.position.y;
        
        // Convert quaternion to yaw angle
        tf2::Quaternion q(
            msg->pose.pose.orientation.x,
            msg->pose.pose.orientation.y,
            msg->pose.pose.orientation.z,
            msg->pose.pose.orientation.w
        );
        
        double roll, pitch, yaw;
        tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
        
        // Write to file
        file << timestamp << " " << x << " " << y << " " << yaw << "\n";
        
        // Ensure data is written immediately
        file.flush();
        
        // Debug output for first few messages
        if (msg_count_ < 5) {
            RCLCPP_DEBUG(this->get_logger(), 
                        "%s odom: t=%.3f, x=%.3f, y=%.3f, yaw=%.3f rad (%.1f deg)",
                        odom_type.c_str(), timestamp, x, y, yaw, yaw * 180.0 / M_PI);
        }
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<OdomLoggerNode>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("odom_logger"), "Exception: %s", e.what());
    }
    
    rclcpp::shutdown();
    return 0;
}