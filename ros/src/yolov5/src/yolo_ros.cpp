#include "yolov5_trt.h"
#include <opencv2/opencv.hpp>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>

yolov5TRT *yolotrt_ptr;
ros::Publisher pub;

void imageCallback(const sensor_msgs::ImagePtr& data){
    ROS_INFO("Callback");
    cv_bridge::CvImagePtr cv_ptr;
    try{
        cv_ptr = cv_bridge::toCvCopy(data, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e){
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv::Mat image = cv_ptr->image;
    cv::Mat out_image = yolotrt_ptr->infer(image);
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", out_image).toImageMsg();
    pub.publish(*msg);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "yolov5");
    ros::NodeHandle nh;

    pub = nh.advertise<sensor_msgs::Image>("/output", 10);

    ros::Subscriber sub = nh.subscribe("/image_raw", 10, imageCallback);

    yolov5TRT yolotrt_("/home/ros/src/yolov5/model/engine_fp16.engine");
    yolotrt_ptr = &yolotrt_;
    ros::spin();

}