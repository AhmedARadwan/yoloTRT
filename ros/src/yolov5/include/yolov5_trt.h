#include <iostream>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include "NvInfer.h"
#include <cuda_runtime_api.h>



class yolov5TRT{
    public:
        yolov5TRT(std::string model_path) : model_path_(model_path){

        };
        vector<Mat> preprocess(Mat &input_image, Net &net);
        Mat postprocess(Mat &input_image, vector<Mat> &outputs, const vector<string> &class_name);
        void load_engine();
        void yolov5TRT::draw_label(Mat& input_image, string label, int left, int top);
    
    private:
        // Constants.
        const float INPUT_WIDTH = 640.0;
        const float INPUT_HEIGHT = 640.0;
        const float SCORE_THRESHOLD = 0.5;
        const float NMS_THRESHOLD = 0.45;
        const float CONFIDENCE_THRESHOLD = 0.45;
        
        // Text parameters.
        const float FONT_SCALE = 0.7;
        const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
        const int THICKNESS = 1;
        
        // Colors.
        Scalar BLACK = Scalar(0,0,0);
        Scalar BLUE = Scalar(255, 178, 50);
        Scalar YELLOW = Scalar(0, 255, 255);
        Scalar RED = Scalar(0,0,255);
        std::string model_path_;

}





