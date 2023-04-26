#include <iostream>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <vector>
#include "buffers.h"
#include "common.h"
#include <algorithm>

#include <sys/time.h>
#include <chrono>

#include "boundingBox.h"

#include <opencv2/opencv.hpp>
#include <fstream>


// Namespaces.
using namespace cv;
using namespace std;

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger {
 public:
  explicit Logger(Severity severity = Severity::kWARNING)
      : reportable_severity(severity) {}

  void log(Severity severity, const char* msg) noexcept override {
    // suppress messages with severity enum value greater than the reportable
    if (severity > reportable_severity) return;

    switch (severity) {
      case Severity::kINTERNAL_ERROR:
        std::cerr << "INTERNAL_ERROR: ";
        break;
      case Severity::kERROR:
        std::cerr << "ERROR: ";
        break;
      case Severity::kWARNING:
        std::cerr << "WARNING: ";
        break;
      case Severity::kINFO:
        std::cerr << "INFO: ";
        break;
      default:
        std::cerr << "UNKNOWN: ";
        break;
    }
    std::cerr << msg << std::endl;
  }

  Severity reportable_severity;
};

class yolov5TRT{
    public:
        yolov5TRT(string model_path) : model_path_(model_path){

          load_engine();
          
          this->mInputDims.nbDims = this->inputShape.size();
          this->mInputDims.d[0] = this->inputShape[0];
          this->mInputDims.d[1] = this->inputShape[1];
          this->mInputDims.d[2] = this->inputShape[2];
          this->mInputDims.d[3] = this->inputShape[3];

        };
        void preprocess(Mat image, float *data);
        Mat postprocess(Mat input_image, float *outputs, const vector<string> &class_name);
        void load_engine();
        void draw_label(const std::vector<BoundingBox> &bboxes, cv::Mat &testImg);
        Mat infer(Mat image);
    
    private:

        // create TRT model from engine
        std::shared_ptr<nvinfer1::ICudaEngine> EngineToTRTModel(const std::string &engine_file);

        // apply nms to boxes
        void NMSBoxes(const std::vector<cv::Rect>& bboxes, const std::vector<float>& scores, const float score_threshold, const float nms_threshold, std::vector<int>& indices);

        // Constants.
        const float INPUT_WIDTH = 640.0;
        const float INPUT_HEIGHT = 640.0;
        const float SCORE_THRESHOLD = 0.4;
        const float NMS_THRESHOLD = 0.4;
        const float CONFIDENCE_THRESHOLD = 0.45;
        
        // Text parameters.
        const float FONT_SCALE = 1;
        const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
        const int THICKNESS = 0.1;
        
        // Colors.
        Scalar BLACK = Scalar(0,0,0);
        Scalar BLUE = Scalar(255, 178, 50);
        Scalar YELLOW = Scalar(0, 255, 255);
        Scalar RED = Scalar(0,0,255);

        const cv::Scalar mean = cv::Scalar(0.406, 0.456, 0.485);
        const cv::Scalar scale = cv::Scalar(0.225, 0.224, 0.229);
        std::string model_path_;

        std::shared_ptr<nvinfer1::ICudaEngine> engine_;
        nvinfer1::IExecutionContext* context_;
        Logger g_logger_;
        std::vector<int> inputShape = std::vector<int>{1, 3, 640, 640};
        nvinfer1::Dims mInputDims;

        std::vector<int> image_rows;
        std::vector<int> image_cols;
        std::vector<int> image_pad_rows;
        std::vector<int> image_pad_cols;

        std::vector<cv::Scalar> colors{
          cv::Scalar(56, 56, 255),
          cv::Scalar(151, 157, 255),
          cv::Scalar(31, 112, 255),
          cv::Scalar(29, 178, 255),
          cv::Scalar(49, 210, 207) 
        };
        

        std::vector<std::string> coco_class_names = {
            "person"
            ,"bicycle"
            ,"car"
            ,"motorbike"
            ,"aeroplane"
            ,"bus"
            ,"train"
            ,"truck"
            ,"boat"
            ,"traffic light"
            ,"fire hydrant"
            ,"stop sign"
            ,"parking meter"
            ,"bench"
            ,"bird"
            ,"cat"
            ,"dog"
            ,"horse"
            ,"sheep"
            ,"cow"
            ,"elephant"
            ,"bear"
            ,"zebra"
            ,"giraffe"
            ,"backpack"
            ,"umbrella"
            ,"handbag"
            ,"tie"
            ,"suitcase"
            ,"frisbee"
            ,"skis"
            ,"snowboard"
            ,"sports ball"
            ,"kite"
            ,"baseball bat"
            ,"baseball glove"
            ,"skateboard"
            ,"surfboard"
            ,"tennis racket"
            ,"bottle"
            ,"wine glass"
            ,"cup"
            ,"fork"
            ,"knife"
            ,"spoon"
            ,"bowl"
            ,"banana"
            ,"apple"
            ,"sandwich"
            ,"orange"
            ,"broccoli"
            ,"carrot"
            ,"hot dog"
            ,"pizza"
            ,"donut"
            ,"cake"
            ,"chair"
            ,"sofa"
            ,"pottedplant"
            ,"bed"
            ,"diningtable"
            ,"toilet"
            ,"tvmonitor"
            ,"laptop"
            ,"mouse"
            ,"remote"
            ,"keyboard"
            ,"cell phone"
            ,"microwave"
            ,"oven"
            ,"toaster"
            ,"sink"
            ,"refrigerator"
            ,"book"
            ,"clock"
            ,"vase"
            ,"scissors"
            ,"teddy bear"
            ,"hair drier"
            ,"toothbrush"



        };

};







