#include "yolov5_trt.h"

/**
 * Function: Loads the pre-trained TensorRT engine and creates the execution context

 * @param None
 * @return None
 */
void yolov5TRT::load_engine(){
    
    // loading model
    engine_ = EngineToTRTModel(model_path_);

    // create execution context from the engine
    context_ = engine_->createExecutionContext();

}

/**
 * Function: Loads the pre-trained TensorRT engine from a file         

 * @param engine_file: path to the TensorRT engine file
 * @return engine_ptr1: shared pointer to the TensorRT engine object (nvinfer1::ICudaEngine)
                        The shared pointer is a smart pointer that automatically frees the memory of the engine 
                        when the engine is no longer in use.
 */
std::shared_ptr<nvinfer1::ICudaEngine> yolov5TRT::EngineToTRTModel(const std::string &engine_file)  {
    
    int verbosity = static_cast<int>(nvinfer1::ILogger::Severity::kWARNING);
    std::stringstream gieModelStream;
    gieModelStream.seekg(0, gieModelStream.beg);

    std::ifstream cache(engine_file); 
    gieModelStream << cache.rdbuf();
    cache.close();
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(g_logger_); 

    if (runtime == nullptr) {
        std::string msg("failed to build runtime parser");
        g_logger_.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        exit(EXIT_FAILURE);
    }
    gieModelStream.seekg(0, std::ios::end);
    const int modelSize = gieModelStream.tellg();

    gieModelStream.seekg(0, std::ios::beg);
    void* modelMem = malloc(modelSize); 
    gieModelStream.read((char*)modelMem, modelSize);

    std::cout << "                                                                  "<< std::endl;
    std::cout << "------------------------------------------------------------------"<< std::endl;
    std::cout << ">>>>                                                          >>>>"<< std::endl;
    std::cout << "                                                                  "<< std::endl;
    std::cout << "Input filename:   " << engine_file << std::endl;
    std::cout << "                                                                  "<< std::endl;
    std::cout << ">>>>                                                          >>>>"<< std::endl;
    std::cout << "------------------------------------------------------------------"<< std::endl;
    std::cout << "                                                                  "<< std::endl;

    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(modelMem, modelSize, NULL); 
    if (engine == nullptr) {
        std::string msg("failed to build engine parser");
        g_logger_.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        exit(EXIT_FAILURE);
    }
    std::shared_ptr<nvinfer1::ICudaEngine> engine_ptr1(engine, [](nvinfer1::ICudaEngine* engine){engine->destroy();});
    return engine_ptr1;
}

/**
 * Function to draw a label on top of a bounding box in an image
 * 
 * @param input_image: input image on which label needs to be drawn
 * @param label: label text to be displayed
 * @param left: left coordinate of the bounding box
 * @param top: top coordinate of the bounding box
 * 
 * @return None
 */

void yolov5TRT::draw_label(const std::vector<BoundingBox> &bboxes, cv::Mat &testImg){
    int H = testImg.rows;
    int W = testImg.cols;

    for (size_t k = 0; k < bboxes.size(); k++)
    {
        if (bboxes[k].cls == -1)
        {
            break;
        }

        int x = (bboxes[k].x / 640) * W;
        int y = (bboxes[k].y / 640) * H;
        int w = (bboxes[k].w / 640) * W;
        int h = (bboxes[k].h / 640) * H;

        auto box_rect = cv::Rect(x, y, w, h);
        auto color = colors[bboxes[k].cls % colors.size()];
        
        cv::rectangle(testImg, box_rect, color, 2);
        cv::putText(testImg, coco_class_names[bboxes[k].cls], cv::Point(x, y), cv::FONT_HERSHEY_DUPLEX, 0.8, color, 1);
        try
        {
            cv::Mat roi = testImg(box_rect);

            cv::Mat color(box_rect.size(), CV_8UC3, colors[bboxes[k].cls % colors.size()]);
            double alpha = 0.3;
            cv::addWeighted(color, alpha, roi, 1.0 - alpha, 0.0, roi);
            cv::rectangle(testImg, box_rect, colors[bboxes[k].cls], 2);
        }
        catch (int exp)
        {
            continue;
        }
    }
}

/**
 * Function: preprocess the input image into buffer

 *           Converts the image from BGR to RGB format, resizes the image to desired dimensions
             and normalizes the pixel values by dividing each pixel value by 255.0.
 *           The input image is stored in a 1D data array in the order of batch, channel, height, width

 * @param None
 * @return None
 */
void yolov5TRT::preprocess(Mat image, float *buffer)
{
    const int inputB = this->mInputDims.d[0];
    const int inputC = this->mInputDims.d[1];
    const int inputH = this->mInputDims.d[2];
    const int inputW = this->mInputDims.d[3];

    std::vector<std::vector<cv::Mat>> input_channels;
    for (int b = 0; b < inputB; ++b)
    {
        input_channels.push_back(std::vector<cv::Mat> {static_cast<size_t>(inputC)});
    }

    this->image_rows.clear();
    this->image_cols.clear();
    this->image_pad_rows.clear();
    this->image_pad_cols.clear();

    cv::Mat rgb_img;

    // Convert BGR to RGB
    cv::cvtColor(image, rgb_img, cv::COLOR_BGR2RGB);

    auto scaleSize = cv::Size(640, 640);
    cv::Mat resized;
    cv::resize(rgb_img, resized, scaleSize, 0, 0, cv::INTER_LINEAR);

    // Each element in batch share the same image matrix
    for (int b = 0; b < inputB; ++b)
    {
        cv::split(resized, input_channels[b]);
    }

    int volBatch = inputC * inputH * inputW;
    int volChannel = inputH * inputW;
    int volW = inputW;

    int d_batch_pos = 0;
    for (int b = 0; b < inputB; b++)
    {
        int d_c_pos = d_batch_pos;
        for (int c = 0; c < inputC; c++)
        {
            int s_h_pos = 0;
            int d_h_pos = d_c_pos;
            for (int h = 0; h < inputH; h++)
            {
                int s_pos = s_h_pos;
                int d_pos = d_h_pos;
                for (int w = 0; w < inputW; w++)
                {
                    buffer[d_pos] = (float)input_channels[b][c].data[s_pos] / 255.0f;
                    ++s_pos;
                    ++d_pos;
                }
                s_h_pos += volW;
                d_h_pos += volW;
            }
            d_c_pos += volChannel;
        }
        d_batch_pos += volBatch;
    }
}


Mat yolov5TRT::infer(Mat image)
{
    
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(engine_);

    auto startTime = std::chrono::high_resolution_clock::now();

    float* hostInputBuffer = static_cast<float*>(buffers.getHostBuffer("images"));
    preprocess(image, hostInputBuffer);
    auto endTime = std::chrono::high_resolution_clock::now();
    double preprocessDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count()/1000000.0;


    startTime = std::chrono::high_resolution_clock::now();
    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();
    bool status = context_->executeV2(buffers.getDeviceBindings().data());
    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();
    endTime = std::chrono::high_resolution_clock::now();
    double inferenceDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count()/1000000.0;

    // postprocessing
    startTime = std::chrono::high_resolution_clock::now();
    Mat out_image = postprocess(image, static_cast<float*>(buffers.getHostBuffer("output0")), coco_class_names);
    endTime = std::chrono::high_resolution_clock::now();
    double postprocessDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count()/1000000.0;
    sample::gLogInfo << "PreProcess Time: " << preprocessDuration << " ms"<< std::endl;
    sample::gLogInfo << "inferenceDuration Time: " << inferenceDuration << " ms"<< std::endl;
    sample::gLogInfo << "postprocessDuration Time: " << postprocessDuration << " ms"<< std::endl;
    return out_image;
}

Mat yolov5TRT::postprocess(Mat input_image, float *outputs, const vector<string> &class_name)
{
    // Initialize vectors to hold respective outputs while unwrapping     detections.
    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;
    // Resizing factor.
    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;
    float *data = outputs;

    int i = 0;
    int nc = 80;
    while (i < 25200)
    {
        // Box
        int k = i * 85;
        float object_conf = data[k + 4];

        if (object_conf < 0.5)
        {
            i++;
            continue;
        }

        // (center x, center y, width, height) to (x, y, w, h)
        float x = (data[k] - data[k + 2] / 2);
        float y = (data[k + 1] - data[k + 3] / 2);
        float width = data[k + 2];
        float height = data[k + 3];

        // Classes
        float class_conf = data[k + 5];
        int classId = 0;

        for (int j = 1; j < nc; j++)
        {
            if (class_conf < data[k + 5 + j])
            {
                classId = j;
                class_conf = data[k + 5 + j];
            }
        }
        
        i++;
        
        class_conf *= object_conf;

        class_ids.push_back(classId);
        confidences.push_back(class_conf);
        boxes.emplace_back(cv::Rect((int)x, (int)y, (int)width, (int)height));

    }
    
    // Perform Non-Maximum Suppression and draw predictions.
    vector<int> indices;
    NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
    cout << "length of boxes: " << boxes.size() << endl;


    std::vector<BoundingBox> bboxes;
    bboxes.reserve(indices.size());
    for (size_t i = 0; i < indices.size(); ++i)
    {
        BoundingBox box;
        // (x, y, w, h) to (x1, y1, x2, y2)
        box.x = boxes[indices[i]].x;
        box.y = boxes[indices[i]].y;
        box.w = boxes[indices[i]].width;
        box.h = boxes[indices[i]].height;
        box.score = confidences[indices[i]];
        box.cls = class_ids[indices[i]];

        bboxes.emplace_back(box);
    }

    auto scaleSize = cv::Size(640, 640);
    cv::Mat resized;
    cv::resize(input_image, resized, scaleSize, 0, 0, cv::INTER_LINEAR);

    draw_label(bboxes, resized);
    
    return resized;
}

// Compute intersection over union (IOU) between two bounding boxes
float iou(const cv::Rect& boxA, const cv::Rect& boxB)
{
    cv::Rect intersect = boxA & boxB;
    float intersect_area = intersect.area();
    float union_area = boxA.area() + boxB.area() - intersect_area;
    return intersect_area / union_area;
}

// Apply NMS to the bounding boxes
void yolov5TRT::NMSBoxes(const std::vector<cv::Rect>& bboxes, const std::vector<float>& scores, const float score_threshold, const float nms_threshold, std::vector<int>& indices)
{
    // Keep track of the selected bounding boxes
    indices.clear();

    // Get the number of bounding boxes
    int n = bboxes.size();

    // Sort the bounding boxes by score in descending order
    std::vector<int> sorted_indices(n);
    for (int i = 0; i < n; ++i)
    {
        sorted_indices[i] = i;
    }
    std::sort(sorted_indices.begin(), sorted_indices.end(), [&scores](int i1, int i2) { return scores[i1] > scores[i2]; });

    // Loop through the sorted bounding boxes and remove low-scoring boxes
    for (int i = 0; i < n; ++i)
    {
        int idx = sorted_indices[i];
        if (scores[idx] < score_threshold)
        {
            break;
        }
        indices.push_back(idx);
        for (int j = i + 1; j < n; ++j)
        {
            int next = sorted_indices[j];
            if (iou(bboxes[idx], bboxes[next]) > nms_threshold)
            {
                sorted_indices.erase(sorted_indices.begin() + j);
                --j;
                --n;
            }
        }
    }
}