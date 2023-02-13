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
void yolov5TRT::draw_label(Mat& input_image, string label, int left, int top)
{
    // Display the label at the top of the bounding box.
    int baseLine;
    Size label_size = getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = max(top, label_size.height);
    // Top left corner.
    Point tlc = Point(left, top);
    // Bottom right corner.
    Point brc = Point(left + label_size.width, top + label_size.height + baseLine);
    // Draw white rectangle.
    rectangle(input_image, tlc, brc, BLACK, FILLED);
    // Put the label on the black rectangle.
    putText(input_image, label, Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
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
    const int dimensions = 85;
    // 25200 for default size 640.
    const int rows = 25200;
    // Iterate through 25200 detections.
    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4];
        // Discard bad detections and continue.
        if (confidence >= CONFIDENCE_THRESHOLD)
        {
            float * classes_scores = data + 5;
            // Create a 1x85 Mat and store class scores of 80 classes.
            Mat scores(1, class_name.size(), CV_32FC1, classes_scores);
            // Perform minMaxLoc and acquire the index of best class  score.
            Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            // Continue if the class score is above the threshold.
            if (max_class_score > SCORE_THRESHOLD)
            {
                // Store class ID and confidence in the pre-defined respective vectors.
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);
                // Center.
                float cx = data[0];
                float cy = data[1];
                // Box dimension.
                float w = data[2];
                float h = data[3];
                // Bounding box coordinates.
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                // Store good detections in the boxes vector.
                boxes.push_back(Rect(left, top, width, height));
            }
        }
        // Jump to the next row.
        data += 85;
    }
    cout << "length of boxes: " << boxes.size() << endl;
    // Perform Non-Maximum Suppression and draw predictions.
    vector<int> indices;
    NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
    for (int i = 0; i < indices.size(); i++)
    {
        int idx = indices[i];
        Rect box = boxes[i];
        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;
        // Draw bounding box.
        rectangle(input_image, Point(left, top), Point(left + width, top + height), BLUE, 3*THICKNESS);
        // Get the label for the class name and its confidence.
        string label = format("%.2f", confidences[idx]);
        label = class_name[class_ids[idx]] + ":" + label;
        // Draw class labels.
        draw_label(input_image, label, left, top);
    }
    return input_image;
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