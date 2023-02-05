cmake_minimum_required(VERSION 3.0.2)
project(yolov5)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
)


find_package(CUDA REQUIRED QUIET)
include_directories(${CUDA_INCLUDE_DIRS})
if (CUDA_VERSION VERSION_LESS "11.0")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11 -Wno-deprecated-gpu-targets)
else ()
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++14 -Wno-deprecated-gpu-targets)
endif ()
set(TENSORRT_ROOT /app/TensorRT-7.2.2.3)
message("Using TENSORRT_ROOT: ${TENSORRT_ROOT}")


catkin_package()





include_directories(
  include
  ${TENSORRT_ROOT}/include/
  ${catkin_INCLUDE_DIRS}
)

link_directories(
    ${TENSORRT_ROOT}/lib
)

list(APPEND LINK_LIBRARY
            ${CUDA_cusparse_LIBRARY}
            ${CUDA_LIBRARIES}
            ${CUDA_CUBLAS_LIBRARIES}
            ${CUDA_curand_LIBRARY}            
            ${CUDA_cusparse_LIBRARY}
            ${TENSORRT_ROOT}/lib
            nvinfer
            nvonnxparser
            yaml-cpp
            )


add_executable(yolo_ros src/yolo_ros.cpp src/yolov5_trt.cpp)
target_link_libraries(yolo_ros ${LINK_LIBRARY} ${catkin_LIBRARIES})

