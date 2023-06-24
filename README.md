# yoloTRTv5

## Installation

Assuming you have docker, docker-compose, nvidia container runtime installed on host.

Download TensorRT 7.2.2 from https://developer.nvidia.com/tensorrt (Filename should be ``` TensorRT-7.2.2.3.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.0.tar.gz``` )   

  

### Building and running the container:
```sh
git clone https://github.com/AhmedARadwan/yoloTRT.git
cd yoloTRT/
cp pathto/TensorRT-7.2.2.3.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.0.tar.gz .
docker-compose up --build
```
  



### Creating TensorRT engine:
In another terminal
```sh
docker exec -it yoloTRT_m bash
cd src/yolov5/model/
./generate_trt.sh
```

### Running the inference node: 
In another terminal
```sh
docker exec -it yoloTRT_m bash
catkin_make
source devel/setup.bash
roslaunch yolov5 yolov5.launch
```
