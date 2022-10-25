# Camera & Radar feature-level sensor fusion for object detection 

https://user-images.githubusercontent.com/25930661/195581012-5cb6a223-7294-4fa6-afb2-7393345831ee.mp4





## Generating samples
  Please download and preprocess nuscenes dataset according to [this](https://github.com/mrnabati/CenterFusion#dataset-preparation)
  assuming your directory structure is like this :
  
  ```
  ${CF_ROOT}
  `-- data
    `-- nuscenes
        |-- annotations_6sweeps
        |-- maps
        |-- samples
        |   |-- CAM_BACK
        |   |   | -- xxx.jpg
        |   |   ` -- ...
        |   |-- CAM_BACK_LEFT
        |   |-- CAM_BACK_RIGHT
        |   |-- CAM_FRONT
        |   |-- CAM_FRONT_LEFT
        |   |-- CAM_FRONT_RIGHT
        |   |-- RADAR_BACK_LEFT
        |   |   | -- xxx.pcd
        |   |   ` -- ...
        |   |-- RADAR_BACK_RIGHT
        |   |-- RADAR_FRON
        |   |-- RADAR_FRONT_LEFT
        |   `-- RADAR_FRONT_RIGHT
        |-- sweeps
        |-- v1.0-mini
        |-- v1.0-test
        `-- v1.0-trainval
  ```
  move the annotation file `data/val_top1000.json` to `data/nuscenes/annotations_6sweeps` .
  run the following commands :
  ```
  cd tools/CenterFusion
  sh experiments/create_data.sh
  ```
  Then you should have the generated datas in `data/predata` :

    - images, contains 1000 frame input images for trt engines, each has its shape (3, 448, 800)
    - calibs, contains 1000 frame camera intrinsics, each has its shape (3,4)
    - pc_3ds, contains 1000 frame radar points, each has its shape  (5,1000), each row stands for [loc_x, loc_y, loc_z, velo_x, velo_y]
    - data_num.bin, wich shape (1000,), records valid point nums for each radar frame 
  These datas wille be used to feed the trt engines. 
## Installation

  For regular python package installation, type in `pip install -r requirements.txt`

  
  It is not possible for CenterFusion to run without the support of DCNv2 plugin(A Deformable convolutional network algorighm), you should install it with `python` and compile it as TensorRT plugin. 

  - For python package installation, see [here](https://github.com/Abraham423/DCNv2)
  
  - For compiling the DCNv2-TensorRT plugin, see [here](https://github.com/Abraham423/TensorRT-Plugins)
  
  You should then be able to build our source code by 
  
  ```
  cd PATH/TO/THIS/PROJECT
  mkdir build && cd build
  cmake .. -DTRT_LIB_PATH=${TRT_LIB_PATH}
  make 
  ```

  where `${TRT_LIB_PATH}` refers to the library path where you built your DCN-TRT plugin, only in this way can your TRT onnx parser recognize the dcn node in ONNX graph.  


  
## Run trt engine with samples 
  After the installation and input data generation, you can simply go to `${CF_ROOT}`, type in `sh run.sh` to run the project.
  Note that for the first time you run the project, it will take some time to generate trt engine from onnx files.   
  Then you should have the generated results in directory `${CF_ROOT}/results`
## Visualization the results with ros
  To show the results like the video do, you should previously install ros package according you ubuntu version. 
  Then you should compile with you python package 
  ```
  sudo apt-get install python-catkin-tools python3-dev python3-catkin-pkg-modules python3-numpy python3-yaml ros-melodic-cv-bridge
  # Create catkin workspace
  cd ${CF_ROOT}/tools/visualization/catkin_workspace
  catkin init
  # Instruct catkin to set cmake variables, feel free to change path according to your python version
  catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/ libpython3.6m.so
  # Instruct catkin to install built packages into install place. It is $CATKIN_WORKSPACE/install folder
  catkin config --install

  # Find version of cv_bridge in your repository
  apt-cache show ros-melodic-cv-bridge | grep Version
  # Version: 1.13.0-0bionic.20220127.152918
  # Checkout right version in git repo. In our case it is 1.13.0
  cd src/vision_opencv/
  git checkout 1.13.0
  cd ../../
  # Build
  catkin build cv_bridge
  # Extend environment with new package
  source install/setup.bash --extend  #or source install/setup.zsh
  ```
  
  Open another two terminals, one type in `rescore`, another type in `rviz`, in this terminal, type in `python fusion_det_cpp.py`
  You can then add topics  `nusc_image/Image`, `nusc_ego_car/Marker`, `nusc_3dbox/MarkerArray` , then you'll be able to see the detection results.
  
## Exporting to onnx
  Our example use the pretrain model 
  [centerfusion_e60](https://drive.google.com/uc?export=download&id=1XaYx7JJJmQ6TBjCJJster-Z7ERyqu4Ig) (you can export you own model according to this process), you can turn to [here](https://github.com/mrnabati/CenterFusion) to see the detailed metrics of this default model. To export the default model, download this model weight and put it to `tools/CenterFusion/models` .

  Before exporting to onnx, you should previousely install the CenterFusion python dependencies
  ```
  cd tools/CenterFusion
  pip install -r requirements.txt
  ```
  For DCNv2 python package installation, pls turn to [here](https://github.com/Abraham423/DCNv2).

  You can then exporting as onnx models by
  ```
  cd tools/CenterFusion
  sh experiments/export.sh
  ```
  Then you'll see two onnx files in `tools/CenterFusion/models`.


## Computation speed
||Preprocess|CameraInfer|FrustumAssoc|FeatMerge|FusInfer|PostProcess|
|---|---|---|---|---|---|---|
|engine_fp16|0.09|7.44|7.64|0.05|1.00|0.62|
|engine_fp32|0.16|12.03|8.81|0.04|3.05|0.75|

## Computation Graph
 The main modules of procession can be seen here 

<img width="1340" alt="截屏2022-10-25 上午10 25 33" src="https://user-images.githubusercontent.com/25930661/197667512-b3183a3b-0afa-4bd5-b2f8-f28ce2687aa9.png">

The most innovative parts are pc_dep generation and frustum association, we'll illustrate the main ideas.

#### PC_DEP Generation
This is a step where we encode the raw radar points as a structured pseudo-image data format . 

Given each radar points, we firstly generate its coordinated pillar(with size 0.2, 0.2, 1.5 for length,width,height) in camera-viewed 3d space, we then project its 8 corners to image pixel coordinates,  calculating its 2d top-left & bottom-right corners, which defines a 2d box, we insert (loc_z, velo_x,velo_z) of this radar point to each pixel inside this 2d box , when two boxs are intersected, we insert the nearer point feature to the intersected area. The 3d pillars and generated pc_dep can be seen here :

<img width="996" alt="截屏2022-10-24 下午9 12 07" src="https://user-images.githubusercontent.com/25930661/197533715-e46e3b83-7791-4e1f-b058-50df04c5718e.png">

#### Frustum Association
<img width="1021" alt="截屏2022-10-24 下午9 21 34" src="https://user-images.githubusercontent.com/25930661/197694179-711968ec-16ba-423b-b107-4c1f48656098.png">

An object detected using the image features (left), generating the ROI frustum based on object’s 3D
bounding box (middle), and the BEV of the ROI frustum showing radar detections inside the frustum (right). δ is used to increase the
frustum size in the testing phase. d is the ground truth depth in the training phase and the estimated object depth in the testing phase.

## Acknowledgements
This project refers to some codes from 
[CenterFusion](https://github.com/mrnabati/CenterFusion)
but some codes have been slightly modified.

## Contact
Haohao by christian.wong423@gmail.com

