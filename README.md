# 视觉毫米波雷达融合

https://user-images.githubusercontent.com/25930661/195581012-5cb6a223-7294-4fa6-afb2-7393345831ee.mp4




## Todo List
- DCNv2 source code & Trt plugin support 
- rviz visualization
- center fusion source code 
- center fusion onnx exporting script
- Frustum association method illustration

## Installation
  
  It is not possible for CenterFusion to run without the support of DCNv2 plugin(A Deformable convolutional network algorighm), you should install it with `python` and compile it as TensorRT plugin. 

  - For python package installation, see [here](https://github.com/Abraham423/DCNv2)
  
  - For compiling the DCNv2-TensorRT plugin, see [here](https://github.com/Abraham423/TensorRT-Plugins)
  
  You should then be able to build our source code by 
  
  ```
  cd PATH/TO/THIS/PROJECT
  mkdir build && cd build
  cmake .. -DTRT_LIB_PATH=${TRT_LIB_PATH}
  ```

  where `${TRT_LIB_PATH}` refers to the library path where you built your DCN-TRT plugin, only in this way can your TRT onnx parser recognize the dcn node in ONNX graph.  
  
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

## Input data
  Todo


## Computation speed
||Preprocess|CameraInfer|FrustumAssoc|FeatMerge|FusInfer|PostProcess|
|---|---|---|---|---|---|---|
|engine_fp16|0.09|7.44|7.64|0.05|1.00|0.62|
|engine_fp32|0.16|12.03|8.81|0.04|3.05|0.75|


