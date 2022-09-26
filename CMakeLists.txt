cmake_minimum_required(VERSION 2.8.3)
project(centerfusion)


set(SAMPLES_COMMON_SOURCES
${PROJECT_SOURCE_DIR}/include/common/logger.cpp
)
set(TRT_LIB_PATH /mnt/Desktop/projects/TensorRT/lib) #/mnt/Desktop/pkgs/TensorRT-8.0.1.6/lib)

find_package(PkgConfig QUIET)
pkg_check_modules(
    CF_DEPEND
    eigen3
)

# pcl and boost related !
find_package(Boost COMPONENTS program_options REQUIRED )
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
 

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

# set flags for CUDA availability
find_package(CUDA REQUIRED)
message("CUDA dir paths", ${CUDA_LIBRARIES})

include_directories( 
  ${PROJECT_SOURCE_DIR}/include/common
  ${PROJECT_SOURCE_DIR}/include
  ${CUDA_INCLUDE_DIRS}
  ${CF_DEPEND_INCLUDE_DIRS}
  )

message("CUDA Libs: ${CUDA_LIBRARIES}")
message("CUDA Headers: ${CUDA_INCLUDE_DIRS}")


# set flags for TensorRT availability
# try to find the tensorRT modules
find_library(NVINFER NAMES nvinfer HINTS ${TRT_LIB_PATH} REQUIRED )
find_library(NVPARSERS NAMES nvparsers HINTS ${TRT_LIB_PATH} REQUIRED )
find_library(NVONNXPARSERS NAMES nvonnxparser HINTS ${TRT_LIB_PATH} REQUIRED )

message("TRT LIB :   ${TRT_LIB_PATH}")
message("NVINFER: ${NVINFER}")
message("NVPARSERS: ${NVPARSERS}")
message("NVONNXPARSERS: ${NVONNXPARSERS}")


# set flags for CUDNN availability
# try to find the CUDNN module
find_library(CUDNN_LIBRARY
  NAMES libcudnn.so${__cudnn_ver_suffix} libcudnn${__cudnn_ver_suffix}.dylib ${__cudnn_lib_win_name}
  PATHS $ENV{LD_LIBRARY_PATH} ${__libpath_cudart} ${CUDNN_ROOT_DIR} ${PC_CUDNN_LIBRARY_DIRS} ${CMAKE_INSTALL_PREFIX}
  PATH_SUFFIXES lib lib64 bin
  DOC "CUDNN library."
  REQUIRED
)

message("CUDNN_LIBRARY: ${CUDNN_LIBRARY}")
set(CUDNN_AVAIL ON)



file(GLOB PostFusionLibs ${PROJECT_SOURCE_DIR}/src/*cpp)
add_library(postfusions SHARED ${PostFusionLibs})
target_link_libraries(postfusions nvonnxparser)

file(GLOB PointPillarCU ${PROJECT_SOURCE_DIR}/src/*cu)
cuda_add_library(postfusionsCU SHARED ${PointPillarCU})

set(CUDA_LIB_PATH /usr/local/cuda/lib64/)

MESSAGE(${PROJECT_SOURCE_DIR})
add_executable(centerfusion ${PROJECT_SOURCE_DIR}/src/samplecenterfusion.cpp ${SAMPLES_COMMON_SOURCES})

target_link_libraries(centerfusion 
                         ${CUDA_LIBRARY} 
                         ${CUDA_RUNTIME_LIBRARY} 
                         ${CF_DEPEND_LIBRARIES}  
                         ${Boost_LIBRARIES}
                         ${OpenCV_LIBRARIES}
                         nvinfer
                         postfusions
                         ${CUDA_LIB_PATH}libcudart.so
                         postfusionsCU
                         )


