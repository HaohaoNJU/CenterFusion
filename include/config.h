#ifndef __CENTERFUSION_CONFIG_H__
#define __CENTERFUSION_CONFIG_H__


// ========================================NUSCENES CNETERFUSION======================================== 

// input-output size 
#define INPUT_H 448 
#define INPUT_W 800
#define OUTPUT_H 112
#define OUTPUT_W 200
#define OUT_SIZE_FACTOR 4.0f    

#define FEATMAP_CHANNEL 64
#define PC_CHANNEL 3

#define PI 3.141592653f
const int THREADS_PER_BLOCK_NMS = sizeof(unsigned long long) * 8;
#define TASK_NUM 1
#define INPUT_NMS_MAX_SIZE 1024
#define OUTPUT_NMS_MAX_SIZE 200
#define NMS_KERNEL_SIZE 3

// For radar pc encode
// camera coordinates 
#define PILLAR_SIZE_X 0.2
#define PILLAR_SIZE_Y 1.5
#define PILLAR_SIZE_Z 0.2

#define MAX_POINT 200
#define TOP_K 100
#define MAX_DISTANCE 300
#define HM_MASK_RATIO 0.3f 
#define HM_DEPTH_NORM 60.0f
#endif

// ========================================WAYMO CENTERPOINT CONFIG======================================== 
// // point size
// #define MAX_POINTS 220000
// #define POINT_DIM 5

// // pillar size 
// #define X_STEP 0.32f
// #define Y_STEP 0.32f
// #define X_MIN -74.88f
// #define X_MAX 74.88f
// #define Y_MIN -74.88f
// #define Y_MAX 74.88f
// #define Z_MIN -2.0f
// #define Z_MAX 4.0f

// #define X_CENTER_MIN -80.0f
// #define X_CENTER_MAX 80.0f
// #define Y_CENTER_MIN -80.0f
// #define Y_CENTER_MAX 80.0f
// #define Z_CENTER_MIN -10.0f
// #define Z_CENTER_MAX 10.0f

// #define PI 3.141592653f
// // paramerters for preprocess
// #define BEV_W 468
// #define BEV_H 468
// #define MAX_PILLARS 32000 //20000 //32000
// #define MAX_PIONT_IN_PILLARS 20
// #define FEATURE_NUM 10
// #define PFE_OUTPUT_DIM 64
// #define THREAD_NUM 4
// // paramerters for postprocess
// #define SCORE_THREAHOLD 0.1f
// #define NMS_THREAHOLD 0.7f
// #define INPUT_NMS_MAX_SIZE 4096
// #define OUTPUT_NMS_MAX_SIZE 500
// // #define THREADS_PER_BLOCK_NMS  sizeof(unsigned long long) * 8
// const int THREADS_PER_BLOCK_NMS = sizeof(unsigned long long) * 8;

// // OUT_SIZE_FACTOR * OUTPUT_H  * Y_STEP = Y_MAX - Y_MIN
// #define OUT_SIZE_FACTOR 1.0f    

// #define TASK_NUM 1
// #define REG_CHANNEL 2
// #define HEIGHT_CHANNEL 1
// #define ROT_CHANNEL 2
// // #define VEL_CHANNEL 2 //don't defined in waymo
// #define DIM_CHANNEL 3

// // spatial output size of rpn 
// #define OUTPUT_H 468  
// #define OUTPUT_W 468
// #endif








