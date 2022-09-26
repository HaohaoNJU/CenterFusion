#ifndef __CENTERFUSION_POSTPROCESS__
#define __CENTERFUSION_POSTPROCESS__

#include "buffers.h"
#include "common.h"
#include "config.h"
#include <stdint.h>

#include <cuda_runtime_api.h>

#define GPU_CHECK(ans)                                                                                                                               \
  {                                                                                                                                                                                 \                                      
    GPUAssert((ans), __FILE__, __LINE__);                                                                                                 \
  }
                                                                                                                                                                                   
inline void GPUAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

struct Box{
    float x;
    float y;
    float z;
    float dx;
    float dy;
    float dz;
    float velX;
    float velY;
    float theta;

    float score;
    int cls;
    bool isDrop; // for nms
};

int _raw_maxpool_nms(const float* score, float* reg,  const float* depth, float* dim , float* rot,
                    int* score_indexs,  const float score_thre, const float* calib, const float* inv_calib);

int _raw_nms_gpu(float* reg,  const float* depth, float* dim , float* rot,
                const int* indexs, long* host_keep_data,unsigned long long* mask_cpu, unsigned long long* remv_cpu,
                int boxes_num,  float nms_overlap_thresh,
                const float* calib, const float* inv_calib) ;

void _sort_by_key(float* keys, int* values,int size) ;

// gather values when maxpool nms is enabled ! 
void _gather_all(float* host_boxes, int* host_label, 
                float* reg, float* depth, float* dim, float* rot,  float* score, int32_t* label,  
                int* dev_indexs, int boxSizeAft) ;

// gathrer values when bev nms is enabled ! 
void _gather_all(float* host_boxes, int* host_label, 
                float* reg, float* height, float* dim, float* rot,  float* sorted_score, int32_t* label,  
                int* dev_indexs, long* host_keep_indexs,  int boxSizeBef, int boxSizeAft) ; 

int _find_valid_score_num(float* score, float thre, int output_h, int output_w) ;
// void _find_valid_score_num(float* score, float thre, int output_h, int output_w, int* box_size); //,  thrust::host_vector<int>  host_box_size);


void postprocessGpu(const samplesCommon::BufferManager& det_buffers,
                                                 std::vector<Box>& predResult ,
                                                 int* dev_score_indexs,
                                                 unsigned long long* mask_cpu,
                                                 unsigned long long* remv_cpu,
                                                 int* host_score_indexs,
                                                 long* host_keep_data,
                                                 float* host_boxes,
                                                 int* host_label,
                                                 float score_threshold,
                                                 float nms_threshold,
                                                 float* dev_calib,
                                                 float* dev_invcalib
                                                 );


void postprocessFusGpu(const samplesCommon::BufferManager& det_buffers,
                   const samplesCommon::BufferManager& fus_buffers,
                                                 std::vector<Box>& predResult ,
                                                 int* dev_score_indexs,
                                                 unsigned long long* mask_cpu,
                                                 unsigned long long* remv_cpu,
                                                 int* host_score_indexs,
                                                 long* host_keep_data,
                                                 float* host_boxes,
                                                 int* host_label,
                                                 float score_threshold,
                                                 float nms_threshold,
                                                 float* dev_calib,
                                                 float* dev_invcalib
                                                 );

#endif