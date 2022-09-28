
#include <string>
#include <sys/time.h>
#include <chrono>
#include <thread>
#include <vector>
#include <math.h>
#include "buffers.h"
#include "common.h"
#include <stdio.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include "postprocess.h"
#include "config.h"


void postprocessGpu( const samplesCommon::BufferManager& det_buffers,
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
                    )
{
    for (size_t taskIdx = 0; taskIdx < TASK_NUM; taskIdx++){
        std::vector<Box> predBoxs;
        float* reg = static_cast<float*>(det_buffers.getDeviceBuffer("reg"));
        float* dim = static_cast<float*>(det_buffers.getDeviceBuffer("dim"));
        float* score = static_cast<float*>(det_buffers.getDeviceBuffer("score"));
        int32_t* cls = static_cast<int32_t*>(det_buffers.getDeviceBuffer("label"));

        float* depth = static_cast<float*>(det_buffers.getDeviceBuffer("dep"));
        float* rot = static_cast<float*>(det_buffers.getDeviceBuffer("rot"));

        // cudaStream_t stream;

        int boxSizeAft = _raw_maxpool_nms(score, reg, depth, dim, rot,
                                         dev_score_indexs, score_threshold, dev_calib, dev_invcalib);

        boxSizeAft = boxSizeAft > OUTPUT_NMS_MAX_SIZE ? OUTPUT_NMS_MAX_SIZE : boxSizeAft;
        std::cout << " Num boxes after " <<boxSizeAft << "\n";

        _gather_all(host_boxes, host_label, 
                    reg, depth, dim, rot, score, cls,
                    dev_score_indexs,  boxSizeAft );

        for(auto i =0; i < boxSizeAft; i++){    
            Box box;
            // convert to waymo coord
            box.x = host_boxes[i  + 2 * boxSizeAft];
            box.y = 0.0 - host_boxes[i  + 0 * boxSizeAft];
            box.z = 0.0 - host_boxes[i +  1 * boxSizeAft];
            box.dx = host_boxes[i +  5 * boxSizeAft];
            box.dy = host_boxes[i + 4 * boxSizeAft];
            box.dz = host_boxes[i + 3 * boxSizeAft];
            box.theta = 0.0 - host_boxes[i + 6 * boxSizeAft] - PI/2;
            box.score  = host_boxes[i + 7 * boxSizeAft];
            box.cls = host_label[i];
            box.velX = 0;
            box.velY = 0;
            predResult.push_back(box);
        }
    }
}

        
void postprocessFusGpu( const samplesCommon::BufferManager& det_buffers,
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
                    )
{
    for (size_t taskIdx = 0; taskIdx < TASK_NUM; taskIdx++){
        std::vector<Box> predBoxs;
        float* reg = static_cast<float*>(det_buffers.getDeviceBuffer("reg"));
        float* dim = static_cast<float*>(det_buffers.getDeviceBuffer("dim"));
        float* score = static_cast<float*>(det_buffers.getDeviceBuffer("score"));
        int32_t* cls = static_cast<int32_t*>(det_buffers.getDeviceBuffer("label"));

        float* depth = static_cast<float*>(fus_buffers.getDeviceBuffer("dep_sec"));
        float* rot = static_cast<float*>(fus_buffers.getDeviceBuffer("rot_sec"));

        // cudaStream_t stream;

        int boxSizeAft = _raw_maxpool_nms(score, reg, depth, dim, rot,
                                         dev_score_indexs, score_threshold, dev_calib, dev_invcalib);

        boxSizeAft = boxSizeAft > OUTPUT_NMS_MAX_SIZE ? OUTPUT_NMS_MAX_SIZE : boxSizeAft;
        std::cout << " Num boxes after " <<boxSizeAft << "\n";

        _gather_all(host_boxes, host_label, 
                    reg, depth, dim, rot, score, cls,
                    dev_score_indexs,  boxSizeAft );

        for(auto i =0; i < boxSizeAft; i++){    
            Box box;
            // convert to waymo coord
            box.x = host_boxes[i  + 2 * boxSizeAft];
            box.y = 0.0 - host_boxes[i  + 0 * boxSizeAft];
            box.z = 0.0 - host_boxes[i +  1 * boxSizeAft];
            box.dx = host_boxes[i +  5 * boxSizeAft];
            box.dy = host_boxes[i + 4 * boxSizeAft];
            box.dz = host_boxes[i + 3 * boxSizeAft];
            box.theta = 0.0 - host_boxes[i + 6 * boxSizeAft] - PI/2;
            box.score  = host_boxes[i + 7 * boxSizeAft];
            box.cls = host_label[i];
            box.velX = 0;
            box.velY = 0;
            predResult.push_back(box);
        }
    }
}





