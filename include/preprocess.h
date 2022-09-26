#ifndef __CENTERFUSION_PREPROCESS__
#define __CENTERFUSION_PREPROCESS__

#include <iostream>
#include <fstream>
#include <sstream>
#include "config.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>
using namespace std;


void do_merge(const float* feat_in, const float* pc_dep, float* feat_out);

bool readBinFile(std::string& filename, void*& bufPtr, int& pointNum, int pointDim);
bool readPcNum( void*& bufPtr,std::string& filename);
bool readPC( void*& bufPtr,std::string& filename, int& pointNum);




// radar point cloud to pc_dep
void preprocess(const float* pc_3d, const float* pc_vx, const float* pc_vz, 
           int num_point, float* pc_dep, const float* calib, const float* inv_calib);

void _raw_pc_encode(const float* pc_3d, const float* pc_vx, const float* pc_vz, 
                   const int num_point, float* pc_dep, 
                   const float* calib,const float* inv_calib);

// pc_dep to pc_hm, frustum association 
void generate_pc_hm( const samplesCommon::BufferManager& det_buffers,
                    const float* pc_dep, float* pc_hm,
                    int* dev_score_index, const float* calib, const float* inv_calib);

void _raw_generate_pc_hm(const float* pc_dep, float* pc_hm, 
                        const float* reg, const float* wh, const float* score, 
                        const float* depth,  const float* dim, const float* rot, 
                        int* dev_score_index, const float* calib, const float* inv_calib);

#endif

