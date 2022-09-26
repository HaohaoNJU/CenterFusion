#include <string>
#include <sys/time.h>
#include <chrono>
#include <thread>
#include <vector>
#include "logger.h"
#include "iostream"
#include <fstream>
#include <iostream>
#include <sstream>
#include "common.h"
#include"preprocess.h"

void preprocess(const float* pc_3d, const float* pc_vx, const float* pc_vz,  int num_point, float* pc_dep, const float* calib, const float* inv_calib)
{
    num_point = num_point>MAX_POINT ? MAX_POINT : num_point;
    _raw_pc_encode(pc_3d, pc_vx, pc_vz, 
                   num_point, pc_dep, 
                   calib,inv_calib);
}

void generate_pc_hm( const samplesCommon::BufferManager& det_buffers,
                    const float* pc_dep, float* pc_hm,
                    int* dev_score_index, const float* calib, const float* inv_calib)
{
    float* reg = static_cast<float*>(det_buffers.getDeviceBuffer("reg"));
    float* dim = static_cast<float*>(det_buffers.getDeviceBuffer("dim"));
    float* score = static_cast<float*>(det_buffers.getDeviceBuffer("score"));
    float* depth = static_cast<float*>(det_buffers.getDeviceBuffer("dep"));
    float* rot = static_cast<float*>(det_buffers.getDeviceBuffer("rot"));
    float* wh = static_cast<float*>(det_buffers.getDeviceBuffer("wh"));

    _raw_generate_pc_hm(pc_dep,  pc_hm,  reg, wh, score, 
                        depth,  dim,  rot, dev_score_index,  calib,  inv_calib);
}

bool readPcNum(void*& bufPtr, std::string& filename)
{
    // open the file:
    std::streampos fileSize;
    std::ifstream file(filename, std::ios::binary);
    
    if (!file) {
        sample::gLogError << "[Error] Open file " << filename << " failed" << std::endl;
        return false;
    }
    // get its size:
    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    
    bufPtr = malloc(fileSize);
    if(bufPtr == nullptr){
        sample::gLogError << "[Error] Malloc Memory Failed! Size: " << fileSize << std::endl;
        return false;
    }
    // read the data:
    file.read((char*) bufPtr, fileSize);
    file.close();
    return true;
}


bool readPC(void*& bufPtr, std::string& filename, int& pointNum)
{
     int pointDim = 5;
    // open the file:
    std::streampos fileSize;
    std::ifstream file(filename, std::ios::binary);
    
    if (!file) {
        sample::gLogError << "[Error] Open file " << filename << " failed" << std::endl;
        return false;
    }
    // get its size:
    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    
    bufPtr = malloc(fileSize);
    if(bufPtr == nullptr){
        sample::gLogError << "[Error] Malloc Memory Failed! Size: " << fileSize << std::endl;
        return false;
    }
    // read the data:
    file.read((char*) bufPtr, fileSize);
    file.close();
    
    pointNum = fileSize /sizeof(float) / pointDim;
    if( fileSize /sizeof(float) % pointDim != 0){
         sample::gLogError << "[Error] File Size Error! " << fileSize << std::endl;
    }
    return true;
}

bool readBinFile(std::string& filename, void*& bufPtr, int& pointNum, int pointDim)
{
    // open the file:
    std::streampos fileSize;
    std::ifstream file(filename, std::ios::binary);
    
    if (!file) {
        sample::gLogError << "[Error] Open file " << filename << " failed" << std::endl;
        return false;
    }
    // get its size:
    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    
    bufPtr = malloc(fileSize);
    if(bufPtr == nullptr){
        sample::gLogError << "[Error] Malloc Memory Failed! Size: " << fileSize << std::endl;
        return false;
    }
    // read the data:
    file.read((char*) bufPtr, fileSize);
    file.close();
    
    pointNum = fileSize /sizeof(float) / pointDim;
    if( fileSize /sizeof(float) % pointDim != 0){
         sample::gLogError << "[Error] File Size Error! " << fileSize << std::endl;
    }
    return true;
}
