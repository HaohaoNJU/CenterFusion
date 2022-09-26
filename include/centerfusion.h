#ifndef __CENTERFUSION_H__
#define __CENTERFUSION_H__
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <chrono>

#include <Eigen/Core>
#include "preprocess.h"
#include "postprocess.h"

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

struct Params{
    std::string detectionOnnxFilePath = "";
    std::string fusionOnnxFilePath = "";
    std::string detectionSerializedEnginePath = "";
    std::string fusionSerializedEnginePath = "";

    // Input Output Names
    std::vector<std::string> detectionInputTensorNames;
    std::vector<std::string> fusionInputTensorNames;
    std::vector<std::string> detectionOutputTensorNames;
    std::vector<std::string> fusionOutputTensorNames;

    // Input Output Paths
    std::string savePath ;
    std::vector<std::string>  filePaths;
        
    // Attrs
    float nms_thre = 0.7;
    float score_thre = 0.2;
    int dlaCore = -1;
    bool fp16 = false;
    bool int8 = false;
    bool load_engine = false;
    int batch_size = 1;

    // img parameters
    // Eigen::Matrix<float,3,3> imgCalib = Eigen::Matrix<float,3,3>::Identity();
    // Eigen::Matrix<float,3,3> imgFeatTrans = Eigen::Matrix<float,3,3>::Identity();

    // imgCalib << 1.2528e+03, 0.0000e+00, 8.2659e+02,
    //                 0.0000e+00, 1.2528e+03, 4.6998e+02,
    //                 0.0000e+00, 0.0000e+00, 1.0000e+00;
    
    // imgFeatTrans << 0.125, 0 , 0,
    //                 0, 0.125, -0.25,
    //                 0, 0,  1 ;
    
    ////////////////////////////////////////////////////////
    //get calib, XYZ -> u,v,1 in feature map
    // Eigen::Matrix3f featCalib =  imgFeatTrans * imgCalib;
    // Eigen::Matrix3f featInvCalib = featCalib.inverse();
    Eigen::Matrix3f featCalib;
    Eigen::Matrix3f featInvCalib;
    
};
class CenterFusion
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    CenterFusion(const Params params)
        : m_params_(params)
        ,BATCH_SIZE_(params.batch_size)
        , m_engine_detection_(nullptr)
        ,m_engine_fusion_(nullptr)
    {

        //const int NUM_THREADS, const int MAX_NUM_PILLARS, const int GRID_X_SIZE, const int GRID_Y_SIZE):
        // scatter_cuda_ptr_.reset(new ScatterCuda(PFE_OUTPUT_DIM, PFE_OUTPUT_DIM, BEV_W, BEV_H ));
        // mallocate a global memory for pointer

        // GPU_CHECK(cudaMalloc((void**)&dev_points_, MAX_POINTS * POINT_DIM * sizeof(float)));
        // GPU_CHECK(cudaMemset(dev_points_,0, MAX_POINTS * POINT_DIM * sizeof(float)));

        /**
         * @brief : Create and Init Variables for PostProcess
         * 
         */
        GPU_CHECK(cudaMalloc((void**)&dev_pc_dep_, OUTPUT_H * OUTPUT_W * 3 * sizeof(float)));
        GPU_CHECK(cudaMemset(dev_pc_dep_, 0, OUTPUT_H * OUTPUT_W * 3 * sizeof(float)));

        GPU_CHECK(cudaMalloc((void**)&dev_score_idx_, OUTPUT_W * OUTPUT_H * sizeof(int)));
        GPU_CHECK(cudaMemset(dev_score_idx_, -1 , OUTPUT_W * OUTPUT_H * sizeof(int)));

        GPU_CHECK(cudaMalloc((void**)&dev_feat_calib_, 3 * 3 * sizeof(float)));
        GPU_CHECK(cudaMemcpy(dev_feat_calib_, m_params_.featCalib.data(),  3 * 3 * sizeof(float), cudaMemcpyHostToDevice));

        GPU_CHECK(cudaMalloc((void**)&dev_feat_invcalib_, 3 * 3 * sizeof(float)));
        GPU_CHECK(cudaMemcpy(dev_feat_invcalib_, m_params_.featInvCalib.data(),  3 * 3 * sizeof(float), cudaMemcpyHostToDevice));

        GPU_CHECK(cudaMallocHost((void**)& mask_cpu, INPUT_NMS_MAX_SIZE * DIVUP (INPUT_NMS_MAX_SIZE ,THREADS_PER_BLOCK_NMS) * sizeof(unsigned long long)));
        GPU_CHECK(cudaMemset(mask_cpu, 0 ,  INPUT_NMS_MAX_SIZE * DIVUP (INPUT_NMS_MAX_SIZE ,THREADS_PER_BLOCK_NMS) * sizeof(unsigned long long)));

        GPU_CHECK(cudaMallocHost((void**)& remv_cpu, THREADS_PER_BLOCK_NMS * sizeof(unsigned long long)));
        GPU_CHECK(cudaMemset(remv_cpu, 0 ,  THREADS_PER_BLOCK_NMS  * sizeof(unsigned long long)));

        GPU_CHECK(cudaMallocHost((void**)&host_score_idx_, OUTPUT_W * OUTPUT_H  * sizeof(int)));
        GPU_CHECK(cudaMemset(host_score_idx_, -1, OUTPUT_W * OUTPUT_H  * sizeof(int)));

        GPU_CHECK(cudaMallocHost((void**)&host_keep_data_, INPUT_NMS_MAX_SIZE * sizeof(long)));
        GPU_CHECK(cudaMemset(host_keep_data_, -1, INPUT_NMS_MAX_SIZE * sizeof(long)));

        GPU_CHECK(cudaMallocHost((void**)&host_boxes_, OUTPUT_NMS_MAX_SIZE * 8 * sizeof(float)));
        GPU_CHECK(cudaMemset(host_boxes_, 0 ,  OUTPUT_NMS_MAX_SIZE * 8 * sizeof(float)));

        GPU_CHECK(cudaMallocHost((void**)&host_label_, OUTPUT_NMS_MAX_SIZE * sizeof(int)));
        GPU_CHECK(cudaMemset(host_label_, -1, OUTPUT_NMS_MAX_SIZE * sizeof(int)));


    }

    ~CenterFusion() {
    // Free host pointers
    // Free global pointers 
    sample::gLogInfo << "Free Variables . \n";
    GPU_CHECK(cudaFree(dev_score_idx_));
    GPU_CHECK(cudaFree(dev_feat_calib_));
    GPU_CHECK(cudaFree(dev_feat_invcalib_));

    GPU_CHECK(cudaFreeHost(host_keep_data_));
    GPU_CHECK(cudaFreeHost(host_boxes_));
    GPU_CHECK(cudaFreeHost(host_label_));
    GPU_CHECK(cudaFreeHost(host_score_idx_));
    GPU_CHECK(cudaFreeHost(remv_cpu));
    GPU_CHECK(cudaFreeHost(mask_cpu));
    }

    std::shared_ptr<nvinfer1::ICudaEngine>  buildFromOnnx( std::string onnxFilePath);
    std::shared_ptr<nvinfer1::ICudaEngine>  buildFromSerializedEngine(std::string serializedEngineFile);
    bool infer();
    bool engineInitlization();
    


private:
    // device pointers 
    float* dev_img_;
    float* dev_featmap_; // output of detection
    float* dev_feat_; // inpuf of fusion 
    float* dev_pc_dep_;
    float* dev_pc_hm_;
    float* dev_calib_;
    
    float* dev_feat_calib_;
    float* dev_feat_invcalib_;

    int* dev_score_idx_;

    // float* dev_pillars_ ;
    // float* dev_scattered_feature_;
    // float* dev_points_ ;
    // int* dev_indices_;
    // long* dev_keep_data_;
    // // SampleUniquePtr<ScatterCuda> scatter_cuda_ptr_;

    // // device pointers for preprocess
    // int* p_bev_idx_; 
    // int* p_point_num_assigned_;
    // bool* p_mask_;
    // int* bev_voxel_idx_; // H * W
    // float* v_point_sum_;
    // int* v_range_;
    // int* v_point_num_;
    

    // // host  variables for post process
    long* host_keep_data_;
    float* host_boxes_;
    int* host_label_;
    int* host_score_idx_;
    unsigned long long* mask_cpu;
    unsigned long long* remv_cpu;


    Params m_params_;
    int BATCH_SIZE_ = 1;
    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
    int mNumber{0};             //!< The number to classify
    std::shared_ptr<nvinfer1::ICudaEngine> m_engine_detection_; //!< The TensorRT engine used to run the network
    std::shared_ptr<nvinfer1::ICudaEngine> m_engine_fusion_;

    SampleUniquePtr<nvinfer1::IExecutionContext> m_context_detection_;
    SampleUniquePtr<nvinfer1::IExecutionContext> m_context_fusion_;
    //!
    //! \brief Parses an ONNX model for MNIST and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser,
       std::string onnxFilePath);
    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(void*& points, std::string& pointFilePath, int& pointNum);
    //!
    //! \brief Classifies digits and verify result
    //!
    void saveOutput(std::vector<Box>& predResult, std::string& inputFileName, std::string savePath);
};

#endif