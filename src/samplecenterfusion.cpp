/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include "NvInfer.h"
#include <cstdlib>
#include <map>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <chrono>
#include "utils.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "centerfusion.h"


using namespace Eigen;

const std::string gSampleName = "TensorRT.sample_onnx_centerfusion";

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout
        << "Usage: ./centerpoint [-h or --help]"
        << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--filePath       Specify path to a data directory. "
              << std::endl;
    std::cout << "--savePath       Specify path to a directory you want save detection results."
              << std::endl;

    std::cout << "--loadEngine       Load from serialized engine files or from onnx files, provide this argument only when you want to create "
    "engine from serialized engine files you previously generated(and provide paths to engine files), or you will need to provide paths to onnx files. "
              << std::endl;   

    std::cout << "--detectionOnnxPath       Specify path to detection onnx model. This option can be used when you want to create engine from onnx file. "
              << std::endl;
    std::cout << "--fusionOnnxPath       Specify path to fusion onnx model. This option can be used when you want to create engine from onnx file. "
              << std::endl;      
    std::cout << "--detectionEnginePath       Specify path to detection engine model. This option can be used when you want to create engine from serialized engine file you previously generated. "
              << std::endl;
    std::cout << "--fusionEnginePath       Specify path to fusion engine model. This option can be used when you want to create engine from serialized engine file you previously generated.  "
              << std::endl;   

    std::cout << "--fp16       Provide this argument only when you want  to do inference on fp16 mode, note that this config is only valid when you create engine from onnx files. "
              << std::endl;   
    std::cout << "--scoreThre      Specify score threshould when making post-process. " << std::endl;
    std::cout << "--nmsThre      Specify iou threshould when make non-maximum supression. " << std::endl;

    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform, by default it's set -1."
              << std::endl;
}

int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }
    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);
    sample::gLogger.reportTestStart(sampleTest);



    ///////////////////////////////////////////////////////////////PARAM INITIALIZATION///////////////////////////////////////////////////////////////
    Params params;
    // initialize sample parameters 
    params.detectionOnnxFilePath =  args.detectionOnnxPath;
    params.fusionOnnxFilePath =  args.fusionOnnxPath;
    params.detectionSerializedEnginePath = args.detectionEnginePath;
    params.fusionSerializedEnginePath = args.fusionEnginePath;
    params.savePath = args.savePath;
    params.filePaths=glob(args.filePath + "/*.bin");

    params.nms_thre = args.nmsThre;
    params.score_thre = args.scoreThre;

    params.fp16 = args.runInFp16;
    std::cout << "fp 16 " << params.fp16 << std::endl; 
    params.load_engine = args.loadEngine;

    // Input Output Names, according to TASK_NUM
    // params.detectionInputTensorNames.push_back("input.1");
    // params.fusionInputTensorNames.push_back("input.1");
    // params.detectionOutputTensorNames.push_back("47");
    

    Eigen::Matrix<float,3,3> imgCalib = Eigen::Matrix<float,3,3>::Identity();
    Eigen::Matrix<float,3,3> imgFeatTrans = Eigen::Matrix<float,3,3>::Identity();





    // imgCalib << 1.2664172e+03, 0.0000000e+00, 8.1626703e+02,
    //             0.0000000e+00, 1.2664172e+03, 4.9150708e+02,
    //             0.0000000e+00, 0.0000000e+00, 1.0000000e+00;

    imgCalib << 1.2528e+03, 0.0000e+00, 8.2659e+02,
                    0.0000e+00, 1.2528e+03, 4.6998e+02,
                    0.0000e+00, 0.0000e+00, 1.0000e+00;

    // imgFeatTrans << 8.,0,0,
    //                 0,8.,2.,
    //                 0,0,1. ;

    imgFeatTrans << 0.125, 0 , 0,
                    0, 0.125, -0.25,
                    0, 0,  1 ;
    
    //////////////////////////////////////////////////////
    //get calib, XYZ -> u,v,1 in feature map
    Eigen::Matrix3f featCalib =  imgFeatTrans * imgCalib;

    // TODO :  transposr from itself, this is because matrix.data() stores the values by columns ! 
    featCalib.transposeInPlace(); 
    Eigen::Matrix3f featInvCalib = featCalib.inverse();

    params.featCalib = featCalib;
    params.featInvCalib = featInvCalib;

    // Attrs
    params.dlaCore = args.useDLACore;
    params.batch_size = 1;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // std::string savePath = "/home/wanghao/Desktop/projects/notebooks/centerpoint_output_cpp" ;
    CenterFusion sample(params);
    sample::gLogInfo << "Building and running a GPU inference engine for CenterFusion" << std::endl;
    if (!sample.engineInitlization())
    {
        sample::gLogInfo << "sample build error  " << std::endl;
        return sample::gLogger.reportFail(sampleTest);
    }
    
    if (!sample.infer())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    sample::gLogger.reportPass(sampleTest);
    return 1;
}







