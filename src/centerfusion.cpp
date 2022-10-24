#include "centerfusion.h"
#include "utils.h"

 void tmpSave(float* results , std::string outputFilePath, size_t size, size_t line_num ) 
 {
    ofstream resultFile;
    resultFile.exceptions ( std::ifstream::failbit | std::ifstream::badbit );
    try {
        resultFile.open(outputFilePath);
        for (size_t idx = 0; idx < size ; idx++)
        {
                resultFile <<  results[idx ];
                if ((idx+1) % line_num==0)
                    resultFile <<"\n";
                else resultFile << " ";
        }
        resultFile.close();
    }
    catch (std::ifstream::failure e) 
    {
        sample::gLogError << "Open File: " << outputFilePath << " Falied"<< std::endl;
    }
 }


// Only used in nusc calib data preprocess . 
void tmpCalibPrep(float* raw_img_calib, float* dev_feat_calib, float* dev_feat_invcalib)
{

    Eigen::Matrix<float,3,3> imgCalib = Eigen::Matrix<float,3,3>::Identity();
    Eigen::Matrix<float,3,3> imgFeatTrans = Eigen::Matrix<float,3,3>::Identity();

    imgFeatTrans << 0.125, 0 , 0,
                    0, 0.125, -0.25,
                    0, 0,  1 ;

    imgCalib << raw_img_calib[0 + 4 * 0],raw_img_calib[1 + 4 * 0],raw_img_calib[2 + 4 * 0],
                raw_img_calib[0 + 4 * 1],raw_img_calib[1 + 4 * 1],raw_img_calib[2 + 4 * 1],
                raw_img_calib[0 + 4 * 2],raw_img_calib[1 + 4 * 2],raw_img_calib[2 + 4 * 2];
    Eigen::Matrix3f featCalib =  imgFeatTrans * imgCalib;
    // TODO :  transposr from itself, this is because matrix.data() stores the values by columns ! 
    featCalib.transposeInPlace(); 
    Eigen::Matrix3f featInvCalib = featCalib.inverse();
    
    GPU_CHECK(cudaMemcpy(dev_feat_calib, featCalib.data(),  3 * 3 * sizeof(float), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(dev_feat_invcalib, featInvCalib.data(),  3 * 3 * sizeof(float), cudaMemcpyHostToDevice));

}

bool serialize_engine_disk(samplesCommon::SampleUniquePtr<nvinfer1::IHostMemory>& gieModelStream, std::string& engine_path)
{
    // serialize mEngine to disk
    sample::gLogInfo << "allocate memory size: " << gieModelStream->size() << " bytes." << std::endl;
    std::ofstream outfile(engine_path.c_str(), std::ios::out | std::ios::binary);
    if(!outfile.is_open()){
        sample::gLogError << "Fail to open file to write: " << engine_path << std::endl;
        return false;
    }

    unsigned char* p = (unsigned char*)gieModelStream->data();
    outfile.write((char*)p, gieModelStream->size());
    outfile.close();
    sample::gLogInfo << "Serialized engine to disk . . .  " << engine_path << std::endl;

    return true;
}




bool CenterFusion::engineInitlization()
 {

        bool plugin_inited = initLibNvInferPlugins(&sample::gLogger.getTRTLogger(),"");
        if (!plugin_inited && m_params_.load_engine)
        {
            sample::gLogError << "Failed to initialize plugins. \n";
        }
        if (m_params_.load_engine)
        {
            sample::gLogInfo << "Building detection engine . . .  "<< std::endl;
            m_engine_detection_ = buildFromSerializedEngine(m_params_.detectionSerializedEnginePath);
            sample::gLogInfo << "Building fusion engine . . .  "<< std::endl;
            m_engine_fusion_ = buildFromSerializedEngine(m_params_.fusionSerializedEnginePath); 
        }
        else
        {
            sample::gLogInfo << "Building detection engine . . .  "<< std::endl;
            m_engine_detection_ = buildFromOnnx(m_params_.detectionOnnxFilePath);
            sample::gLogInfo << "Building fusion engine . . .  "<< std::endl;
            m_engine_fusion_ = buildFromOnnx(m_params_.fusionOnnxFilePath);

            SampleUniquePtr<nvinfer1::IHostMemory> gieModelStreamDet{m_engine_detection_->serialize()};
            serialize_engine_disk(gieModelStreamDet, m_params_.detectionSerializedEnginePath);
             // sample::gLogInfo << "Serialize engine to: " << m_params_.pfeSerializedEnginePath << ", " << m_params_.rpnSerializedEnginePath << std::endl;
            sample::gLogInfo << "Serialize detection engine to: "  << m_params_.detectionSerializedEnginePath << std::endl;

            SampleUniquePtr<nvinfer1::IHostMemory> gieModelStreamFus{m_engine_fusion_->serialize()};
            serialize_engine_disk(gieModelStreamFus, m_params_.fusionSerializedEnginePath);
             // sample::gLogInfo << "Serialize engine to: " << m_params_.pfeSerializedEnginePath << ", " << m_params_.rpnSerializedEnginePath << std::endl;
            sample::gLogInfo << "Serialize fusion engine to: "  << m_params_.fusionSerializedEnginePath << std::endl;
        }

        sample::gLogInfo << "All has Built !  "<< std::endl;

        return true;
}

std::shared_ptr<nvinfer1::ICudaEngine> CenterFusion::buildFromSerializedEngine(std::string serializedEngineFile) 
{

     std::vector<char> trtModelStream_;
     size_t size{0};
     std::ifstream file(serializedEngineFile, std::ios::binary);
     if (file.good()) 
    {
         file.seekg(0, file.end);
         size = file.tellg();
         file.seekg(0,file.beg);
         trtModelStream_.resize(size);
         file.read(trtModelStream_.data(), size);
         file.close() ;
     }
     else 
     {
        sample::gLogError<< " Failed to read serialized engine ! " << std::endl;
        return nullptr;
     }
    SampleUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
    if(!runtime) { sample::gLogError << "Failed to create runtime \n"; return nullptr;}
    sample::gLogInfo<<"Create ICudaEngine  !" << std::endl;
    std::shared_ptr<nvinfer1::ICudaEngine>  engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(trtModelStream_.data(), size), 
        samplesCommon::InferDeleter());

    if (!engine)
    {
        sample::gLogError << "Failed to create engine \n";
        return nullptr;
    }

    return engine;
}

std::shared_ptr<nvinfer1::ICudaEngine>  CenterFusion::buildFromOnnx(std::string  onnxFilePath)
{
    // We assumed that nvinfer1::createInferBuilder is droped in TRT 8.0 or above
    
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    { 
        sample::gLogError<< "Builder not created !" << std::endl;
        return nullptr;
    }
   



    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        sample::gLogError<< "Network not created ! " << std::endl;
        return nullptr;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        sample::gLogError<< "Config not created ! " << std::endl;
        return nullptr;
    }
    auto parser
        = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        sample::gLogError<< "Parser not created ! " << std::endl;
        return nullptr;
    }
    sample::gLogInfo<<"ConstructNetwork !" << std::endl;
    
    cudaEvent_t  start, end;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start,stream);    

    auto constructed = constructNetwork(builder, network, config, parser,onnxFilePath);


    if (!constructed)
    {
        return nullptr;
    }
    
    // cuda stream used to profiling the builder
    auto profileStream = samplesCommon::makeCudaStream();
    if(!profileStream) {
        sample::gLogError<<"Failed to create a profile stream !\n";
        return  nullptr;
    }
    config->setProfileStream(*profileStream);    

    SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {sample::gLogError << "Failed to create IHostMemory plan \n";return  nullptr;}


    SampleUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
    if(!runtime) { sample::gLogError << "Failed to create runtime \n"; return nullptr;}
    sample::gLogInfo<<"Create ICudaEngine  !" << std::endl;
    std::shared_ptr<nvinfer1::ICudaEngine> engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    

    if (!engine)
    {
        sample::gLogError << "Failed to create engine \n";
        return nullptr;
    }

    sample::gLogInfo << "getNbInputs: " << network->getNbInputs() << " \n" << std::endl;
    sample::gLogInfo << "getNbOutputs: " << network->getNbOutputs() << " \n" << std::endl;
    sample::gLogInfo << "getOutputs Name: \n" ;
    for (int i=0;i< static_cast<int>(network->getNbOutputs());i++)
        sample::gLogInfo << "name " << i << " : " << network->getOutput(i)->getName() <<std::endl; 

    // mInputDims = network->getInput(0)->getDimensions();
    // mOutputDims = network->getOutput(0)->getDimensions();


    return engine;
}



bool CenterFusion::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser,
    std::string  onnxFilePath)
{   
    auto parsed = parser->parseFromFile(
        // locateFile(m_params_.onnxFileName, m_params_.dataDirs).c_str(),
        // params.onnxFileName.c_str(),
        onnxFilePath.c_str(),
        static_cast<int>(sample::gLogger.getReportableSeverity()));

        // ILogger::Severity::kWARNING);
    if (!parsed)
    {
        sample::gLogError<< "Onnx model cannot be parsed ! " << std::endl;
        return false;
    }
    builder->setMaxBatchSize(BATCH_SIZE_);
    config->setMaxWorkspaceSize(5_GiB); //8_GiB);
    if (m_params_.fp16)
        config->setFlag(BuilderFlag::kFP16);
    if (m_params_.dlaCore >=0 ){
    samplesCommon::enableDLA(builder.get(), config.get(), m_params_.dlaCore);
    sample::gLogInfo << "Deep Learning Acclerator (DLA) was enabled . \n";
    }
    return true;
}


//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!


bool CenterFusion::infer()
{
    // Create RAII buffer manager object
    sample::gLogInfo << "Creating detection context " << std::endl;
    samplesCommon::BufferManager buffers_detection(m_engine_detection_);
    auto context_detection = SampleUniquePtr<nvinfer1::IExecutionContext>(m_engine_detection_->createExecutionContext());

    sample::gLogInfo << "Creating fusion context " << std::endl;
    samplesCommon::BufferManager buffers_fusion(m_engine_fusion_);
    auto context_fusion = SampleUniquePtr<nvinfer1::IExecutionContext>(m_engine_fusion_->createExecutionContext());

    if (!context_detection || !context_fusion)
    {
        sample::gLogError<< "Failed to create context " << std::endl;
        return false;
    }

    dev_img_ = static_cast<float*>(buffers_detection.getDeviceBuffer("img"));
    dev_feat_ = static_cast<float*>(buffers_fusion.getDeviceBuffer("feat"));
    dev_pc_hm_ = static_cast<float*>(buffers_fusion.getDeviceBuffer("pc_dep"));


    void* inputImgBuf = nullptr;
    void* inputPcBuf = nullptr;
    void* inputCalibBuf = nullptr;
    void* inputPcNumBuf = nullptr;


    // create event  object , which are used time computing
    cudaEvent_t start, stop;
    float preprocess_time = 0;
    float detection_time = 0;
    float frustum_association_time = 0 ;
    float merge_time = 0;
    float fusion_time  = 0;
    float post_time = 0;
    
    float totalPrepDur = 0;
    float totalDetectionDur = 0;
    float totalAssocDur = 0;
    float totalMergeDur = 0;
    float totalFusionDur =0 ;
    float totalPostDur = 0;

    std::vector<std::string> imgPaths = glob(m_params_.filePaths[0]+"/images/*bin");
    std::vector<std::string> calibPaths = glob(m_params_.filePaths[0]+"/calibs/*bin");
    std::vector<std::string> pc3dPaths = glob(m_params_.filePaths[0]+"/pc_3ds/*bin");
    int fileSize = imgPaths.size();

    if (!fileSize) {
        sample::gLogError<< "No Bin File Was Found ! " << std::endl;
        return false;
    }



    // For Loop Every Pcl Bin 
     for(auto idx = 0; idx < fileSize; idx++){
        sample::gLogInfo << "===========FilePath[" << idx <<"/"<<fileSize<<"]:" << imgPaths[idx] <<"=============="<< std::endl;
        // if (idx<164) continue;
        // 0 :calib, 1: img, 2: pc_3d, 3: pc_dep
        // if (idx<30) continue;

        int32_t img_size, pc_dep_size;
        std::string tmp_pc_num_path = m_params_.filePaths[0]+"/data_num.bin";
        // if (!processInput(inputImgBuf, m_params_.filePaths[idx + fileSize], img_size) 
        //     || !readPC(inputPcBuf, m_params_.filePaths[idx + fileSize * 2], img_size )
        //     || !readPcNum( inputPcNumBuf, tmp_pc_num_path)
        //     || !readPcNum( inputCalibBuf,  m_params_.filePaths[idx]))
        if (!processInput(inputImgBuf, imgPaths[idx], img_size) 
            || !readPC(inputPcBuf, pc3dPaths[idx], img_size )
            || !readPcNum( inputPcNumBuf, tmp_pc_num_path)
            || !readPcNum( inputCalibBuf,  calibPaths[idx]))

            // ||!processInput(inputPcBuf, m_params_.filePaths[idx + fileSize * 2], pc_dep_size))
            {sample::gLogError << "Read File Error! " << std::endl;
                return false;}

         // Specially for waymo 4 dim 
        float* img = static_cast<float*>(inputImgBuf);
        float* pc_raw = static_cast<float*>(inputPcBuf);
        float* pc_num = static_cast<float*>( inputPcNumBuf);
        float* img_calib = static_cast<float*>(inputCalibBuf);
        
        tmpCalibPrep(img_calib, dev_feat_calib_, dev_feat_invcalib_);


        pc_dep_size = pc_num[idx];
        sample::gLogInfo<< "Radar Point Num : " << pc_dep_size << std::endl;

        GPU_CHECK(cudaMemcpy(dev_img_, img,  INPUT_H * INPUT_W * 3 * sizeof(float), cudaMemcpyHostToDevice));
        
        float pc_raw_trans[pc_dep_size * 3];
        float pc_vx[pc_dep_size];
        float pc_vz[pc_dep_size];
        for (int i=0;i<pc_dep_size;i++)
        {
            pc_raw_trans[i*3] = pc_raw[i];
            pc_raw_trans[i*3+1] = pc_raw[i+1000];
            pc_raw_trans[i*3+2] = pc_raw[i+2000];
            pc_vx[i] = pc_raw[i+3000];
            pc_vz[i] = pc_raw[i+4000];
        }

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        preprocess(pc_raw_trans, pc_vx, pc_vz, pc_dep_size, dev_pc_dep_, dev_feat_calib_, dev_feat_invcalib_);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&preprocess_time, start, stop);        

        std::vector<Box> predResult;
        predResult.clear();


        // Doing inference 
        cudaEventRecord(start);
        bool status = context_detection->executeV2(buffers_detection.getDeviceBindings().data());
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&detection_time, start, stop); 
        if (!status)
        {
            sample::gLogError<< "Error with fusion contex execution ! " << std::endl;
            return false;
        }


        // now generate pc_hm, make frustum association 
        cudaEventRecord(start);
        generate_pc_hm( buffers_detection,
                    dev_pc_dep_, dev_pc_hm_,
                    dev_score_idx_, dev_feat_calib_, dev_feat_invcalib_);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&frustum_association_time, start, stop); 
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // void* inputPcDepBuf = nullptr;
        // std::cout <<  m_params_.filePaths[idx + fileSize * 3] << std::endl;
        // processInput(inputPcDepBuf, m_params_.filePaths[idx + fileSize * 3], pc_dep_size);
        // float*  host_pc_dep = static_cast<float*>(inputPcDepBuf);
        // GPU_CHECK(cudaMemcpy(dev_pc_hm_, host_pc_dep, OUTPUT_H * OUTPUT_W * 3 * sizeof(float), cudaMemcpyHostToDevice));

        // float tmp_pc_dep[OUTPUT_H * OUTPUT_W * 3];
        // GPU_CHECK(cudaMemcpy(tmp_pc_dep, dev_pc_hm_,OUTPUT_H * OUTPUT_W * 3 * sizeof(float), cudaMemcpyDeviceToHost));
        // tmpSave(tmp_pc_dep, "tmp_pc_dep.txt", OUTPUT_H * OUTPUT_W * 3, OUTPUT_W);
        // break;
        // cudaMemset(dev_pc_hm_, 0 , OUTPUT_H * OUTPUT_W * 3* sizeof(float));
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        
        //  cast value type on the GPU device 


        cudaEventRecord(start);
        dev_featmap_ = static_cast<float*>(buffers_detection.getDeviceBuffer("feat"));
        do_merge(dev_featmap_, dev_pc_hm_, dev_feat_);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&merge_time, start, stop);

        cudaEventRecord(start);
        status = context_fusion->executeV2( buffers_fusion.getDeviceBindings().data());
        if (!status)
        {
            sample::gLogError<< "Error with fusion contex execution ! " << std::endl;
            return false;
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&fusion_time, start, stop);
        cudaEventRecord(start);

         // TODO : now take some time to write your post-process function 
        
        postprocessFusGpu(buffers_detection, buffers_fusion, predResult, 
                                                dev_score_idx_,
                                                mask_cpu,
                                                remv_cpu,
                                                host_score_idx_,
                                                host_keep_data_,
                                                host_boxes_,
                                                host_label_,
                                                m_params_.score_thre,
                                                m_params_.nms_thre,
                                                dev_feat_calib_,
                                                dev_feat_invcalib_
                                                );

        // postprocessGpu(buffers_detection,  predResult, 
        //                                         dev_score_idx_,
        //                                         mask_cpu,
        //                                         remv_cpu,
        //                                         host_score_idx_,
        //                                         host_keep_data_,
        //                                         host_boxes_,
        //                                         host_label_,
        //                                         m_params_.score_thre,
        //                                         m_params_.nms_thre,
        //                                         dev_feat_calib_,
        //                                         dev_feat_invcalib_
        //                                         );

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&post_time, start, stop);

        totalPrepDur += preprocess_time ;
        totalDetectionDur += detection_time ;
        totalAssocDur += frustum_association_time;
        totalMergeDur += merge_time ;
        totalFusionDur += fusion_time ;
        totalPostDur += post_time ;

        saveOutput(predResult, imgPaths[idx], m_params_.savePath);
        

        free(img);
        // free(pc_dep);
        free(pc_raw);
        free(pc_num);

    }
    sample::gLogInfo << "Average PreProcess Time: " << totalPrepDur /fileSize << " ms"<< std::endl;
    sample::gLogInfo << "Average DetectionInfer Time: " << totalDetectionDur /fileSize << " ms"<< std::endl;
    sample::gLogInfo << "Average FrustumAssoc Time: " << totalAssocDur /fileSize << " ms"<< std::endl;
    sample::gLogInfo << "Average merge Time: " << totalMergeDur /fileSize << " ms"<< std::endl;
    sample::gLogInfo << "Average FusInfer  Time: " << totalFusionDur /fileSize << " ms"<< std::endl;
    sample::gLogInfo << "Average PostProcess Time: " << totalPostDur /  fileSize<< " ms"<< std::endl;

    return true;
}

/* There is a bug. 
 * If I change void to bool, the "for (size_t idx = 0; idx < m_engine_detection_->getNbBindings(); idx++)" loop will not stop.
 */

void CenterFusion::saveOutput(std::vector<Box>& predResult, std::string& inputFileName,  std::string savePath)
{
    
    std::string::size_type pos = inputFileName.find_last_of("/");
    std::string outputFilePath = savePath + "/" +  inputFileName.substr(pos) + ".txt";


    ofstream resultFile;

    resultFile.exceptions ( std::ifstream::failbit | std::ifstream::badbit );
    try {
        resultFile.open(outputFilePath);
        for (size_t idx = 0; idx < predResult.size(); idx++){
                resultFile << predResult[idx].x << " " << predResult[idx].y << " " << predResult[idx].z << " "<< \
                predResult[idx].dx << " " << predResult[idx].dy << " " << predResult[idx].dz << " " << predResult[idx].velX \
                << " " << predResult[idx].velY << " " << predResult[idx].theta << " " << predResult[idx].score << \ 
                " "<< predResult[idx].cls << std::endl;
        }
        resultFile.close();
    }
    catch (std::ifstream::failure e) {
        sample::gLogError << "Open File: " << outputFilePath << " Falied"<< std::endl;
    }
}


//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool CenterFusion::processInput(void*& inputPointBuf, std::string& pointFilePath, int& pointNum)
{

    bool ret = readBinFile(pointFilePath, inputPointBuf, pointNum,  1);
    std::cout << "Success to read and Point Num  Is: " << pointNum << std::endl;
    if(!ret){
        sample::gLogError << "Error read point file: " << pointFilePath<< std::endl;
        free(inputPointBuf);
        return ret;
    }
    return ret;
}
