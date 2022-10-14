/*
raw point clout to structured data format 
Written by Wang Hao
All Rights Reserved 2021-2022.
*/

#include <iostream>
#include <stdio.h>
#include <vector>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/gather.h>
#include <thrust/transform.h>
#include <thrust/count.h>
#include <thrust/device_free.h>
#include <config.h>

#include "cublas_v2.h"
#include "preprocess.h"
#define THREADS_PER_BLOCK 16
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
// int THREADS_PER_BLOCK_NMS =  sizeof(unsigned long long) * 8
// #define DEBUG


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// cublas stores array in a column-first manner , however in C++ we stores array row-fisrt, 
// therefore we shouldn make some conversion using cublas internal attributes. 
// gemm fomular : alpha * A * B + beta * C = C
// cublasStatus_t _cublasSgemm(cublasHandle_t handle, 
//                             cublasOperation_t transa, cublasOperation_t transb, // whethre to transpose A or B 
//                             int m, int n, int k,  // output shape C(m,n), input shape A(m, k)
//                             const float alpha, const float *A,  const float *B, // matrix A,B
//                             const float beta, float *C// matrix c
//                             )
// {
//     return cublasSgemm(handle, transa, transb, 
//                       n, m, k,
//                       &alpha, B, n, A, k,
//                       beta, C, n );
// }

// currently cuda doesn't support float point atomicMin & atomicMax computation, so we have to think of another way to fix this issur . 
__device__ __forceinline__ float atomicMaxF(float* address, float val)
{
    float old;
    old = (val > 0) ? __int_as_float(atomicMax((int*)address, __float_as_int(val))):
                        __uint_as_float(atomicMax((unsigned int*)address, __float_as_uint(val)));
    return old;
}

__device__ __forceinline__ float atomicMinF(float* address, float val)
{
    float old;
    old = (val > 0) ? __int_as_float(atomicMin((int*)address, __float_as_int(val))):
                        __uint_as_float(atomicMin((unsigned int*)address, __float_as_uint(val)));
    return old;
}



//    / z
//   /
//  /________  x
//  |
//  |
//  | y 


__global__ void pad_point_kernel( float* xyz, int* tlbr, const int num_point, const float* calib )
{

    int32_t point_idx = blockIdx.x * blockDim.y + threadIdx.y;
    int32_t i = threadIdx.x;
    // printf("size %d , %d \n", blockDim.x, blockDim.y);

    if (point_idx < num_point)
    {

    tlbr[point_idx * 4 + 0] = OUTPUT_H;
    tlbr[point_idx * 4 + 1] = OUTPUT_W;
    tlbr[point_idx * 4 + 2] = 0;
    tlbr[point_idx * 4 + 3] = 0;
    }
    __threadfence();
    

    __shared__ float corners[8 * 3];
    corners[i + 8 * 0] = (i/2 + 1)%2 ?  PILLAR_SIZE_X/2:-PILLAR_SIZE_X/2;
    corners[i + 8 * 1] = (i/4 + 1)%2 ?  0:-PILLAR_SIZE_Y;
    corners[i + 8 * 2] = (i + 1)%2 ?  PILLAR_SIZE_Z/2:-PILLAR_SIZE_Z/2;

    __syncthreads();



    if (point_idx < num_point)
    {
    // xyz shape (point_num, 3)
    float x = xyz[point_idx * 3 + 0];
    float y = xyz[point_idx * 3 + 1];
    float z = xyz[point_idx * 3 + 2];

    float cur_z =  calib[6] * (corners[i]+x) + calib[7] * (corners[i+8]+y) + calib[8] * (z+corners[i+16]);
    float cur_x = (calib[0] * (corners[i]+x) + calib[1] * (corners[i+8]+y) + calib[2] * (z+corners[i+16])) / cur_z;
    float cur_y = (calib[3] * (corners[i]+x) + calib[4] * (corners[i+8]+y) + calib[5] * (z+corners[i+16])) / cur_z;

    cur_x = cur_x < 0 ? 0 : cur_x;
    cur_x = cur_x > OUTPUT_W ? OUTPUT_W : cur_x;
    cur_y = cur_y < 0 ? 0 : cur_y;
    cur_y = cur_y > OUTPUT_H ? OUTPUT_H : cur_y;


    atomicMin(tlbr + point_idx * 4 + 1, __float2int_rn(cur_y));
    atomicMin(tlbr + point_idx * 4 + 1, __float2int_rn(cur_x));
    atomicMax(tlbr + point_idx * 4 + 2, __float2int_rn(cur_y));
    atomicMax(tlbr + point_idx * 4 + 3, __float2int_rn(cur_x));
    }
    // __threadfence();


}




__global__ void pixel_assign_kernel(const float pc_z, const float pc_vx, const float pc_vz, const int top, const int left, float* pc_dep)
{
    int32_t cur_pixel_y = top + blockIdx.x;
    int32_t cur_pixel_x = left + threadIdx.x;
    int32_t cur_pixel_idx = cur_pixel_y  * OUTPUT_W + cur_pixel_x;
    float cur_pixel_depth = pc_dep[cur_pixel_idx + OUTPUT_W * OUTPUT_H * 0];
    float cur_point_depth = pc_z;
    if(cur_pixel_depth == 0 || cur_pixel_depth> cur_point_depth)
    {
        pc_dep[cur_pixel_idx + OUTPUT_W * OUTPUT_H * 0] = cur_point_depth;
        pc_dep[cur_pixel_idx + OUTPUT_W * OUTPUT_H * 1] = pc_vx;
        pc_dep[cur_pixel_idx + OUTPUT_W * OUTPUT_H * 2] = pc_vz;
    }

}



void _raw_pc_encode(const float* pc_3d, const float* pc_vx, const float* pc_vz, 
                   const int num_point, float* pc_dep, 
                   const float* calib,const float* inv_calib)
{
    int* tlbr;
    cudaMalloc((void**)&tlbr, num_point * 4 * sizeof(int));
    // cudaMemset(tlbr, 0 , num_point * 4 * sizeof(int));
    cudaMemset(pc_dep, 0 , OUTPUT_H * OUTPUT_W * 3 * sizeof(float));

    float* dev_pc_3d;
    cudaMalloc((void**)& dev_pc_3d, num_point * 3 * sizeof(float));
    cudaMemcpy(dev_pc_3d, pc_3d, num_point * 3 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(8,32); // x=8, y = 32
    int blocks = DIVUP(num_point, 32);
    
    pad_point_kernel<<<blocks, threads>>>(dev_pc_3d, tlbr, num_point, calib);
    
    int corner2d[ 4 * num_point];
    cudaMemcpy(corner2d, tlbr, num_point * 4 * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_point; i++)
    {
         // get image-2dbox fromc 3d box
        int box_height = corner2d[i * 4 + 2]-corner2d[i * 4 + 0];
        int box_width = corner2d[i * 4 + 3]-corner2d[i * 4 + 1];
        // printf(" height %d, width %d \n", box_height, box_width);
        if (box_height>0 && box_width>0 && pc_3d[i * 3 + 2] >0)
            pixel_assign_kernel<<<box_height, box_width>>>(pc_3d[i * 3 + 2], pc_vx[i], pc_vz[i], corner2d[i * 4 + 0], corner2d[i * 4 + 1], pc_dep);
    }
    cudaFree(dev_pc_3d);
    cudaFree(tlbr);

}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// below is to generate pc_hm, given pc_dep
// TODO : Note, beacuse cuda have no machinesm to make thread-synchronization among blocks, we have to split the kernel into 2 parts .  .^.
__global__ void frustum_association_kernel1(const float* pc_dep, const float* reg, const float* depth, const float* dim, const float* rot, 
                                        float* output, 
                                        const int top, const int left, const int center_pixel_idx,
                                        const float* calib, const float* inv_calib)
{
    // first of all, get distance range of 3d box
    int32_t spatial_size = OUTPUT_W * OUTPUT_H;


    // decode roation angle 
    float alpha = 0;
    if (rot[1 * spatial_size + center_pixel_idx]>rot[5*spatial_size+center_pixel_idx])
    {
        alpha = atan2f(rot[2*spatial_size + center_pixel_idx], rot[3*spatial_size + center_pixel_idx]);
        alpha -= 0.5 * PI;
    }
    else
    {
        alpha = atan2f(rot[6*spatial_size + center_pixel_idx], rot[7*spatial_size + center_pixel_idx]);
        alpha += 0.5 * PI;
    }

    float u = reg[center_pixel_idx] + center_pixel_idx % OUTPUT_W;
    float v = reg[spatial_size + center_pixel_idx] + center_pixel_idx / OUTPUT_W;
    float Z = depth[center_pixel_idx];
    float X = Z * (u * inv_calib[0]  + inv_calib[2]);

    
    alpha += atan2f(X,Z); // TODO : defined by centerfusion. decoded yaw is added to center point yaw angle. 
    
    // compute distance range 
    float cos_alpha = cosf(alpha);
    float sin_alpha = sinf(alpha);
    // dim0 [h,w,l]  ==> [dy, dz, dx] 
    float dz = dim[center_pixel_idx + spatial_size * 1];
    float dx = dim[center_pixel_idx + spatial_size * 2];

    float dz_min = sqrtf( powf(dx,2) + powf(dz,2));

    float dz_max = - dz_min;
    // (c, 0, s)  |x|
    // (0, 1, 0)  |y|
    // (-s, 0, c) |z|, now, only consider z dim, so we get z1 = (-s * x + c * z)
    for(int i=0; i< 4; i++)
    {
        float tmp = -sin_alpha * ((i/2+1) % 2 ? dx : -dx ) + cos_alpha * ((i+1)%2 ? dz : -dz);
        dz_max = tmp > dz_max ? tmp : dz_max;
        dz_min = tmp < dz_min ? tmp : dz_min;
    }    
    float dz_range = max((dz_max-dz_min)/4, 0.0); // half of the min-max distance 
    
    // now find the valid depth 
    output[0] = MAX_DISTANCE;
    output[1] = 0;
    output[2] = 0;
    __threadfence();
    __syncthreads();
    int32_t cur_pixel_idx = (top + blockIdx.x) * OUTPUT_W + left + threadIdx.x;
    float cur_z = pc_dep[cur_pixel_idx];
    // if (cur_z != 0)
    //     printf("pc_dep, %f , dist_range %f, depth %f \n", cur_z, dz_range, Z);
    if (cur_z > 0 && cur_z < MAX_DISTANCE && cur_z <= Z + dz_range && cur_z >= Z - dz_range)
        atomicMinF(output, cur_z);
    __threadfence();
}

__global__ void frustum_association_kernel2(const float* pc_dep, 
                                        float* output,   
                                        const int top, const int left)
{
    int32_t cur_pixel_idx = (top + blockIdx.x) * OUTPUT_W + left + threadIdx.x;
    if (output[0]== pc_dep[cur_pixel_idx])
    {   
        output[0] = output[0] / HM_DEPTH_NORM;
        output[1] = pc_dep[cur_pixel_idx + OUTPUT_W * OUTPUT_H];
        output[2] = pc_dep[cur_pixel_idx + 2 * OUTPUT_W * OUTPUT_H];
    }
}


void _raw_generate_pc_hm(const float* pc_dep, float* pc_hm, 
                        const float* reg, const float* wh, const float* score, 
                        const float* depth,  const float* dim, const float* rot, 
                        int* dev_score_index, const float* calib, const float* inv_calib)
{
    cudaMemset(pc_hm, 0 , OUTPUT_W * OUTPUT_H * 3 * sizeof(float));
    
    // preserve the original score, do not modify its permutation. 
    thrust::device_vector<float>  dev_score(OUTPUT_H * OUTPUT_W);
    thrust::copy(score, score+OUTPUT_H * OUTPUT_W, dev_score.begin());
    
    thrust::sequence(thrust::device, dev_score_index, dev_score_index + OUTPUT_H * OUTPUT_W);
    // size = OUTPUT_H * OUTPUT_W; 
    thrust::sort_by_key(thrust::device, dev_score.begin(), dev_score.end(),   dev_score_index,  thrust::greater<float>());
    
    thrust::device_vector<float> tmp_pack_(4 * TOP_K + 3);
    // float* tmp_pack_;
    // cudaMalloc((void**)&tmp_pack_, 4 * TOP_K * sizeof(float));

    thrust::device_vector<int> dev_score_index_(dev_score_index, dev_score_index+TOP_K);
    thrust::device_vector<float> reg_(reg, reg+ 2 * OUTPUT_H * OUTPUT_W);
    thrust::device_vector<float> wh_(wh, wh+ 2 * OUTPUT_H * OUTPUT_W);

    thrust::gather(dev_score_index_.begin(), dev_score_index_.end(), reg_.begin(), tmp_pack_.begin());
    thrust::gather(dev_score_index_.begin(), dev_score_index_.end(), reg_.begin() + OUTPUT_H * OUTPUT_W, tmp_pack_.begin() + TOP_K);
    thrust::gather(dev_score_index_.begin(), dev_score_index_.end(), wh_.begin() + 0 * OUTPUT_H * OUTPUT_W, tmp_pack_.begin() + TOP_K * 2);
    thrust::gather(dev_score_index_.begin(), dev_score_index_.end(), wh_.begin() + 1 * OUTPUT_H * OUTPUT_W, tmp_pack_.begin() + TOP_K *3);
    float* tmp_pack_ptr = thrust::raw_pointer_cast(&tmp_pack_[0]);
    int host_score_index[TOP_K];
    
    cudaMemcpy(host_score_index, dev_score_index, TOP_K * sizeof(int), cudaMemcpyDeviceToHost);


    for (int i = 0; i < TOP_K; i++)
    {
        float ctr_x = host_score_index[i] % OUTPUT_W;
        float ctr_y = host_score_index[i] / OUTPUT_W;
        
        // ctr_x += tmp_pack[i + TOP_K * 0];
        // ctr_y += tmp_pack[i + TOP_K * 1];
        //// or, only add half of featuremap pixel
        ctr_x += 0.5;
        ctr_y += 0.5;

        int top = static_cast<int>(floorf32(ctr_y - tmp_pack_[i + TOP_K * 3]/2)) ;
        int left = static_cast<int>(floorf32(ctr_x - tmp_pack_[i + TOP_K * 2]/2));
        int bottom = static_cast<int>(ceilf(ctr_y + tmp_pack_[i + TOP_K * 3]/2));
        int right = static_cast<int>(ceilf(ctr_x + tmp_pack_[i + TOP_K * 2]/2));

        top = max(top, 0);
        left = max(left, 0);
        bottom = min(bottom, OUTPUT_H);
        right = min(right, OUTPUT_W);

        if (bottom-top>0 && right-left >0)
        {
            frustum_association_kernel1<<<bottom-top, right-left>>>(pc_dep, reg,  depth,   dim, rot, 
                                        tmp_pack_ptr + TOP_K * 4, 
                                        top, left, host_score_index[i],
                                        calib, inv_calib);

            frustum_association_kernel2<<<bottom-top, right-left>>>(pc_dep, 
                                        tmp_pack_ptr + TOP_K * 4,   
                                        top, left);
        }
        else continue;

        // now assgine the values 
        top = static_cast<int>(floorf32(ctr_y - tmp_pack_[i + TOP_K * 3]/2 * HM_MASK_RATIO )) ;
        left = static_cast<int>(floorf32(ctr_x - tmp_pack_[i + TOP_K * 2]/2 * HM_MASK_RATIO ));
        bottom = static_cast<int>(ceilf(ctr_y + tmp_pack_[i + TOP_K * 3]/2 * HM_MASK_RATIO ));
        right = static_cast<int>(ceilf(ctr_x + tmp_pack_[i + TOP_K * 2]/2 * HM_MASK_RATIO ));
        
        top = max(top, 0);
        left = max(left, 0);
        bottom = min(bottom, OUTPUT_H);
        right = min(right, OUTPUT_W);

        if (bottom-top > 0 && right - left > 0 && tmp_pack_[0 + 4 * TOP_K] < MAX_DISTANCE)

            pixel_assign_kernel<<<bottom-top, right-left>>>( tmp_pack_[0 + 4 * TOP_K],
                                                            tmp_pack_[1 + 4 * TOP_K], 
                                                            tmp_pack_[2 + 4 * TOP_K],  
                                                            top,  left, pc_hm);
    }

    tmp_pack_.clear();
}







