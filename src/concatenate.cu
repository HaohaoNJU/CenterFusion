/*
 * Copyright 2018-2019 Autoware Foundation. All rights reserved.
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

//headers in local files
#include <stdio.h>
#include <iostream>
#include <vector>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/gather.h>
#include <thrust/transform.h>
#include <thrust/count.h>
#include <config.h>
#include <preprocess.h>

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))


__global__ void merge_kernel(const float* feat_in,const float* pc_dep, float* feat_out) 
{
    int pixel_idx = blockIdx.x * blockDim.x +  threadIdx.x;
    int channel_idx = threadIdx.y;
    if (pixel_idx < OUTPUT_H * OUTPUT_W )
    {
        if (channel_idx < FEATMAP_CHANNEL)
            feat_out[channel_idx * (OUTPUT_H * OUTPUT_W) + pixel_idx] = feat_in[channel_idx * (OUTPUT_H * OUTPUT_W) + pixel_idx];
        else
            feat_out[channel_idx * (OUTPUT_H * OUTPUT_W) + pixel_idx] = pc_dep[(channel_idx - FEATMAP_CHANNEL) * (OUTPUT_H * OUTPUT_W) + pixel_idx];
    }
}


// NUM_THREADS_ need to be consistent with channels of pfe output , default is 64
void do_merge(const float* feat_in,const float* pc_dep, float* feat_out)
{   
    int feat_out_channel = FEATMAP_CHANNEL + PC_CHANNEL;
    int num_pixels_per_block = 1024/feat_out_channel;
    
    dim3 threads(num_pixels_per_block, feat_out_channel);
    int blocks = DIVUP( OUTPUT_H * OUTPUT_W, num_pixels_per_block);

    merge_kernel<<<blocks, threads>>>(feat_in, pc_dep, feat_out);
}

