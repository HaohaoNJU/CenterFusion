/*
3D iou, bev iou, max pool iou
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
#include <config.h>

#include "postprocess.h"
#define THREADS_PER_BLOCK 16
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
// int THREADS_PER_BLOCK_NMS =  sizeof(unsigned long long) * 8
// #define DEBUG
const float EPS = 1e-8;

struct Point {
    float x, y;
    __device__ Point() {}
    __device__ Point(double _x, double _y){
        x = _x, y = _y;
    }

    __device__ void set(float _x, float _y){
        x = _x; y = _y;
    }

    __device__ Point operator +(const Point &b)const{
        return Point(x + b.x, y + b.y);
    }

    __device__ Point operator -(const Point &b)const{
        return Point(x - b.x, y - b.y);
    }
};

__device__ inline float cross(const Point &a, const Point &b){
    return a.x * b.y - a.y * b.x;
}

__device__ inline float cross(const Point &p1, const Point &p2, const Point &p0){
    return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}

__device__ int check_rect_cross(const Point &p1, const Point &p2, const Point &q1, const Point &q2){
    int ret = min(p1.x,p2.x) <= max(q1.x,q2.x)  &&
              min(q1.x,q2.x) <= max(p1.x,p2.x) &&
              min(p1.y,p2.y) <= max(q1.y,q2.y) &&
              min(q1.y,q2.y) <= max(p1.y,p2.y);
    return ret;
}

__device__ inline int check_in_box2d(const float *box, const Point &p){
    //params: (7) [x, y, z, dx, dy, dz, heading]
    const float MARGIN = 1e-2;

    float center_x = box[0], center_y = box[1];
    float angle_cos = cos(-box[6]), angle_sin = sin(-box[6]);  // rotate the point in the opposite direction of box
    float rot_x = (p.x - center_x) * angle_cos + (p.y - center_y) * (-angle_sin);
    float rot_y = (p.x - center_x) * angle_sin + (p.y - center_y) * angle_cos;

    return (fabs(rot_x) < box[3] / 2 + MARGIN && fabs(rot_y) < box[4] / 2 + MARGIN);
}

__device__ inline int intersection(const Point &p1, const Point &p0, const Point &q1, const Point &q0, Point &ans){
    // fast exclusion
    if (check_rect_cross(p0, p1, q0, q1) == 0) return 0;

    // check cross standing
    float s1 = cross(q0, p1, p0);
    float s2 = cross(p1, q1, p0);
    float s3 = cross(p0, q1, q0);
    float s4 = cross(q1, p1, q0);

    if (!(s1 * s2 > 0 && s3 * s4 > 0)) return 0;

    // calculate intersection of two lines
    float s5 = cross(q1, p1, p0);
    if(fabs(s5 - s1) > EPS){
        ans.x = (s5 * q0.x - s1 * q1.x) / (s5 - s1);
        ans.y = (s5 * q0.y - s1 * q1.y) / (s5 - s1);

    }
    else{
        float a0 = p0.y - p1.y, b0 = p1.x - p0.x, c0 = p0.x * p1.y - p1.x * p0.y;
        float a1 = q0.y - q1.y, b1 = q1.x - q0.x, c1 = q0.x * q1.y - q1.x * q0.y;
        float D = a0 * b1 - a1 * b0;

        ans.x = (b0 * c1 - b1 * c0) / D;
        ans.y = (a1 * c0 - a0 * c1) / D;
    }

    return 1;
}

__device__ inline void rotate_around_center(const Point &center, const float angle_cos, const float angle_sin, Point &p){
    float new_x = (p.x - center.x) * angle_cos + (p.y - center.y) * (-angle_sin) + center.x;
    float new_y = (p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y;
    p.set(new_x, new_y);
}

__device__ inline int point_cmp(const Point &a, const Point &b, const Point &center){
    return atan2(a.y - center.y, a.x - center.x) > atan2(b.y - center.y, b.x - center.x);
}

__device__ inline float box_overlap(const float *box_a, const float *box_b){
    // params box_a: [x, y, z, dx, dy, dz, heading]
    // params box_b: [x, y, z, dx, dy, dz, heading]

    float a_angle = box_a[6], b_angle = box_b[6];
    float a_dx_half = box_a[3] / 2, b_dx_half = box_b[3] / 2, a_dy_half = box_a[4] / 2, b_dy_half = box_b[4] / 2;
    float a_x1 = box_a[0] - a_dx_half, a_y1 = box_a[1] - a_dy_half;
    float a_x2 = box_a[0] + a_dx_half, a_y2 = box_a[1] + a_dy_half;
    float b_x1 = box_b[0] - b_dx_half, b_y1 = box_b[1] - b_dy_half;
    float b_x2 = box_b[0] + b_dx_half, b_y2 = box_b[1] + b_dy_half;

    Point center_a(box_a[0], box_a[1]);
    Point center_b(box_b[0], box_b[1]);

    Point box_a_corners[5];
    box_a_corners[0].set(a_x1, a_y1);
    box_a_corners[1].set(a_x2, a_y1);
    box_a_corners[2].set(a_x2, a_y2);
    box_a_corners[3].set(a_x1, a_y2);

    Point box_b_corners[5];
    box_b_corners[0].set(b_x1, b_y1);
    box_b_corners[1].set(b_x2, b_y1);
    box_b_corners[2].set(b_x2, b_y2);
    box_b_corners[3].set(b_x1, b_y2);

    // get oriented corners
    float a_angle_cos = cos(a_angle), a_angle_sin = sin(a_angle);
    float b_angle_cos = cos(b_angle), b_angle_sin = sin(b_angle);

    for (int k = 0; k < 4; k++){

        rotate_around_center(center_a, a_angle_cos, a_angle_sin, box_a_corners[k]);
        rotate_around_center(center_b, b_angle_cos, b_angle_sin, box_b_corners[k]);

    }

    box_a_corners[4] = box_a_corners[0];
    box_b_corners[4] = box_b_corners[0];

    // get intersection of lines
    Point cross_points[16];
    Point poly_center;
    int cnt = 0, flag = 0;

    poly_center.set(0, 0);
    for (int i = 0; i < 4; i++){
        for (int j = 0; j < 4; j++){
            flag = intersection(box_a_corners[i + 1], box_a_corners[i], box_b_corners[j + 1], box_b_corners[j], cross_points[cnt]);
            if (flag){
                poly_center = poly_center + cross_points[cnt];
                cnt++;
            }
        }
    }

    // check corners
    for (int k = 0; k < 4; k++){
        if (check_in_box2d(box_a, box_b_corners[k])){
            poly_center = poly_center + box_b_corners[k];
            cross_points[cnt] = box_b_corners[k];
            cnt++;

        }
        if (check_in_box2d(box_b, box_a_corners[k])){
            poly_center = poly_center + box_a_corners[k];
            cross_points[cnt] = box_a_corners[k];
            cnt++;
        }
    }

    poly_center.x /= cnt;
    poly_center.y /= cnt;

    // sort the points of polygon
    Point temp;
    for (int j = 0; j < cnt - 1; j++){
        for (int i = 0; i < cnt - j - 1; i++){
            if (point_cmp(cross_points[i], cross_points[i + 1], poly_center)){
                temp = cross_points[i];
                cross_points[i] = cross_points[i + 1];
                cross_points[i + 1] = temp;
            }
        }
    }


    // get the overlap areas
    float area = 0;
    for (int k = 0; k < cnt - 1; k++){
        area += cross(cross_points[k] - cross_points[0], cross_points[k + 1] - cross_points[0]);
    }

    return fabs(area) / 2.0;
}

__device__ inline float iou_bev(const float *box_a, const float *box_b){
    // params box_a: [x, y, z, dx, dy, dz, heading]
    // params box_b: [x, y, z, dx, dy, dz, heading]
    float sa = box_a[3] * box_a[4];
    float sb = box_b[3] * box_b[4];
    float s_overlap = box_overlap(box_a, box_b);
    return s_overlap / fmaxf(sa + sb - s_overlap, EPS);
}
__device__ inline float iou_normal(float const * const a, float const * const b) {
    //params: a: [x, y, z, dx, dy, dz, heading]
    //params: b: [x, y, z, dx, dy, dz, heading]

    float left = fmaxf(a[0] - a[3] / 2, b[0] - b[3] / 2), right = fminf(a[0] + a[3] / 2, b[0] + b[3] / 2);
    float top = fmaxf(a[1] - a[4] / 2, b[1] - b[4] / 2), bottom = fminf(a[1] + a[4] / 2, b[1] + b[4] / 2);
    float width = fmaxf(right - left, 0.f), height = fmaxf(bottom - top, 0.f);
    float interS = width * height;
    float Sa = a[3] * a[4];
    float Sb = b[3] * b[4];
    return interS / fmaxf(Sa + Sb - interS, EPS);
}

__global__ void boxes_overlap_kernel(const int num_a, const float *boxes_a, const int num_b, const float *boxes_b, float *ans_overlap){
    // params boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]
    const int a_idx = blockIdx.y * THREADS_PER_BLOCK + threadIdx.y;
    const int b_idx = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

    if (a_idx >= num_a || b_idx >= num_b){
        return;
    }
    const float * cur_box_a = boxes_a + a_idx * 7;
    const float * cur_box_b = boxes_b + b_idx * 7;
    float s_overlap = box_overlap(cur_box_a, cur_box_b);
    ans_overlap[a_idx * num_b + b_idx] = s_overlap;
}

__global__ void boxes_iou_bev_kernel(const int num_a, const float *boxes_a, const int num_b, const float *boxes_b, float *ans_iou){
    // params boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]
    const int a_idx = blockIdx.y * THREADS_PER_BLOCK + threadIdx.y;
    const int b_idx = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

    if (a_idx >= num_a || b_idx >= num_b){
        return;
    }
    const float * cur_box_a = boxes_a + a_idx * 7;
    const float * cur_box_b = boxes_b + b_idx * 7;
    float cur_iou_bev = iou_bev(cur_box_a, cur_box_b);
    ans_iou[a_idx * num_b + b_idx] = cur_iou_bev;
}





/////////////////////////////////////////////////////////////////////////////////////BEGIN////////////////////////////////////////////////////////////////////////////////////////////
__device__ int ReadAndAdd(int* address, int val)
{
    int old = *address;
    int assumed;
    do {
        assumed = old;
        old = atomicCAS(address, assumed, val + assumed);
    } while (assumed != old);
    return old;
}



// __device__ __forceinline__ float atomicMaxF(float* address, int val)
// {
//     float old;
//     old = (val > 0) ? __int_as_float(atomicMax((int*)address, __float_as_int(val))):
//                         __uint_as_float(atomicMax((unsigned int*)address, __float_as_uint(val)));
//     return old;
// }

__global__ void max_pool_nms_kernel(const float* score, int* score_indexs, int* cur_idx, const float score_thre)
{
    // int conv_idx = blockIdx.y * blockDim.x + blockIdx.x;
    // // int conv_xIdx = conv_idx % NMS_KERNEL_SIZE;
    // // int conv_yIdx = conv_idx / NMS_KERNEL_SIZE;
    // int conv_xIdx = blockDim.x;
    // int conv_yIdx = blockDim.y;

    int conv_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int conv_xIdx = threadIdx.x;
    int conv_yIdx = blockIdx.x;
    
    int pad_size = (NMS_KERNEL_SIZE-1) /2;

    // cudaMemset(perceptive_field_max_idx, 0, 1 * sizeof(int32_t));

    float max_v = score[conv_idx];
    bool center_max_flag = false;
    if (max_v >= score_thre)
    {   
        center_max_flag = true;
        for (int i=0;i<NMS_KERNEL_SIZE * NMS_KERNEL_SIZE;i++)
        {
            int offset_x = i % NMS_KERNEL_SIZE - pad_size;
            int offset_y = i / NMS_KERNEL_SIZE - pad_size;
            int cur_pixel_idx = conv_idx + offset_y * OUTPUT_W + offset_x;
            int cur_pixel_xIdx = conv_xIdx + offset_x;
            int cur_pixel_yIdx = conv_yIdx + offset_y;

            if (cur_pixel_xIdx>=0 && 
                cur_pixel_xIdx < OUTPUT_W && 
                cur_pixel_yIdx>=0 && 
                cur_pixel_yIdx< OUTPUT_H)
            {   
                if (score[cur_pixel_idx] > max_v)
                {
                    center_max_flag = false;
                    break;
                }
            }
            else continue;
        }
    }
    
    if (center_max_flag)
    {
        int idx_ = ReadAndAdd(cur_idx, 1);
        __threadfence();
        score_indexs[idx_] = conv_idx;
    }
}



__global__ void head_decode_kernel(const int boxes_num, const int* indexs,
                            float *reg, const float* depth,  float* dim, float* rot, 
                            const float* calib, const float* inv_calib)
{
    // CenterFusion DIM (h,w,l) , default rotationc along image x coord, postive rotation direction is clock-wise from BEV perspective . 
    // default heading of obj towarding to X. 
    //    / Z
    //   /_____ X
    //   |
    //   |  Y
    
    int32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t spatial_size  = OUTPUT_H * OUTPUT_W;

    __shared__ float block_boxes[THREADS_PER_BLOCK_NMS * 7];

    if (thread_idx < boxes_num) 
    {
        const int cur_idx = indexs[thread_idx];
        const float xIdx = cur_idx % OUTPUT_W ;
        const float yIdx = cur_idx / OUTPUT_W ;

        // reformat dim as (dx,dy,dz) from initial heading . 
        float h = dim[cur_idx + spatial_size * 0]; 

        //decode center point of bbox
        float u = reg[cur_idx] + xIdx;
        float v = reg[spatial_size + cur_idx] + yIdx;
        float Z = depth[cur_idx];
        reg[cur_idx] = Z * (u * inv_calib[0]  + inv_calib[2]);
        reg[cur_idx + spatial_size] = Z * (v * inv_calib[4]  + inv_calib[5]); // this is specially defined by centerfusion
        

        // decode roation angle 
        float alpha = 0;
        if (rot[1 * spatial_size + cur_idx]>rot[5*spatial_size+cur_idx])
        {
            alpha = atan2f(rot[2*spatial_size + cur_idx], rot[3*spatial_size + cur_idx]);
            alpha -= 0.5 * PI;
        }
        else
        {
            alpha = atan2f(rot[6*spatial_size + cur_idx], rot[7*spatial_size + cur_idx]);
            alpha += 0.5 * PI;
        }
        alpha += atan2f(reg[cur_idx],Z); // TODO : defined by centerfusion. decoded yaw is added to center point yaw angle. 
        rot[cur_idx] = alpha;
    }
}

__global__ void raw_nms_kernel(const int boxes_num, const float nms_overlap_thresh,
                            const float* reg, const float* depth, const float* dim, const float* rot, 
                            const int* indexs,  unsigned long long *mask)
{

    const int row_start = blockIdx.y;
    const int col_start = blockIdx.x;
    const int32_t spatial_size  = OUTPUT_H * OUTPUT_W;

    // at least how many threads left after current block of threads . 
    const int row_size = fminf(boxes_num - row_start * THREADS_PER_BLOCK_NMS, THREADS_PER_BLOCK_NMS);
    const int col_size = fminf(boxes_num - col_start * THREADS_PER_BLOCK_NMS, THREADS_PER_BLOCK_NMS);

    __shared__ float block_boxes[THREADS_PER_BLOCK_NMS * 7];

    if (threadIdx.x < col_size) {
        const int col_actual_idx = indexs[THREADS_PER_BLOCK_NMS * col_start + threadIdx.x];
        // assigning to shared boxes, convert to kitty coord
        // default heading of obj towarding to X. 
        //        Z|  /X 
        //         | / 
        //  Y______|/
        //       

        block_boxes[threadIdx.x * 7 + 0] = depth[col_actual_idx];
        block_boxes[threadIdx.x * 7 + 1] = 0.0 - reg[col_actual_idx];
        block_boxes[threadIdx.x * 7 + 2] = 0.0 - reg[col_actual_idx+spatial_size];

        block_boxes[threadIdx.x * 7 + 3] = dim[col_actual_idx + spatial_size * 2];
        block_boxes[threadIdx.x * 7 + 4] = dim[col_actual_idx + spatial_size * 1];
        block_boxes[threadIdx.x * 7 + 5] = dim[col_actual_idx + spatial_size * 0];

        block_boxes[threadIdx.x * 7 + 6] = 0.0 - rot[col_actual_idx] - PI/2;
    }

    __syncthreads();

    if (threadIdx.x < row_size) {
        const int row_actual_idx = indexs[THREADS_PER_BLOCK_NMS * row_start + threadIdx.x];
        const int cur_box_idx = THREADS_PER_BLOCK_NMS * row_start + threadIdx.x;
        const int xIdx = row_actual_idx % OUTPUT_W;
        const int yIdx = row_actual_idx / OUTPUT_W;

        //encode boxs according waymo  format : (N, 7) [x, y, z, dx, dy, dz, heading]
        float cur_box[7];
        cur_box[0] = depth[row_actual_idx];
        cur_box[1] = 0.0 - reg[row_actual_idx];
        cur_box[2] = 0.0 - reg[row_actual_idx + spatial_size];

        cur_box[3] = dim[row_actual_idx + spatial_size * 2];
        cur_box[4] = dim[row_actual_idx + spatial_size];
        cur_box[5] = dim[row_actual_idx];

        cur_box[6] = 0.0 - rot[row_actual_idx] - PI/2;

        // const float *cur_box = boxes + cur_box_idx * 7;

        int i = 0;
        unsigned long long t = 0;
        int start = 0;
        if (row_start == col_start) {
          start = threadIdx.x + 1;
        }
        for (i = start; i < col_size; i++) {
            if (iou_bev(cur_box, block_boxes + i * 7) > nms_overlap_thresh){
                t |= 1ULL << i;
            }
        }

        const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);
        // assume cur_box_idx = 21, col_start = 0, row_start = 0 , threadIdx = 21, mark 21 th box and top 64 boxes 
        mask[cur_box_idx * col_blocks + col_start] = t; 
    }
}





// compute how many scores are valid 
struct is_greater{
    is_greater(float thre) : _thre(thre) { }
    __host__ __device__ 
    bool operator()(const float &x) {
        return x>= _thre;
    }
    float _thre;
};
struct is_odd
{
    __host__ __device__ 
    bool operator()(const int &x) 
    {
        return true ;
    }
};



__global__ void _find_valid_score_numKernel_(float* score, float* thre, float* N) 
{
    int yIdx = blockIdx.x;
    int xIdx = threadIdx.x;
    int idx = yIdx * blockDim.x + xIdx;
    if (score[idx] >= 0.1) 
     atomicAdd(N, 1.0);
}



int _find_valid_score_num(float* score, float thre, int output_h, int output_w) 
{
    // thrust::device_vector<float> score_vec(score,score + output_h * output_w);
    return thrust::count_if(thrust::device, score, score + output_h * output_w,  is_greater(thre));
    // return thrust::count_if(thrust::device, score_vec.begin(),score_vec.end(),is_greater(thre));
}




void _sort_by_key(float* keys, int* values,int size) {

        thrust::sequence(thrust::device, values, values+size);
        // size = OUTPUT_H * OUTPUT_W;
        thrust::sort_by_key(thrust::device, keys, keys + size,   values,  thrust::greater<float>());

}

// for maxpool nms 
void _gather_all(float* host_boxes, int* host_label, 
                float* reg, float* depth, float* dim, float* rot,  float* score, int32_t* label,  
                int* dev_indexs, int boxSizeAft) 
{

    thrust::device_vector<int> dev_indexs_aft(dev_indexs, dev_indexs + boxSizeAft);

    // gather boxes, score, label
    thrust::device_vector<float> tmp_boxes(boxSizeAft * 8);
    thrust::device_vector<int> tmp_label(boxSizeAft);
    // gather x, y 
    thrust::device_vector<float> reg_vec(reg,reg+OUTPUT_H * OUTPUT_W * 2);
    thrust::gather(dev_indexs_aft.begin(),dev_indexs_aft.end(), reg_vec.begin(),tmp_boxes.begin());
    thrust::gather(dev_indexs_aft.begin(), dev_indexs_aft.end(), reg_vec.begin() + OUTPUT_W * OUTPUT_H, tmp_boxes.begin() + boxSizeAft);
    // gather depth 
    thrust::device_vector<float> depth_vec(depth, depth + OUTPUT_H * OUTPUT_W);
    thrust::gather(dev_indexs_aft.begin(),dev_indexs_aft.end(), depth_vec.begin(),tmp_boxes.begin() + 2 * boxSizeAft);
    // gather  dim
    thrust::device_vector<float> dim_vec(dim, dim + 3 * OUTPUT_H * OUTPUT_W);
    thrust::gather(dev_indexs_aft.begin(),dev_indexs_aft.end(), dim_vec.begin() + OUTPUT_W * OUTPUT_H * 0,tmp_boxes.begin() + 3 * boxSizeAft);
    thrust::gather(dev_indexs_aft.begin(),dev_indexs_aft.end(), dim_vec.begin() + OUTPUT_W * OUTPUT_H * 1,tmp_boxes.begin() + 4 * boxSizeAft);
    thrust::gather(dev_indexs_aft.begin(),dev_indexs_aft.end(), dim_vec.begin() + OUTPUT_W * OUTPUT_H * 2,tmp_boxes.begin() + 5 * boxSizeAft);
    // gather rotation
    thrust::device_vector<float> rot_vec(rot, rot + 1 * OUTPUT_H * OUTPUT_W);
    thrust::gather(dev_indexs_aft.begin(),dev_indexs_aft.end(), rot_vec.begin(),tmp_boxes.begin() + 6 * boxSizeAft);
    // gather score
    thrust::device_vector<float> score_vec(score, score + 1 * OUTPUT_H * OUTPUT_W);
    thrust::gather(dev_indexs_aft.begin(),dev_indexs_aft.end(), score_vec.begin(),tmp_boxes.begin() + 7 * boxSizeAft);
    // gather label
    thrust::device_vector<int> label_vec(label, label + 1 * OUTPUT_H * OUTPUT_W);
    thrust::gather(dev_indexs_aft.begin(),dev_indexs_aft.end(), label_vec.begin() + OUTPUT_W * OUTPUT_H * 0, tmp_label.begin());
    
    thrust::copy(tmp_boxes.begin(), tmp_boxes.end(), host_boxes);
    thrust::copy(tmp_label.begin(),tmp_label.end(), host_label);
}


// for bev nms 
void _gather_all(float* host_boxes, int* host_label, 
                float* reg, float* depth, float* dim, float* rot,  float* sorted_score, int32_t* label,  
                int* dev_indexs, long* host_keep_indexs,  int boxSizeBef, int boxSizeAft) 
{

    // copy keep_indexs from host to device
    // int* tmp_keep_indexs = static_cast<int*>(host_keep_indexs);
    thrust::device_vector<long> dev_keep_indexs(host_keep_indexs, host_keep_indexs + boxSizeAft);
    // thrust::host_vector<long> host_keep_indexs_vec(host_keep_indexs,host_keep_indexs+boxSizeAft);
    // // thrust::copy(host_keep_indexs,host_keep_indexs+boxSizeAft, dev_keep_indexs.begin());
    // thrust::copy(host_keep_indexs_vec.begin(), host_keep_indexs_vec.end(), dev_keep_indexs.begin());
    // gather keeped indexs after nms
    thrust::device_vector<int> dev_indexs_bef(dev_indexs, dev_indexs + boxSizeBef);
    thrust::device_vector<int> dev_indexs_aft(boxSizeAft);
    thrust::gather(dev_keep_indexs.begin(), dev_keep_indexs.end(),
                                  dev_indexs_bef.begin(),
                                  dev_indexs_aft.begin());
    // gather boxes, score, label
    thrust::device_vector<float> tmp_boxes(boxSizeAft * 8);
    thrust::device_vector<int> tmp_label(boxSizeAft);
    // gather x, y 
    thrust::device_vector<float> reg_vec(reg,reg+OUTPUT_H * OUTPUT_W * 2);
    thrust::gather(dev_indexs_aft.begin(),dev_indexs_aft.end(), reg_vec.begin(),tmp_boxes.begin());
    thrust::gather(dev_indexs_aft.begin(), dev_indexs_aft.end(), reg_vec.begin() + OUTPUT_W * OUTPUT_H, tmp_boxes.begin() + boxSizeAft);
    // gather depth 
    thrust::device_vector<float> depth_vec(depth, depth + OUTPUT_H * OUTPUT_W);
    thrust::gather(dev_indexs_aft.begin(),dev_indexs_aft.end(), depth_vec.begin(),tmp_boxes.begin() + 2 * boxSizeAft);
    // gather  dim
    thrust::device_vector<float> dim_vec(dim, dim + 3 * OUTPUT_H * OUTPUT_W);
    thrust::gather(dev_indexs_aft.begin(),dev_indexs_aft.end(), dim_vec.begin() + OUTPUT_W * OUTPUT_H * 0,tmp_boxes.begin() + 3 * boxSizeAft);
    thrust::gather(dev_indexs_aft.begin(),dev_indexs_aft.end(), dim_vec.begin() + OUTPUT_W * OUTPUT_H * 1,tmp_boxes.begin() + 4 * boxSizeAft);
    thrust::gather(dev_indexs_aft.begin(),dev_indexs_aft.end(), dim_vec.begin() + OUTPUT_W * OUTPUT_H * 2,tmp_boxes.begin() + 5 * boxSizeAft);
    // gather rotation
    thrust::device_vector<float> rot_vec(rot, rot + 1 * OUTPUT_H * OUTPUT_W);
    thrust::gather(dev_indexs_aft.begin(),dev_indexs_aft.end(), rot_vec.begin() + OUTPUT_W * OUTPUT_H * 0,tmp_boxes.begin() + 6 * boxSizeAft);
    // gather score
    thrust::device_vector<float> sorted_score_vec(sorted_score, sorted_score + 1 * OUTPUT_H * OUTPUT_W);
    thrust::gather(dev_keep_indexs.begin(),dev_keep_indexs.end(), sorted_score_vec.begin(),tmp_boxes.begin() + 7 * boxSizeAft);
    // gather label
    thrust::device_vector<int> label_vec(label, label + 1 * OUTPUT_H * OUTPUT_W);
    thrust::gather(dev_indexs_aft.begin(),dev_indexs_aft.end(), label_vec.begin() + OUTPUT_W * OUTPUT_H * 0, tmp_label.begin());

    // copy values from device => host 
    // host_boxes = tmp_boxes;
    // host_label = tmp_label;
    
    thrust::copy(tmp_boxes.begin(), tmp_boxes.end(), host_boxes);
    thrust::copy(tmp_label.begin(),tmp_label.end(), host_label);


}



///////////////////////////////////////////////////////////////////////////////////END//////////////////////////////////////////////////////////////////////////////////////////


void rawDecode(const int boxes_num, const int* indexs,
                            float *reg, const float* depth,  float* dim, float* rot, 
                            const float* calib, const float* inv_calib)
{   
    int32_t threads = 32;
    int32_t blocks  = DIVUP(boxes_num, threads);
    head_decode_kernel<<<blocks, threads>>> (boxes_num, indexs, reg, depth, dim, rot, calib, inv_calib);
}


void rawNmsLauncher(const float *reg, const float* depth, const float* dim, const float* rot, 
            const int* indexs, unsigned long long * mask, int boxes_num, float nms_overlap_thresh)
{
    dim3 blocks(DIVUP(boxes_num, THREADS_PER_BLOCK_NMS),
                DIVUP(boxes_num, THREADS_PER_BLOCK_NMS));
    dim3 threads(THREADS_PER_BLOCK_NMS);
    raw_nms_kernel<<<blocks, threads>>>(boxes_num, nms_overlap_thresh, reg,depth,dim,rot, indexs, mask);
}



int _raw_maxpool_nms(const float* score, float* reg,  const float* depth, float* dim , float* rot,
                    int* score_indexs,  const float score_thre, const float* calib, const float* inv_calib)
{
    int32_t kernel_volume = NMS_KERNEL_SIZE * NMS_KERNEL_SIZE;
    // dim3 threads(1024/kernel_volume, kernel_volume);
    // dim3 blocks(DIVUP(OUTPUT_H * OUTPUT_W, 1024/kernel_volume));
    // dim3 blocks(OUTPUT_W,OUTPUT_H); // x, y

    int* cur_idx;
    cudaMalloc((void**)&cur_idx, 1 * sizeof(int));
    cudaMemset(cur_idx, 0 , sizeof(int));
    
    // cudaMalloc((void**)&mask_data, boxes_num * col_blocks * sizeof(unsigned long long));
    // max_pool_nms_kernel<<<blocks, NMS_KERNEL_SIZE * NMS_KERNEL_SIZE>>>(score, score_indexs, cur_idx,score_thre);
    
    max_pool_nms_kernel<<<OUTPUT_H, OUTPUT_W>>>(score, score_indexs, cur_idx,score_thre);
    int num_boxes=0;
    cudaMemcpy(&num_boxes,cur_idx, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(cur_idx);

    rawDecode(num_boxes, score_indexs, reg, depth, dim, rot, calib, inv_calib);
    return num_boxes;

}

int _raw_nms_gpu(float* reg,  const float* depth, float* dim , float* rot,
                const int* indexs, long* host_keep_data,unsigned long long* mask_cpu, unsigned long long* remv_cpu,
                int boxes_num,  float nms_overlap_thresh,
                const float* calib, const float* inv_calib)
{
    // params boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params keep: (N)

    // int boxes_num = boxes.size(0);
    // const float * boxes_data = boxes.data<float>();
    // long * keep_data = keep.data<long>();

    rawDecode(boxes_num, indexs, reg, depth, dim, rot, calib, inv_calib);
    const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);

    unsigned long long *mask_data = NULL;
    cudaMalloc((void**)&mask_data, boxes_num * col_blocks * sizeof(unsigned long long));
    rawNmsLauncher(reg, depth, dim, rot, indexs, mask_data, boxes_num, nms_overlap_thresh);

    // unsigned long long mask_cpu[boxes_num * col_blocks];
    // unsigned long long *mask_cpu = new unsigned long long [boxes_num * col_blocks];
    // std::vector<unsigned long long> mask_cpu(boxes_num * col_blocks);

    //    printf("boxes_num=%d, col_blocks=%d\n", boxes_num, col_blocks);
    cudaMemcpy(mask_cpu, mask_data, boxes_num * col_blocks * sizeof(unsigned long long),
                           cudaMemcpyDeviceToHost);

    // TODO : CUT HERE ! ! !
    cudaFree(mask_data);

    // unsigned long long remv_cpu[col_blocks];
    // memset(remv_cpu, 0, col_blocks * sizeof(unsigned long long));

    memset(remv_cpu, 0 , col_blocks * sizeof(unsigned long long ));
    int num_to_keep = 0;

    for (int i = 0; i < boxes_num; i++){
        int nblock = i / THREADS_PER_BLOCK_NMS;
        int inblock = i % THREADS_PER_BLOCK_NMS;

        if (!(remv_cpu[nblock] & (1ULL << inblock))){
            host_keep_data[num_to_keep++] = i;
            for (int j = nblock; j < col_blocks; j++){
                remv_cpu[j] |= mask_cpu[ i * col_blocks + j];
            }
        }
    }

    if ( cudaSuccess != cudaGetLastError() ) printf( "Error with nms !\n" );
    return num_to_keep;
}









