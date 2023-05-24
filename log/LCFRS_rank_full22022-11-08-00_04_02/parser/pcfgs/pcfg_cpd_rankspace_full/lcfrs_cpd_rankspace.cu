#include <stdio.h>
#include <cuda_runtime.h>


__device__ float logsumexpf(float x, float y) {
  if (y <= x)
    return x + log1pf(expf(y - x));
  else
    return y + log1pf(expf(x - y));
}


__global__ void kernel_forward_root( float * root_c,
                                     float * root_d,
                                     float * alpha_l,
                                     float * alpha_r,
                                     float * alpha_cc,
                                     float * alpha_cd,
                                     int B,  int L, int r1, int r2)
{
    int b_idx = blockIdx.x;
    int s_R = threadIdx.x;

    int start = 0;
    int end = L-1;
    float *  s_l = alpha_l + b_idx * L * L * r1;
    float *  s_r = alpha_r + b_idx * L * L * r1;
    float *  s_cc = alpha_cc + b_idx * L * L * r2;
    float *  s_cd = alpha_cd + b_idx * L * L * L * L * r2;

    int L3 = L*L*L*r2;
    int L2 = L*L*r2;
    int L1 = L*r2;

    float tmp_result = logf(0);
    float tmp_result2 = logf(0);
    __shared__ float result[1000];
    __syncthreads();

    for (int split = start+1; split < end; split++)
    {
        tmp_result = logsumexpf(tmp_result, s_l[start*L*r1 + split * r1 + s_R] + s_r[split*L*r1 + end*r1 + s_R]);
        if(s_R < r2){
            for(int split2 = split+1; split2 < end; split2++){
               tmp_result2 = logsumexpf(tmp_result2, s_cc[split*L*r2 + split2*r2 + s_R] + s_cd[start*L*L*L*r2 + split*L*L*r2 + split2*L*r2 + end*r2 + s_R]);
            }
        }
    }

    result[s_R] = tmp_result + root_c[b_idx*r1 + s_R];

    if(s_R < r2){
        result[s_R] = logsumexpf(result[s_R], tmp_result2 + root_d[b_idx*r2 + s_R]);
    }

    __syncthreads();

    int i = 2;
    while(i < r1){
        i*=2;
    }

    for(unsigned int s= i/2; s>0; s>>=1){
        if((s_R < s) & (s_R+ s < r1)){
           result[s_R] = logsumexpf(result[s_R], result[s_R+s]);
        }
        __syncthreads();
    }

    if(s_R==0){
        s_l[start*L*r1 + end*r1 + 0] = result[0];
    }
}


__global__ void kernel_backward_root( float * root_c,
                                     float * root_d,

                                     float * root_c_grd,
                                     float * root_d_grd,

                                     float * alpha_l,
                                     float * alpha_r,
                                     float * alpha_cc,
                                     float * alpha_cd,
                                     int B,  int L, int r1, int r2)
{
    int b_idx = blockIdx.x;
    int s_R = threadIdx.x;

    int start = 0;
    int end = L-1;
    float *  s_l = alpha_l + b_idx * L * L * r1;
    float *  s_r = alpha_r + b_idx * L * L * r1;
    float *  s_cc = alpha_cc + b_idx * L * L * r2;
    float *  s_cd = alpha_cd + b_idx * L * L * L * L * r2;

    int L3 = L*L*L*r2;
    int L2 = L*L*r2;
    int L1 = L*r2;

    float parent_grd = s_l[end*L*r1 + start*r1 + 0];
    float parent_inside = s_l[start*L*r1 + end*r1 + 0];

    float rc_grd = 0;
    float rd_grd = 0;

    for (int split = start+1; split < end; split++)
    {
        float tmp = expf(s_l[start*L*r1 + split * r1 + s_R] + s_r[split*L*r1 + end*r1 + s_R] + root_c[b_idx*r1 + s_R] - parent_inside) * parent_grd;
        atomicAdd(s_l + split*L*r1 + start * r1 + s_R, tmp);
        atomicAdd(s_r + end*L*r1 + split * r1 + s_R, tmp);
        rc_grd += tmp;

        if(s_R < r2){
            for(int split2 = split+1; split2 < end; split2++){
               float tmp = expf(s_cc[split*L*r2 + split2*r2 + s_R] + s_cd[start*L*L*L*r2 + split*L*L*r2 + split2*L*r2 + end*r2 + s_R] + root_d[b_idx*r2 + s_R] - parent_inside) * parent_grd;
               atomicAdd(s_cc + split2*L*r2 + split*r2 + s_R, tmp);
               atomicAdd(s_cd + split*L*L*L*r2 + start*L*L*r2 + split2*L*r2 + end*r2 + s_R, tmp);
               rd_grd += tmp;
            }
        }
    }
    atomicAdd(root_c_grd + b_idx * r1 + s_R, rc_grd);
    if(s_R < r2){
        atomicAdd(root_d_grd + b_idx * r2 + s_R, rd_grd);
    }
}



__global__ void kernel_forward( float *  head_cl,
                                float *  head_cr,
                                float * alpha_l,
                                float * alpha_r,
                                int B, int L, int width, int r)
{
    int b_idx = blockIdx.x;
    int start = blockIdx.y;
    int s_A = blockIdx.z;
    int s_R = threadIdx.x;

    if(start+width >= L){
        return;
    }

    __shared__ float result_l[1000];
    __shared__ float result_r[1000];

    int end = start + width;
    float *  s_l = alpha_l + b_idx * L * L * r1;
    float *  s_r = alpha_r + b_idx * L * L * r1;

    float tmp_result = logf(0);

    for (int split = start+1; split < start+width; split++)
    {
        tmp_result = logsumexpf(tmp_result, s_l[start*L*r1 + split*r1 + s_R] + s_r[split*L*r1 + end*r1 + s_R]);
    }

    result_l[s_R] = tmp_result + head_cl1[b_idx*r1*r1 + s_A*r1 + s_R];
    result_r[s_R] = tmp_result + head_cr1[b_idx*r1*r1 + s_A*r1 + s_R];
    __syncthreads();
    int i = 2;
    while(i < r1){
        i*=2;
    }
    for(unsigned int s= i/2; s>0; s>>=1){
        if((s_R < s) & (s_R+ s < r1)){
           result_l[s_R] = logsumexpf(result_l[s_R+s], result_l[s_R]);
           result_r[s_R] = logsumexpf(result_r[s_R+s], result_r[s_R]);
        }
        __syncthreads();
    }
    if(s_R==0){
        alpha_l[b_idx*L*L*r1 +start*L*r1 +end*r1 + s_A] = result_l[0];
        alpha_r[b_idx*L*L*r1 +start*L*r1 +end*r1 + s_A] = result_r[0];
    }
}



__global__ void kernel_backward( float * head_cl,
                                 float * head_cr,
                                 float * head_cl_grd,
                                 float * head_cr_grd,
                                 float * alpha_l,
                                 float * alpha_r,
                                 int B,   int L,   int width,   int r)
{
      int b_idx = blockIdx.x;
      int start = blockIdx.y;
      int s_A = blockIdx.z;
      int s_R = threadIdx.x;

      if(start+width >= L){
        return;
      }

      int end = start + width;
      float *  s_l = alpha_l + b_idx * L * L * r;
      float *  s_r = alpha_r + b_idx * L * L * r;

      float parent_inside_l = s_l[start * L * r1 + end * r1 + s_A];
      float parent_grd_l = s_l[end * L * r1 + start * r1 + s_A];

      float parent_inside_r = s_r[start * L * r1 + end * r1 + s_A];
      float parent_grd_r = s_r[end * L * r1 + start * r1 + s_A];

      float cl = head_cl[b_idx*r1*r1 + s_A*r1 + s_R];
      float cr = head_cr[b_idx*r1*r1 + s_A*r1 + s_R];

      float cl2_grd = 0;
      float cr2_grd = 0;

      for (int split = start+1; split < start+width; split++)
     {
        float tmp = s_l[start*L*r1 + split * r1 + s_R] + s_r[split*L*r1 + end*r1 + s_R];
        float grd = 0;

        float tmp_grd = expf(tmp + cl1 - parent_inside_l) * parent_grd_l;
        grd += tmp_grd;
        cl1_grd += tmp_grd;

        tmp_grd = expf(tmp + cr1 - parent_inside_r) * parent_grd_r;
        grd += tmp_grd;
        cr1_grd += tmp_grd;

        atomicAdd(s_l + split*L*r1 + start*r1 + s_R, grd);
        atomicAdd(s_r + end*L*r1 + split*r1 + s_R, grd);
       }


     atomicAdd( head_cl_grd + b_idx * r1 * r1 + s_A * r1 + s_R, cl1_grd);
     atomicAdd( head_cr_grd + b_idx * r1 * r1 + s_A * r1 + s_R, cr1_grd);
}




void cuda_forward( float *head_cl1,  float *head_cr1,
                   float *head_cc1,  float *head_dc1,
                   float *head_cl2,  float *head_cr2,
                   float *head_cc2,  float *head_dc2,
                   float *head_cd1, float *head_cd2,
                   float *head_dd1, float *head_dd2,
                   float *root_c, float *root_d,
                   float *alpha_l, float *alpha_r,
                   float *alpha_cc, float *alpha_cd, float *alpha_dc, float *alpha_dd,
                   int B, int L, int r1, int r2)
{
    for(int w=2; w<L-1; w++){
      dim3 gridDim(B, L-w, r1);
      dim3 blockDim(r1);
      kernel_forward<<<gridDim, blockDim>>>(head_cl1, head_cr1, head_cc1, head_dc1, head_cl2, head_cr2, head_cc2, head_dc2, alpha_l, alpha_r, alpha_cc, alpha_cd, alpha_dc,  B, L, w, r1, r2);
    }
    dim3 gridDim3(B);
    dim3 blockDim3(r1);
    kernel_forward_root<<<gridDim3, blockDim3>>>(root_c, root_d, alpha_l, alpha_r, alpha_cc, alpha_cd, B, L, r1, r2);
}

void cuda_backward( float *head_cl1,  float *head_cr1,
                   float *head_cc1,  float *head_dc1,
                   float *head_cl2,  float *head_cr2,
                   float *head_cc2,  float *head_dc2,
                   float *head_cd1, float *head_cd2,
                   float *head_dd1, float *head_dd2,
                   float *root_c, float *root_d,

                   float *head_cl1_grd,  float *head_cr1_grd,
                   float *head_cc1_grd,  float *head_dc1_grd,
                   float *head_cl2_grd,  float *head_cr2_grd,
                   float *head_cc2_grd,  float *head_dc2_grd,
                   float *head_cd1_grd, float *head_cd2_grd,
                   float *head_dd1_grd, float *head_dd2_grd,
                   float *root_c_grd, float *root_d_grd,

                   float *alpha_l, float *alpha_r,
                   float *alpha_cc, float *alpha_cd, float *alpha_dc, float *alpha_dd,
                   int B, int L, int r1, int r2)
{

    dim3 gridDim3(B);
    dim3 blockDim3(r1);
    kernel_backward_root<<<gridDim3, blockDim3>>>(root_c, root_d,
                        root_c_grd, root_d_grd,
                        alpha_l, alpha_r,
                        alpha_cc, alpha_cd,
                        B, L, r1, r2);

    for(int w=L-2; w>1; w--){

       if(w>2){
            dim3 gridDim4(B*(L-w),  (w-1)*(L-w), r2);
            dim3 blockDim4(r2);
            kernel_backward_d2<<<gridDim4, blockDim4>>>(head_cd2, head_cd2_grd, head_dd2, head_dd2_grd, alpha_dc, alpha_dd, alpha_cd, B, L, w, r2, r2);
       }

       dim3 gridDim3(B*(L-w),  (w-1)*(L-w), r2);
       dim3 blockDim3(r1);
       kernel_backward_d1<<<gridDim3, blockDim3>>>(head_cd1, head_cd1_grd, head_dd1, head_dd1_grd, alpha_l, alpha_r, alpha_cd, alpha_dd, B, L, w, r2, r1);

       dim3 gridDim(B, L-w, r1);
       dim3 blockDim(r1);
       kernel_backward<<<gridDim, blockDim>>>(
       head_cl1, head_cr1, head_cc1, head_dc1,
       head_cl2, head_cr2, head_cc2, head_dc2,
       head_cl1_grd, head_cr1_grd, head_cc1_grd, head_dc1_grd,
       head_cl2_grd, head_cr2_grd, head_cc2_grd, head_dc2_grd,
       alpha_l, alpha_r, alpha_cc, alpha_cd, alpha_dc,
       B, L, w, r1, r2);
    }
}