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

    __shared__ float result_l[1000];
    __shared__ float result_r[1000];
    __shared__ float result_c[200];
    __shared__ float result_d1[200];
    __shared__ float result_d2[200];
    __shared__ float result_d3[200];
    __shared__ float result_d4[200];

    int start = 0;
    int end = L;
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


    for (int split = start+1; split < end; split++)
    {
        tmp_result = logsumexpf(tmp_result, s_l[start*L*r1 + split * r1 + s_R] + s_r[split*L*r1 + end*r1 + s_R]);
        if(s_R < r2){
            for(int split2 = split+1; split2 < end; split2++){
               tmp_result2 = logsumexpf(tmp_result2, s_cc[split*L*r2 + split2*r2 + s_R] + s_cd[start*L3 + split*L2 + split2*L1 + s_R]);
            }
        }
    }

    result[s_R] = tmp_result + root_c[b_idx*r1 + s_R];

    if(s_R < r2){
        result[s_R] = logsumexpf(result[s_R], tmp_result2 + root_d[b_idx*r2 + s_R]);
    }
    __syncthreads();

    if(s_R==0){
        float final_result = logf(0);
        for(int i=0;i<r1;i++){
            final_result = logsumexpf(final_result, result[i]);
        }
        s_l[start*L*r1 + end*L*r1 + 0] = final_result;
    }
}




__global__ void kernel_forward( float *  head_cl1,
                                float *  head_cr1,
                                float *  head_cc1,
                                float *  head_dc1,
                                float *  head_cl2,
                                float *  head_cr2,
                                float *  head_cc2,
                                float *  head_dc2,
                                float * alpha_l,
                                float * alpha_r,
                                float * alpha_cc,
                                float * alpha_cd,
                                float * alpha_dc,
                                int B,   int L, int width, int r1, int r2)
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
    __shared__ float result_c[200];
    __shared__ float result_d1[200];
    __shared__ float result_d2[200];
    __shared__ float result_d3[200];
    __shared__ float result_d4[200];

    int end = start + width;
    float *  s_l = alpha_l + b_idx * L * L * r1;
    float *  s_r = alpha_r + b_idx * L * L * r1;
    float *  s_cc = alpha_cc + b_idx * L * L * r2;
    float *  s_cd = alpha_cd + b_idx * L * L * L * L * r2;
    int L3 = L*L*L*r2;
    int L2 = L*L*r2;
    int L1 = L*r2;

    float tmp_result = logf(0);
    float tmp_result2 = logf(0);

    for (int split = start+1; split < start+width; split++)
    {
        tmp_result = logsumexpf(tmp_result, s_l[start*L*r1 + split*r1 + s_R] + s_r[split*L*r1 + end*r1 + s_R]);
        if(s_R < r2){
            for(int split2 = split+1; split2 < start+width; split2++){
               tmp_result2 = logsumexpf(tmp_result2, s_cc[split*L*r2 + split2*r2 + s_R] + s_cd[start*L3 + split*L2 + split2*L1 + s_R]);
            }
        }
    }

    result_l[s_R] = tmp_result + head_cl1[b_idx*r1*r1 + s_A*r1 + s_R];
    result_r[s_R] = tmp_result + head_cr1[b_idx*r1*r1 + s_A*r1 + s_R];

    if (s_A < r2){
       result_c[s_R] = tmp_result + head_cc1[b_idx*r1*r2 + s_A*r1 + s_R];
       result_d1[s_R] = tmp_result + head_dc1[b_idx*r1*r2*4 + s_A*r1*4 + 0*r1 + s_R];
       result_d2[s_R] = tmp_result + head_dc1[b_idx*r1*r2*4 + s_A*r1*4 + 1*r1 + s_R];
       result_d3[s_R] = tmp_result + head_dc1[b_idx*r1*r2*4 + s_A*r1*4 + 2*r1 + s_R];
       result_d4[s_R] = tmp_result + head_dc1[b_idx*r1*r2*4 + s_A*r1*4 + 3*r1 + s_R];
    }

    if(s_R < r2){
        result_l[s_R] = logsumexpf(result_l[s_R], tmp_result2 + head_cl2[b_idx*r1*r2 + s_A*r2 + s_R]);
        result_r[s_R] = logsumexpf(result_r[s_R], tmp_result2 + head_cr2[b_idx*r1*r2 + s_A*r2 + s_R]);

        if (s_A < r2){
             result_c[s_R] = logsumexpf(result_c[s_R], tmp_result2 + head_cc2[b_idx*r2*r2 + s_A*r2 + s_R]);
             result_d1[s_R] = logsumexpf(result_d1[s_R], tmp_result2 + head_dc2[b_idx*r2*r2*4 + s_A*r2*4 + 0*r2 + s_R]);
             result_d2[s_R] = logsumexpf(result_d2[s_R], tmp_result2 + head_dc2[b_idx*r2*r2*4 + s_A*r2*4 + 1*r2 + s_R]);
             result_d3[s_R] = logsumexpf(result_d3[s_R], tmp_result2 + head_dc2[b_idx*r2*r2*4 + s_A*r2*4 + 2*r2 + s_R]);
             result_d4[s_R] = logsumexpf(result_d4[s_R], tmp_result2 + head_dc2[b_idx*r2*r2*4 + s_A*r2*4 + 3*r2 + s_R]);
        }
    }

    __syncthreads();

    if(s_R==0){
        float final_result_l = logf(0);
        float final_result_r = logf(0);

        for(int i=0;i<r1;i++){
            final_result_l = logsumexpf(final_result_l, result_l[i]);
            final_result_r = logsumexpf(final_result_l, result_r[i]);
        }

        alpha_l[b_idx*L*L*r1 +start*L*r1 +end*r1 + s_A] = final_result_l;
        alpha_r[b_idx*L*L*r1 +start*L*r1 +end*r1 + s_A] = final_result_r;

        if(s_A < r2){
            
            float final_result_c = logf(0);
            float final_result_d1 = logf(0);
            float final_result_d2 = logf(0);
            float final_result_d3 = logf(0);
            float final_result_d4 = logf(0);

            for(int i = 0; i < r1; i++){
               final_result_c = logsumexpf(final_result_c, result_c[i]);
               final_result_d1 = logsumexpf(final_result_d1, result_d1[i]);
               final_result_d2 = logsumexpf(final_result_d2, result_d2[i]);
               final_result_d3 = logsumexpf(final_result_d3, result_d3[i]);
               final_result_d4 = logsumexpf(final_result_d4, result_d4[i]);
            }
            alpha_cc[b_idx*L*L*r2 + start*L*r2 +end*r2 + s_A] = final_result_c;
            alpha_cd[b_idx*L*L*r2*4 + start*L*r2*4 +end*r2*4 + 0*r2 + s_A] = final_result_d1;
            alpha_cd[b_idx*L*L*r2*4 + start*L*r2*4 +end*r2*4 + 1*r2 + s_A] = final_result_d2;
            alpha_cd[b_idx*L*L*r2*4 + start*L*r2*4 +end*r2*4 + 2*r2 + s_A] = final_result_d3;
            alpha_cd[b_idx*L*L*r2*4 + start*L*r2*4 +end*r2*4 + 3*r2 + s_A] = final_result_d4;
        }
    }
}


__global__ void kernel_forward_d1( float *  head_cd,
                                   float * head_dd,
                                   float * alpha_dl,
                                   float * alpha_dr,
                                   float * alpha_cd,
                                   float * alpha_dd,
                                   int B,  int L,   int width,   int d,   int r)
{
      int b_idx = blockIdx.x / (L-width);
      int start = blockIdx.x % (L-width);
      int gap_start_minus_start = (blockIdx.y) / (L-width);
      int gap_width =  ((blockIdx.y) % ( L-width)) + 1;
      int end = start + width + gap_width;
      if(end>=L){
        return;
      }
      int s_A = (blockIdx.z);
      int s_R = threadIdx.x;
      int gap_start = start + gap_start_minus_start + 1;
      int gap_end = gap_start + gap_width;
      if(gap_start - start > width-1)
      {
         return;
      }

    __shared__ float result[1000];
    __shared__ float result2[1000];
    __syncthreads();

    float *   s_l = alpha_dl + b_idx * L * L * r;
    float *   s_r = alpha_dr + b_idx * L * L * r;
    float tmp =  s_l[start*L*r + gap_start*r + s_R] + s_r[gap_end*L*r + end*r + s_R];
    result[s_R] = tmp + head_cd[b_idx*d*r + s_R*d + s_A];
    result2[s_R] = tmp + head_dd[b_idx*d*r + s_R*d + s_A];
    __syncthreads();

    if(s_R==0){
    float final_result = logf(0);
    float final_result2 = logf(0);
    for(int i=0;i<r;i++){
        final_result = logsumexpf(final_result, result[i]);
        final_result2 = logsumexpf(final_result2, result2[i]);
    }
    alpha_cd[b_idx*L*L*L*L*d + start*L*L*L*d + gap_start*L*L*d + gap_end*L*d + end*d + s_A] = final_result;
    alpha_dd[b_idx*L*L*L*L*d + start*L*L*L*d + gap_start*L*L*d + gap_end*L*d + end*d + s_A] = final_result2;
    }
}






__global__ void kernel_forward_d2(float *  head_cd,
                                float *  head_dd,
                                float * alpha_dc,
                                float * alpha_dd,
                                float * alpha_cd,
                                int B,  int L,  int width,   int d,   int r)
{
      int b_idx = blockIdx.x / (L-width);
      int start = blockIdx.x % (L-width);
      int gap_start_minus_start = (blockIdx.y) / (L-width);
      int gap_width =  ((blockIdx.y) % ( L-width)) + 1;
      int end = start + width + gap_width;
        if(end>=L){
        return;
        }
     int s_A = (blockIdx.z);
     int s_R = threadIdx.x;
     int gap_start = start + gap_start_minus_start + 1;
     int gap_end = gap_start + gap_width;
     if(gap_start - start > width-1)
     {
        return;
     }

     float *   s = alpha_dc + b_idx * L * L * r * 4;
     float *   s_d = alpha_dd + b_idx * L * L * L * L * r;
     int L3 = L * L * L * r;
     int L2 = L * L * r;
     int L1 = L * r;
     float tmp_result = logf(0);
     __shared__ float result[200];
     __shared__ float result2[200];
     __syncthreads();
     for(int split=start+1; split< gap_start; split++)
     {
        tmp_result = logsumexpf(tmp_result, s_d[split*L3 + gap_start*L2 + gap_end*L1 + end*r + s_R] + s[start*L*r*4 + split*r*4 +  0*r + s_R]);
        tmp_result = logsumexpf(tmp_result, s_d[start*L3 + split*L2 + gap_end*L1 + end*r + s_R] + s[split*L*r*4 + gap_start*r*4 + 1*r + s_R]);
     }

     for(int split = gap_end+1; split <end; split++){
        tmp_result = logsumexpf(tmp_result, s_d[start*L3 + gap_start*L2 + split*L1 + end*r + s_R] + s[gap_end*L*r*4 + split*r*4 + 2*r + s_R]);
        tmp_result = logsumexpf(tmp_result, s_d[start*L3 + gap_start*L2 + gap_end*L1 + split*r + s_R] + s[split*L*r*4 + end*r*4 + 3*r + s_R]);
     }

     result[s_R] = tmp_result + head_cd[b_idx*d*r + s_R*r + s_A];
     result2[s_R] = tmp_result + head_dd[b_idx*d*r + s_R*r + s_A];
     __syncthreads();

     if(s_R==0){
        float final_result = logf(0);
        float final_result2 = logf(0);
        for(int i=0;i<r;i++){
            final_result = logsumexpf(final_result, result[i]);
            final_result2 = logsumexpf(final_result2, result2[i]);
        }
        alpha_cd[b_idx*L*L*L*L*d + start*L*L*L*d + gap_start*L*L*d + gap_end*L*d + end*d + s_A] = logsumexpf(alpha_cd[b_idx*L*L*L*L*d + start*L*L*L*d + gap_start*L*L*d + gap_end*L*d + end*d + s_A],final_result);
        alpha_dd[b_idx*L*L*L*L*d + start*L*L*L*d + gap_start*L*L*d + gap_end*L*d + end*d + s_A] = logsumexpf( alpha_dd[b_idx*L*L*L*L*d + start*L*L*L*d + gap_start*L*L*d + gap_end*L*d + end*d + s_A],final_result2);
    }
}



void cuda_forward( float *head_cl1,  float *head_cr1,
                   float *head_cc1,  float *head_dc1,
                   float *head_cl2,  float *head_cr2,
                   float *head_cc2,  float *head_dc2,
                   float *head_cd1, float *head_cd2,
                   float *head_dd1, float *head_dd2,
                   float *root_c, float *root_d,
                   float *alpha_lc, float *alpha_rc, float *alpha_ld, float *alpha_rd,
                   float *alpha_cc, float *alpha_cd, float *alpha_dc, float *alpha_dd,
                   int B, int L, int r1, int r2)
{
    for(int w=2; w<L-1; w++){
      dim3 gridDim(B, L-w, r1);
      dim3 blockDim(r1);
      kernel_forward<<<gridDim, blockDim>>>(head_cl1, head_cr1, head_cc1, head_dc1, head_cl2, head_cr2, head_cc2, head_dc2, alpha_lc, alpha_rc, alpha_cc, alpha_cd, alpha_dc,  B, L, w, r1, r2);

      dim3 gridDim3(B*(L-w),  (w-1)*(L-w), r2);
      dim3 blockDim3(r1);
      kernel_forward_d1<<<gridDim3, blockDim3>>>(head_cd1, head_dd1, alpha_ld, alpha_rd, alpha_cd, alpha_dd, B, L, w, r2, r1);

      if(w>2){
            dim3 gridDim4(B*(L-w),  (w-1)*(L-w), r2);
            dim3 blockDim4(r2);
            kernel_forward_d2<<<gridDim4, blockDim4>>>(head_cd2, head_dd2, alpha_dc, alpha_dd, alpha_cd, B, L, w, r2, r2);
      }
     }

    dim3 gridDim3(B);
    dim3 blockDim3(r1);
    kernel_forward_root<<<gridDim3, blockDim3>>>(root_c, root_d, alpha_lc, alpha_rc, alpha_cc, alpha_cd, B, L, r1, r2);
}