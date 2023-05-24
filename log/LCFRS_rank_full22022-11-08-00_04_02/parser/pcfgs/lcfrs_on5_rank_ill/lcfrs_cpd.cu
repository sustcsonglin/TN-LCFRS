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


__global__ void kernel_forward( float* alpha,
                                float* alpha_l,
                                float* alpha_r,
                                int B,   int L,   int width,  int r)
{
      int b_idx = blockIdx.x;
      int start = blockIdx.y;
      int s_R = threadIdx.x;

      if(start+width >= L){
        return;
      }

      int end = start + width;
      float*   s_l = alpha_l + b_idx * L * L * r;
      float*   s_r = alpha_r + b_idx * L * L * r;

      float tmp_result = logf(0);

      for (int split = start+1; split < start+width; split++)
      {
        tmp_result = logsumexpf(tmp_result, s_l[start*L*r + split * r + s_R] + s_r[split*L*r + end*r + s_R]);
      }

      alpha[b_idx*L*L*r+start*L*r+end*r+s_R] = tmp_result;
}



__global__ void kernel_backward(
                                float* alpha,
                                float* alpha_l,
                                float* alpha_r,
                                int B,   int L,   int width,  int r)
{
      int b_idx = blockIdx.x;
      int start = blockIdx.y;
      int s_R = threadIdx.x;

      if(start+width >= L){
        return;
      }

      int end = start + width;
      float*   s_l = alpha_l + b_idx * L * L * r;
      float*   s_r = alpha_r + b_idx * L * L * r;

     float parent_inside = alpha[b_idx*L*L*r+start*L*r+end*r+s_R];
     float parent_grd  = alpha[b_idx*L*L*r+end*L*r+start*r+s_R];

     for (int split = start+1; split < start+width; split++)
     {
        float tmp = exp(s_l[start*L*r + split * r + s_R] + s_r[split*L*r + end*r + s_R] - parent_inside) * parent_grd;
        atomicAdd(s_l + split*L*r + start*r + s_R, tmp);
        atomicAdd(s_r + end*L*r + split*r + s_R, tmp);
     }
}



__global__ void kernel_forward_close(
                                 float* alpha,
                                 float* alpha_cd,
                                 float* alpha_cc,
                                 int B,   int L,  int width, int r)
{
      int b_idx = blockIdx.x;
      int start = blockIdx.y;
      int s_R = threadIdx.x;

      if(start+width >= L){
        return;
      }

      int end = start + width;
      float*   s = alpha_cc + b_idx * L * L * r;
      float*   s_d = alpha_cd + b_idx * L * L * L * L * r;

      int L3 = L * L * L * r;
      int L2 = L * L * r;
      int L1 = L * r;

      float tmp_result = logf(0);

      for (int split = start+1; split < start+width-1; split++)
      {
       for(int split2 = split+1; split2 < start+width; split2++)
       {
              tmp_result = logsumexpf(tmp_result,s[split*L*r + split2*r + s_R] + s_d[start*L3 + split*L2 + split2*L1 + end*r + s_R]);
        }
      }

      alpha[b_idx*L*L*r+start*L*r+(end)*r+s_R] = tmp_result;
}




__global__ void kernel_backward_close(float* alpha,
                                      float* alpha_cd,
                                      float* alpha_cc,
                                      const int B, const int L, const int width,  const int r)
{
    const int b_idx = blockIdx.x;
    const int start = blockIdx.y;
    const int s_R = threadIdx.x;

    if(start+width >= L){
        return;
    }

    int end = start + width;
    float*  s = alpha_cc + b_idx * L * L * r;
    float*  s_d = alpha_cd + b_idx * L * L * L * L * r;

    const int L3 = L * L * L * r;
    const int L2 = L * L * r;
    const int L1 = L * r;
    float parent_inside = alpha[b_idx*L*L*r+start*L*r+(end)*r+s_R];
    float parent_grd = alpha[b_idx*L*L*r+end*L*r+(start)*r+s_R];

    for (int split = start+1; split < start+width-1; split++)
    {
         for(int split2 = split+1; split2 < start+width; split2++)
         {
              float tmp = exp(s[split*L*r + split2*r + s_R] + s_d[start*L3 + split*L2 + split2*L1 + end*r + s_R]  - parent_inside) * parent_grd;
              atomicAdd(s + split2*L*r + split*r + s_R, tmp);
              atomicAdd(s_d + split*L3 + start*L2 + split2*L1 + end*r + s_R, tmp);
         }
    }
}

__global__ void kernel_forward_close_ill(
                                 float* alpha,
                                 float* alpha_ill,
                                 int B,   int L,  int width, int r)
{
      int b_idx = blockIdx.x;
      int start = blockIdx.y;
      int s_R = threadIdx.x;

      if(start+width >= L){
        return;
      }

      int end = start + width;
      float*   s_d = alpha_ill + b_idx * L * L * L * L * r;

      int L3 = L * L * L * r;
      int L2 = L * L * r;
      int L1 = L * r;

      float tmp_result = logf(0);

      for (int split = start+1; split < start+width-1; split++)
      {
       for(int split2 = split+1; split2 < start+width; split2++)
       {
            for(int split3 = split2 + 1; split3 < start+width; split3++){
                tmp_result = logsumexpf(tmp_result,s_d[start*L3 + split*L2 + split2*L1 + split3*r + s_R] + s_d[split1*L3 + split2*L2 + split3*L1 + end*r + s_R]);
            }
        }
      }
      alpha[b_idx*L*L*r+start*L*r+(end)*r+s_R] = tmp_result;
}

__global__ void kernel_backward_close_ill(float* alpha,
                                      float* alpha_cd,
                                      const int B, const int L, const int width,  const int r)
{
    const int b_idx = blockIdx.x;
    const int start = blockIdx.y;
    const int s_R = threadIdx.x;

    if(start+width >= L){
        return;
    }

    int end = start + width;
    float*  s = alpha_cc + b_idx * L * L * r;
    float*  s_d = alpha_cd + b_idx * L * L * L * L * r;

    const int L3 = L * L * L * r;
    const int L2 = L * L * r;
    const int L1 = L * r;
    float parent_inside = alpha[b_idx*L*L*r+start*L*r+(end)*r+s_R];
    float parent_grd = alpha[b_idx*L*L*r+end*L*r+(start)*r+s_R];
      for (int split = start+1; split < start+width-1; split++)
      {
       for(int split2 = split+1; split2 < start+width; split2++)
       {
            for(int split3 = split2 + 1; split3 < start+width; split3++){
                float tmp = exp(s_d[start*L3 + split*L2 + split2*L1 + split3*r + s_R] + s_d[split1*L3 + split2*L2 + end*L1 + split3*r + s_R]));
              atomicAdd(s_d + split*L3 + start*L2 + split2*L1 + split3*r + s_R, tmp);
              atomicAdd(s_d + split2*L3 + split1*L2 + end*L1 + split3*r + s_R, tmp);
            }
        }
      }

}

__global__ void kernel_forward_cr( float*   cc,
                                   float* alpha,
                                   float* alpha_cc,
                                   int B,   int L,  int width,  int r1, int r2)
{
    int b_idx = blockIdx.x;
    int start = blockIdx.y;
    int s_A = blockIdx.z;
    int s_R = threadIdx.x;

    int end = start + width;
    if(end>=L){
        return;
    }
    __shared__ float result[1000];
    __syncthreads();
    float inside = alpha[b_idx*L*L*r1 + start*L*r1 + end*r1 + s_R];
    result[s_R] = inside + cc[b_idx*r1*r2 + s_A*r1 + s_R];
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
        alpha_cc[b_idx*L*L*r2 +   start*L*r2 + end*r2 + s_A] = result[0];
    }


}


__global__ void kernel_forward_cr_small( float*   cc,
                                   float* alpha,
                                   float* alpha_cc,
                                   int B,   int L,  int width,  int r1, int r2)
{
    int b_idx = blockIdx.x;
    int start = blockIdx.y;
    int s_A = blockIdx.z;
    int s_R = threadIdx.x;

    int end = start + width;
    if(end>=L){
        return;
    }
    __shared__ float result[1000];
    __syncthreads();
    float inside = alpha[b_idx*L*L*r1 + start*L*r1 + end*r1 + s_R];
    result[s_R] = inside + cc[b_idx*r1*r2 + s_A*r1 + s_R];
    __syncthreads();

    if(s_R==0){
    float final_result = logf(0);
        for(int i=0;i<r1;i++){
            final_result = logsumexpf(final_result, result[i]);
    }
    alpha_cc[b_idx*L*L*r2 + start*L*r2 + end*r2 + s_A] = logsumexpf(final_result,  alpha_cc[b_idx*L*L*r2 + start*L*r2 + end*r2 + s_A]);
   }
}


__global__ void kernel_backward_cmr(float*   cc,  float*   cc_grd,
                                    float* alpha,
                                    float* alpha_cc,
                                    int B,   int L,   int width,  int r1,  int r2)
{
    int b_idx = blockIdx.x;
    int start = blockIdx.y;
    int s_A = blockIdx.z;
    int s_R = threadIdx.x;
    int end = start + width;
    if(end>=L){
        return;
    }
    float parent_inside = alpha_cc[b_idx*L*L*r2 + start*L*r2 + end*r2 + s_A];
    float parent_grd = alpha_cc[b_idx*L*L*r2 + end*L*r2 + start*r2 + s_A];

    float tmp = exp(alpha[b_idx*L*L*r1 + start*L*r1 + end*r1 + s_R] + cc[b_idx*r1*r2 + s_A*r1 + s_R] - parent_inside) * parent_grd;

    atomicAdd(alpha + b_idx*L*L*r1 + end*L*r1 + start*r1 + s_R, tmp);
    atomicAdd(cc_grd + b_idx*r1*r2 + s_A*r1 + s_R, tmp);

}




__global__ void kernel_forward_dmr(  float*   cc,
                                    float* alpha,
                                    float* alpha_cc,
                                    int B,   int L,   int width, int r1,  int r2)
{
      int b_idx = blockIdx.x;
      int start = blockIdx.y;
      int s_A = blockIdx.z;
      int s_R = threadIdx.x;
      int end = start + width;
      if(end>=L){
        return;
       }
      __shared__ float result1[1000];
      __shared__ float result2[1000];
      __shared__ float result3[1000];
      __shared__ float result4[1000];
      float u = alpha[b_idx*L*L*r1 + start*L*r1 + end*r1 + s_R];
      result1[s_R] = u + cc[b_idx*r1*r2*4 + (s_A)*r1*4 + s_R];
      result2[s_R] = u + cc[b_idx*r1*r2*4 + (s_A)*r1*4 + 1*r1 + s_R];
      result3[s_R] = u + cc[b_idx*r1*r2*4 + (s_A)*r1*4 + 2*r1 + s_R];
      result4[s_R] = u + cc[b_idx*r1*r2*4 + (s_A)*r1*4 + 3*r1 + s_R];
      __syncthreads();


    int i = 2;
    while(i < r1){
        i*=2;
    }

    for(unsigned int s= i/2; s>0; s>>=1){
        if((s_R < s) & (s_R+ s < r1)){
           result1[s_R] = logsumexpf(result1[s_R+s], result1[s_R]);
           result2[s_R] = logsumexpf(result2[s_R+s], result2[s_R]);
           result3[s_R] = logsumexpf(result3[s_R+s], result3[s_R]);
           result4[s_R] = logsumexpf(result4[s_R+s], result4[s_R]);
        }
        __syncthreads();
    }

    if(s_R==0){
      alpha_cc[b_idx*L*L*r2*4 + start*L*r2*4 + end*r2*4 + s_A ] = result1[0];
      alpha_cc[b_idx*L*L*r2*4 + start*L*r2*4 + end*r2*4 + 1*r2 + s_A] = result2[0];
      alpha_cc[b_idx*L*L*r2*4 + start*L*r2*4 + end*r2*4 + 2*r2 + s_A] = result3[0];
      alpha_cc[b_idx*L*L*r2*4 + start*L*r2*4 + end*r2*4 + 3*r2 + s_A] = result4[0];
    }


}


__global__ void kernel_forward_dmr_small(  float*   cc,
                                    float* alpha,
                                    float* alpha_cc,
                                    int B,   int L,   int width, int r1,  int r2)
{
      int b_idx = blockIdx.x;
      int start = blockIdx.y;
      int s_A = blockIdx.z;
      int s_R = threadIdx.x;
      int end = start + width;
      if(end>=L){
        return;
       }
    __shared__ float result1[500];
    __shared__ float result2[500];
    __shared__ float result3[500];
    __shared__ float result4[500];
    float u = alpha[b_idx*L*L*r1 + start*L*r1 + end*r1 + s_R];
    result1[s_R] = u + cc[b_idx*r1*r2*4 + (s_A)*r1*4 + s_R ];
    result2[s_R] = u + cc[b_idx*r1*r2*4 + (s_A)*r1*4 + 1*r1 + s_R];
    result3[s_R] = u + cc[b_idx*r1*r2*4 + (s_A)*r1*4 + 2*r1 + s_R];
    result4[s_R] = u + cc[b_idx*r1*r2*4 + (s_A)*r1*4 + 3*r1 + s_R];
    __syncthreads();

    if(s_R==0){
    float final_result1 = logf(0);
    float final_result2 = logf(0);
    float final_result3 = logf(0);
    float final_result4 = logf(0);
    for(int i=0;i<r1;i++){
        final_result1 = logsumexpf(final_result1, result1[i]);
        final_result2 = logsumexpf(final_result2, result2[i]);
        final_result3 = logsumexpf(final_result3, result3[i]);
        final_result4 = logsumexpf(final_result4, result4[i]);
    }
    alpha_cc[b_idx*L*L*r2*4 + start*L*r2*4 + end*r2*4 + s_A ] = logsumexpf(alpha_cc[b_idx*L*L*r2*4 + start*L*r2*4 + end*r2*4 + s_A ], final_result1);
    alpha_cc[b_idx*L*L*r2*4 + start*L*r2*4 + end*r2*4 + 1*r2 + s_A] = logsumexpf(alpha_cc[b_idx*L*L*r2*4 + start*L*r2*4 + end*r2*4 + 1*r2 + s_A], final_result2);
    alpha_cc[b_idx*L*L*r2*4 + start*L*r2*4 + end*r2*4 + 2*r2 + s_A] = logsumexpf( alpha_cc[b_idx*L*L*r2*4 + start*L*r2*4 + end*r2*4 + 2*r2 + s_A], final_result3);
    alpha_cc[b_idx*L*L*r2*4 + start*L*r2*4 + end*r2*4 + 3*r2 + s_A] = logsumexpf(  alpha_cc[b_idx*L*L*r2*4 + start*L*r2*4 + end*r2*4 + 3*r2 + s_A], final_result4);
   }
}



__global__ void kernel_backward_dmr( float*  cc, float*  cc_grd,
                                     float* alpha,
                                     float* alpha_cc,
                                     int B,   int L,   int width,  int r1,  int r2)
{
      int b_idx = blockIdx.x;
      int start = blockIdx.y;
      int s_A = blockIdx.z;
      int s_R = threadIdx.x;

      int end = start + width;
      if(end>=L){
        return;
      }

      float u = alpha[b_idx*L*L*r1 + start*L*r1 + end*r1 + s_R];
      float u_grd = 0;

      float parent_inside = alpha_cc[b_idx*L*L*r2*4 + start*L*r2*4 + end*r2*4 + 0*r2 + s_A];
      float parent_grd = alpha_cc[b_idx*L*L*r2*4 + end*L*r2*4 + start*r2*4 + 0*r2 + s_A];

      float tmp = exp(u + cc[b_idx*r1*r2*4 + (s_A)*r1*4 + s_R] - parent_inside) * parent_grd;
      atomicAdd(cc_grd + b_idx*r1*r2*4 + (s_A)*r1*4 + s_R, tmp);
      u_grd += tmp;

      parent_inside = alpha_cc[b_idx*L*L*r2*4 + start*L*r2*4 + end*r2*4 + 1*r2 + s_A];
      parent_grd = alpha_cc[b_idx*L*L*r2*4 + end*L*r2*4 + start*r2*4 + 1*r2 + s_A];
      tmp = exp(u + cc[b_idx*r2*r1*4 + (s_A)*r1*4 + 1*r1 + s_R]-parent_inside) * parent_grd;
      atomicAdd(cc_grd + b_idx*r1*r2*4 + (s_A)*r1*4 + 1*r1 + s_R, tmp);
      u_grd += tmp;

      parent_inside = alpha_cc[b_idx*L*L*r2*4 + start*L*r2*4 + end*r2*4 + 2*r2 + s_A];
      parent_grd = alpha_cc[b_idx*L*L*r2*4 + end*L*r2*4 + start*r2*4 + 2*r2 + s_A];
      tmp = exp(u + cc[b_idx*r1*r2*4 + (s_A)*r1*4 + 2*r1 + s_R] - parent_inside)*parent_grd;
      atomicAdd(cc_grd + b_idx*r1*r2*4 + (s_A)*r1*4 + 2*r1 + s_R, tmp);
      u_grd += tmp;

      parent_inside = alpha_cc[b_idx*L*L*r2*4 + start*L*r2*4 + end*r2*4 +3*r2 + s_A];
      parent_grd = alpha_cc[b_idx*L*L*r2*4 + end*L*r2*4 + start*r2*4 + 3*r2 + s_A];
      tmp = exp(u + cc[b_idx*r1*r2*4 + (s_A)*r1*4 + 3*r1 + s_R] - parent_inside)*parent_grd;
      atomicAdd(cc_grd + b_idx*r1*r2*4 + (s_A)*r1*4 + 3*r1 + s_R, tmp);
      u_grd += tmp;

      atomicAdd(alpha + b_idx*L*L*r1 + end*L*r1 + start*r1 + s_R, u_grd);
}


__global__ void kernel_forward_d1( float *  head_cd,
                                   float * head_dd,
                                   float * head_ill_in,
                                   float * head_ill_out,
                                   float * alpha_dl,
                                   float * alpha_dr,
                                   float * alpha_cd,
                                   float * alpha_dd,
                                   float * alpha_cd_io,
                                   int B,  int L,   int width,   int r,   int r1, int r2, int r3)
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
    __shared__ float result3[1000];
    __shared__ float result4[1000];
    __syncthreads();

    float *   s_l = alpha_dl + b_idx * L * L * r;
    float *   s_r = alpha_dr + b_idx * L * L * r;
    float tmp =  s_l[start*L*r + gap_start*r + s_R] + s_r[gap_end*L*r + end*r + s_R];

    if (s_A < r1){
        result[s_R] = tmp + head_cd[b_idx*r*r1 + s_A*r + s_R];
    }
    if(s_A < r2){
        result2[s_R] = tmp + head_dd[b_idx*r*r2 + s_A*r + s_R];
    }

    if (s_A < r3){
        result3[s_R] = tmp + head_ill_in[b_idx*r*r3 + s_A*r + s_R];
        result4[s_R] = tmp + head_ill_out[b_idx*r*r3 + s_A*r + s_R];
    }

    __syncthreads();

    int i = 2;
    while(i < r){
        i*=2;
    }

    for(unsigned int s= i/2; s>0; s>>=1){
        if((s_R < s) & (s_R+ s < r)){
           if(s_A < r1){
                result[s_R] = logsumexpf(result[s_R], result[s_R+s]);
           }
           if(s_A < r2){
                result2[s_R] = logsumexpf(result2[s_R], result2[s_R+s]);
           }
           if (s_A < r3){
                result3[s_R] =  logsumexpf(result3[s_R], result3[s_R+s]);
                result4[s_R] =  logsumexpf(result4[s_R], result4[s_R+s]);
           }
        }
        __syncthreads();
    }

    if(s_R==0){
        if(s_A < r1){
            alpha_cd[b_idx*L*L*L*L*r1 + start*L*L*L*r1 + gap_start*L*L*r1 + gap_end*L*r1 + end*r1 + s_A] = result[0];
        }
        if(s_A < r2){
            alpha_dd[b_idx*L*L*L*L*r2 + start*L*L*L*r2 + gap_start*L*L*r2 + gap_end*L*r2 + end*r2 + s_A] = result2[0];
        }
        if(s_A < r3){
            alpha_cd_io[b_idx*L*L*L*L*r3 + start*L*L*L*r3 + gap_start*L*L*r3 + gap_end*L*r3 + end*r3 + s_A] = result3[0];
            alpha_cd_io[b_idx*L*L*L*L*r3 + start*L*L*L*r3 + gap_start*L*L*r3 + end*L*r3 + gap_end*r3 + s_A] = result4[0];
        }
    }
}


__global__ void kernel_backward_d1(  float *   head_cd, float *   head_cd_grd,
                                     float *   head_dd, float * head_dd_grd,
                                     float *   head_ill_in,  float * head_ill_in_grd,
                                     float *   head_ill_out,  float * head_ill_out_grd,
                                     float * alpha_ld, float * alpha_rd, float *  alpha_cd, float *  alpha_dd, float * alpha_cd_io,
                                     int B,   int L,   int width,   int r,  int r1, int r2, int r3)
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
     float *   s_l = alpha_ld + b_idx * L * L * r;
     float *   s_r = alpha_rd + b_idx * L * L * r;

     float tmp_grd = 0;

     float tmp = (s_l[start*L*r + gap_start*r + s_R] + s_r[gap_end*L*r + end*r + s_R]);
      if (s_A < r1){
        float parent_inside = alpha_cd[b_idx*L*L*L*L*r1 + start*L*L*L*r1 + gap_start*L*L*r1 + gap_end*L*r1 + end*r1 + s_A];
        float parent_grd = alpha_cd[b_idx*L*L*L*L*r1 + gap_start*L*L*L*r1 + start*L*L*r1 + gap_end*L*r1 + end*r1 + s_A];

        tmp_grd =  exp(tmp + head_cd[b_idx*r*r1 + s_A*r + s_R] - parent_inside) * parent_grd;
        atomicAdd(head_cd_grd + b_idx*r*r1 + s_A * r + s_R, tmp_grd);
     }

     if(s_A < r2){
        float parent_inside = alpha_dd[b_idx*L*L*L*L*r2 + start*L*L*L*r2 + gap_start*L*L*r2 + gap_end*L*r2 + end*r2 + s_A];
        float parent_grd = alpha_dd[b_idx*L*L*L*L*r2 + gap_start*L*L*L*r2 + start*L*L*r2 + gap_end*L*r2 + end*r2 + s_A];
        float tmp_grd2 = exp(tmp + head_dd[b_idx*r*r2 +s_A*r+ s_R] - parent_inside) * parent_grd;
        tmp_grd += tmp_grd2;
        atomicAdd(head_dd_grd + b_idx*r*r2 + s_A*r + s_R, tmp_grd2);
     }

     if (s_A < r3){
        float parent_inside = alpha_cd_io[b_idx*L*L*L*L*r3 + start*L*L*L*r3 + gap_start*L*L*r3 + gap_end*L*r3 + end*r3 + s_A];
        float parent_grd = alpha_cd_io[b_idx*L*L*L*L*r3 + gap_start*L*L*L*r3 + start*L*L*r3 + gap_end*L*r3 + end*r3 + s_A];
        float tmp_grd2 = exp(tmp + head_ill_in[b_idx*r*r3 +s_A*r+ s_R] - parent_inside) * parent_grd;
        tmp_grd += tmp_grd2;
        atomicAdd(head_ill_in + b_idx*r*r3 + s_A*r + s_R, tmp_grd2);
     }

      if (s_A < r3){
        float parent_inside = alpha_cd_io[b_idx*L*L*L*L*r3 + start*L*L*L*r3 + gap_start*L*L*r3 + end*L*r3 + gap_end*r3 + s_A];
        float parent_grd = alpha_cd_io[b_idx*L*L*L*L*r3 + gap_start*L*L*L*r3 + start*L*L*r3 + end*L*r3 + gap_end*r3 + s_A];
        float tmp_grd2 = exp(tmp + head_ill_out[b_idx*r*r3 +s_A*r+ s_R] - parent_inside) * parent_grd;
        tmp_grd += tmp_grd2;
        atomicAdd(head_ill_out + b_idx*r*r3 + s_A*r + s_R, tmp_grd2);
     }
     atomicAdd(s_l + gap_start*L*r + start*r + s_R, tmp_grd);
     atomicAdd(s_r + end*L*r + gap_end*r + s_R, tmp_grd);
}


__global__ void kernel_forward_d2_step1(
                                float * alpha_dd,
                                float * alpha_dc,
                                int B,  int L,  int width,   int r)
{
      int b_idx = blockIdx.x / (L-width);
      int start = blockIdx.x % (L-width);
      int gap_start_minus_start = (blockIdx.y) / (L-width);
      int gap_width =  ((blockIdx.y) % ( L-width)) + 1;
      int end = start + width + gap_width;
      if(end>=L){
        return;
      }
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

     for(int split=start+1; split< gap_start; split++)
     {
        tmp_result = logsumexpf(tmp_result, s_d[split*L3 + gap_start*L2 + gap_end*L1 + end*r + s_R] + s[start*L*r*4 + split*r*4 +  0*r + s_R]);
        tmp_result = logsumexpf(tmp_result, s_d[start*L3 + split*L2 + gap_end*L1 + end*r + s_R] + s[split*L*r*4 + gap_start*r*4 + 1*r + s_R]);
     }

     for(int split = gap_end+1; split <end; split++){
        tmp_result = logsumexpf(tmp_result, s_d[start*L3 + gap_start*L2 + split*L1 + end*r + s_R] + s[gap_end*L*r*4 + split*r*4 + 2*r + s_R]);
        tmp_result = logsumexpf(tmp_result, s_d[start*L3 + gap_start*L2 + gap_end*L1 + split*r + s_R] + s[split*L*r*4 + end*r*4 + 3*r + s_R]);
     }

     s_d[start*L3 + gap_start*L2 + end*L1 + gap_end*r + s_R] = tmp_result;
}

__global__ void kernel_backward_d2_step1(
                                float * alpha_dc,
                                float * alpha_dd,
                                int B,  int L,  int width,   int r)
{
      int b_idx = blockIdx.x / (L-width);
      int start = blockIdx.x % (L-width);
      int gap_start_minus_start = (blockIdx.y) / (L-width);
      int gap_width =  ((blockIdx.y) % ( L-width)) + 1;
      int end = start + width + gap_width;
      if(end>=L){
        return;
      }
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
     float parent_inside =  s_d[start*L3 + gap_start*L2 + end*L1 + gap_end*r + s_R];
     float parent_grd =  s_d[gap_start*L3 + start*L2 + end*L1 + gap_end*r + s_R];

     for(int split=start+1; split< gap_start; split++)
     {
        float tmp_grd = exp(s_d[split*L3 + gap_start*L2 + gap_end*L1 + end*r + s_R] + s[start*L*r*4 + split*r*4 +  0*r + s_R] - parent_inside) * parent_grd;

        atomicAdd(s_d + gap_start*L3 + split*L2 + gap_end*L1 + end*r + s_R, tmp_grd);
        atomicAdd(s + split*L*r*4 + start*r*4 + 0*r + s_R, tmp_grd);

        tmp_grd = exp(s_d[start*L3 + split*L2 + gap_end*L1 + end*r + s_R] + s[split*L*r*4 + gap_start*r*4 + 1*r + s_R] - parent_inside) * parent_grd;
        atomicAdd(s_d + split*L3 + start*L2 + gap_end*L1 + end*r + s_R, tmp_grd);
        atomicAdd(s + gap_start*L*r*4 + split*r*4 + 1*r + s_R, tmp_grd);
     }

     for(int split = gap_end+1; split <end; split++){
        float tmp_grd = exp(s_d[start*L3 + gap_start*L2 + split*L1 + end*r + s_R] + s[gap_end*L*r*4 + split*r*4 + 2*r + s_R] - parent_inside) * parent_grd;
        atomicAdd(s_d + gap_start*L3 + start*L2 + split*L1 + end*r + s_R, tmp_grd);
        atomicAdd(s + split*L*r*4 + gap_end*r*4 + 2*r + s_R , tmp_grd);

        tmp_grd = exp(s_d[start*L3 + gap_start*L2 + gap_end*L1 + split*r + s_R] + s[split*L*r*4 + end*r*4 + 3*r + s_R] - parent_inside) * parent_grd;
        atomicAdd(s_d + gap_start*L3 + start*L2 + gap_end*L1 + split*r + s_R, tmp_grd);
        atomicAdd(s + end*L*r*4 + split*r*4 + 3*r + s_R, tmp_grd);
     }

}


__global__ void kernel_forward_d2_step2(
                                float *  head_cd,
                                float *  head_dd,
                                float *  head_ill_in,
                                float *  head_ill_out,
                                float * alpha_dd,
                                float * alpha_cd,
                                float * alpha_cd_io,
                                int B,  int L,  int width, int r, int r1, int r2, int r3)
{
      int b_idx = blockIdx.x / (L-width);
      int start = blockIdx.x % (L-width);
      int gap_start_minus_start = (blockIdx.y) / (L-width);
      int gap_width =  ((blockIdx.y) % ( L-width)) + 1;
      int end = start + width + gap_width;
      if(end>=L){
        return;
      }

      int s_R = threadIdx.x;
      int s_A = blockIdx.z;

      int gap_start = start + gap_start_minus_start + 1;
      int gap_end = gap_start + gap_width;
      if(gap_start - start > width-1)
      {
        return;
      }

      __shared__ float result[500];
      __shared__ float result2[500];
      __shared__ float result3[500];
      __shared__ float result4[500];
      __syncthreads();


      float score = alpha_dd[b_idx*L*L*L*L*r + start*L*L*L*r + gap_start*L*L*r + end*L*r + gap_end*r + s_R];

      if(s_A < r1){
        result[s_R] = score + head_cd[b_idx*r*r1 + s_A * r + s_R];
      }

      if(s_A < r2){
         result2[s_R] = score + head_dd[b_idx*r*r2 + s_A * r + s_R];
      }

      if(s_A < r3){
         result3[s_R] = score + head_ill_in[b_idx*r*r3 + s_A * r + s_R];
         result4[s_R] = score + head_ill_out[b_idx*r*r3 + s_A * r + s_R];
      }

      __syncthreads();

      if(s_R==0){
        float final_result = logf(0);
        float final_result2 = logf(0);
        float final_result3 = logf(0);
        float final_result4 = logf(0);
        for(int i=0;i<r;i++){
            if(s_A < r1){
                final_result = logsumexpf(final_result, result[i]);
            }
            if(s_A < r2){
            final_result2 = logsumexpf(final_result2, result2[i]);
            }

            if(s_A < r3){
            final_result3 = logsumexpf(final_result3, result3[i]);
            }

            if(s_A < r3){
            final_result4 = logsumexpf(final_result4, result4[i]);
            }

        }
        if(s_A < r1){
            alpha_cd[b_idx*L*L*L*L*r1 + start*L*L*L*r1 + gap_start*L*L*r1 + gap_end*L*r1 + end*r1 + s_A] = logsumexpf(alpha_cd[b_idx*L*L*L*L*r1 + start*L*L*L*r1 + gap_start*L*L*r1 + gap_end*L*r1 + end*r1 + s_A], final_result);
        }
        if(s_A < r2){
            alpha_dd[b_idx*L*L*L*L*r2 + start*L*L*L*r2 + gap_start*L*L*r2 + gap_end*L*r2 + end*r2 + s_A] = logsumexpf( alpha_dd[b_idx*L*L*L*L*r2 + start*L*L*L*r2 + gap_start*L*L*r2 + gap_end*L*r2 + end*r2 + s_A], final_result2);
        }
        if(s_A < r3){
            alpha_cd_io[b_idx*L*L*L*L*r3 + start*L*L*L*r3 + gap_start*L*L*r3 + gap_end*L*r3 + end*r3 + s_A] = logsumexpf(     alpha_cd_io[b_idx*L*L*L*L*r3 + start*L*L*L*r3 + gap_start*L*L*r3 + gap_end*L*r3 + end*r3 + s_A], final_result3);
            alpha_cd_io[b_idx*L*L*L*L*r3 + start*L*L*L*r3 + gap_start*L*L*r3 + end*L*r3 + gap_end*r3 + s_A] = logsumexpf(     alpha_cd_io[b_idx*L*L*L*L*r3 + start*L*L*L*r3 + gap_start*L*L*r3 + end*L*r3 + gap_end*r3 + s_A], final_result4);
        }
    }
}


__global__ void kernel_backward_d2_step2(
                                float *  head_cd,
                                float *  head_cd_grd,
                                float *  head_dd,
                                float *  head_dd_grd,

                                float * head_ill_in,
                                float * head_ill_in_grd,
                                float * head_ill_out,
                                float * head_ill_out_grd,

                                float *  alpha_dd,
                                float *  alpha_cd,
                                float *  alpha_cd_io,

                                int B,  int L,  int width, int r, int r1, int r2, int r3)
{
      int b_idx = blockIdx.x / (L-width);
      int start = blockIdx.x % (L-width);
      int gap_start_minus_start = (blockIdx.y) / (L-width);
      int gap_width =  ((blockIdx.y) % ( L-width)) + 1;
      int end = start + width + gap_width;

      if(end>=L){
        return;
      }

      int s_R = threadIdx.x;
      int s_A = blockIdx.z;

      int gap_start = start + gap_start_minus_start + 1;
      int gap_end = gap_start + gap_width;
      if(gap_start - start > width-1)
      {
        return;
      }

      float tmp_grd = 0;
      float inside = alpha_dd[b_idx*L*L*L*L*r + start*L*L*L*r + gap_start*L*L*r + end*L*r + gap_end*r + s_R];
      if (s_A < r1){
        float parent_inside =  alpha_cd[b_idx*L*L*L*L*r1 + start*L*L*L*r1 + gap_start*L*L*r1 + gap_end*L*r1 + end*r1 + s_A];
        float parent_grd =  alpha_cd[b_idx*L*L*L*L*r1 + gap_start*L*L*L*r1 + start*L*L*r1 + gap_end*L*r1 + end*r1 + s_A];

        tmp_grd = exp(inside + head_cd[b_idx*r*r1 + s_A * r + s_R] - parent_inside) * parent_grd;
        atomicAdd(head_cd_grd + b_idx*r*r1 + s_A * r + s_R, tmp_grd);
     }

      if(s_A < r2){
        float parent_inside = alpha_dd[b_idx*L*L*L*L*r2 + start*L*L*L*r2 + gap_start*L*L*r2 + gap_end*L*r2 + end*r2 + s_A];
        float parent_grd = alpha_dd[b_idx*L*L*L*L*r2 + gap_start*L*L*L*r2 + start*L*L*r2 + gap_end*L*r2 + end*r2 + s_A];
        float tmp_grd2 = exp(inside + head_dd[b_idx*r*r2 + s_A * r + s_R] - parent_inside);
        tmp_grd2 *= parent_grd;
        atomicAdd(head_dd_grd + b_idx*r*r2 + s_A * r + s_R, tmp_grd2);
        tmp_grd += tmp_grd2;
      }

      if (s_A < r3){
        float parent_inside = alpha_cd_io[b_idx*L*L*L*L*r3 + start*L*L*L*r3 + gap_start*L*L*r3 + gap_end*L*r3 + end*r3 + s_A];
        float parent_grd = alpha_cd_io[b_idx*L*L*L*L*r3 + gap_start*L*L*L*r3 + start*L*L*r3 + gap_end*L*r3 + end*r3 + s_A];
        float tmp_grd2 = exp(tmp + head_ill_in[b_idx*r*r3 +s_A*r+ s_R] - parent_inside) * parent_grd;
        tmp_grd += tmp_grd2;
        atomicAdd(head_ill_in + b_idx*r*r3 + s_A*r + s_R, tmp_grd2);
     }

      if (s_A < r3){
        float parent_inside = alpha_cd_io[b_idx*L*L*L*L*r3 + start*L*L*L*r3 + gap_start*L*L*r3 + end*L*r3 + gap_end*r3 + s_A];
        float parent_grd = alpha_cd_io[b_idx*L*L*L*L*r3 + gap_start*L*L*L*r3 + start*L*L*r3 + end*L*r3 + gap_end*r3 + s_A];
        float tmp_grd2 = exp(tmp + head_ill_out[b_idx*r*r3 +s_A*r+ s_R] - parent_inside) * parent_grd;
        tmp_grd += tmp_grd2;
        atomicAdd(head_ill_out + b_idx*r*r3 + s_A*r + s_R, tmp_grd2);
      }

     atomicAdd(alpha_dd + b_idx*L*L*L*L*r + gap_start*L*L*L*r + start*L*L*r + end*L*r + gap_end*r + s_R, tmp_grd);

}

void cuda_forward( float *head_cl1,  float *head_cr1,
                   float *head_dl1,  float *head_dr1,
                   float *head_cc1,  float *head_dc1,

                   float *head_cl2,  float *head_cr2,
                   float *head_dl2,  float *head_dr2,
                   float *head_cc2,  float *head_dc2,

                   float *head_cl3,  float *head_cr3,
                   float *head_dl3,  float *head_dr3,
                   float *head_cc3,  float *head_dc3,

                   float *head_cd1, float *head_cd2,
                   float *head_dd1, float *head_dd2,
                   float *head_ill_in1, float *head_ill_in2,
                   float *head_ill_out1, float *head_ill_out2,

                   float *root_c, float *root_d,

                   float *alpha_lc, float *alpha_rc,
                   float *alpha_ld, float *alpha_rd,
                   float *alpha_cc, float *alpha_cd, float *alpha_dc, float *alpha_dd, float *alpha_cd_io,
                   float *alpha_tmp_c1, float *alpha_tmp_c2, float *alpha_tmp_c3,
                   int B, int L, int r1, int r2, int r3, int r4, int r5)

{
    for(int w=2; w<L-1; w++){
      dim3 gridDim(B, L-w);
      dim3 blockDim(r1);
      kernel_forward<<<gridDim, blockDim>>>(alpha_tmp_c1, alpha_lc, alpha_rc, B, L, w, r1);

      dim3 gridDim2(B, L-w, r1);
      kernel_forward_cr<<<gridDim2, blockDim>>>(head_cl1, alpha_tmp_c1, alpha_lc, B, L, w, r1, r1);
      kernel_forward_cr<<<gridDim2, blockDim>>>(head_cr1, alpha_tmp_c1, alpha_rc, B, L, w, r1, r1);

      dim3 gridDim3(B, L-w, r3);
      kernel_forward_cr<<<gridDim3, blockDim>>>(head_dl1, alpha_tmp_c1, alpha_ld, B, L, w, r1, r3);
      kernel_forward_cr<<<gridDim3, blockDim>>>(head_dr1, alpha_tmp_c1, alpha_rd, B, L, w, r1, r3);

      dim3 gridDim4(B, L-w, r2);
      kernel_forward_cr<<<gridDim4, blockDim>>>(head_cc1, alpha_tmp_c1, alpha_cc, B, L, w, r1, r2);
      dim3 gridDim5(B, L-w, r4);
      kernel_forward_dmr<<<gridDim5, blockDim>>>(head_dc1, alpha_tmp_c1, alpha_dc, B, L, w, r1, r4);

      if(w>2){
        dim3 gridDim2(B, L-w);
        dim3 blockDim2(r2);
        kernel_forward_close<<<gridDim2, blockDim2>>>(alpha_tmp_c2, alpha_cd, alpha_cc, B, L, w, r2);
        dim3 gridDim3(B, L-w, r1);
        kernel_forward_cr_small<<<gridDim3, blockDim2>>>(head_cl2, alpha_tmp_c2, alpha_lc, B, L, w, r2, r1);
        kernel_forward_cr_small<<<gridDim3, blockDim2>>>(head_cr2, alpha_tmp_c2, alpha_rc, B, L, w, r2, r1);

        dim3 gridDim4(B, L-w, r3);
        kernel_forward_cr_small<<<gridDim4, blockDim2>>>(head_dl2, alpha_tmp_c2, alpha_ld, B, L, w, r2, r3);
        kernel_forward_cr_small<<<gridDim4, blockDim2>>>(head_dr2, alpha_tmp_c2, alpha_rd, B, L, w, r2, r3);
        dim3 gridDim5(B, L-w, r2);
        kernel_forward_cr_small<<<gridDim5, blockDim2>>>(head_cc2, alpha_tmp_c2, alpha_cc, B, L, w, r2, r2);
        dim3 gridDim6(B, L-w, r4);
        kernel_forward_dmr_small<<<gridDim6, blockDim2>>>(head_dc2, alpha_tmp_c2, alpha_dc, B, L, w, r2, r4);
      }

      if (w>3){
        dim3 gridDim2(B, L-w);
        dim3 blockDim2(r5);
        kernel_forward_close_ill<<<gridDim2, blockDim2>>>(alpha_tmp_c3, alpha_cd_io, B, L, w, r5);
        dim3 gridDim3(B, L-w, r1);
        kernel_forward_cr_small<<<gridDim3, blockDim2>>>(head_cl3, alpha_tmp_c3, alpha_lc, B, L, w, r5, r1);
        kernel_forward_cr_small<<<gridDim3, blockDim2>>>(head_cr3, alpha_tmp_c3, alpha_rc, B, L, w, r5, r1);
        dim3 gridDim4(B, L-w, r3);
        kernel_forward_cr_small<<<gridDim4, blockDim2>>>(head_dl3, alpha_tmp_c3, alpha_ld, B, L, w, r5, r3);
        kernel_forward_cr_small<<<gridDim4, blockDim2>>>(head_dr3, alpha_tmp_c3, alpha_rd, B, L, w, r5, r3);
        dim3 gridDim5(B, L-w, r2);
        kernel_forward_cr_small<<<gridDim5, blockDim2>>>(head_cc3, alpha_tmp_c3, alpha_cc, B, L, w, r5, r2);
        dim3 gridDim6(B, L-w, r4);
        kernel_forward_dmr_small<<<gridDim6, blockDim2>>>(head_dc3, alpha_tmp_c3, alpha_dc, B, L, w, r5, r4);
      }

      dim3 gridDim11(B*(L-w),  (w-1)*(L-w), max(max(r2, r4), r5));
      dim3 blockDim11(r3);
      kernel_forward_d1<<<gridDim11, blockDim11>>>(head_cd1, head_dd1, head_ill_in1, head_ill_out1, alpha_ld, alpha_rd, alpha_cd, alpha_dd, alpha_cd_io, B, L, w, r3, r2, r4, r5);

      if(w >2){
      dim3 gridDim12(B*(L-w),  (w-1)*(L-w));
      dim3 blockDim12(r4);
      kernel_forward_d2_step1<<<gridDim12, blockDim12>>>(alpha_dd, alpha_dc, B, L, w, r4);
      dim3 gridDim13(B*(L-w),  (w-1)*(L-w),  max(r2, r4));
      kernel_forward_d2_step2<<<gridDim13, blockDim12>>>(head_cd2, head_dd2, head_ill_in2, head_ill_out2, alpha_dd, alpha_cd, alpha_cd_io, B, L, w, r4, r2, r4, r5);
      }
    }

    dim3 gridDim3(B);
    dim3 blockDim3(r1);
    kernel_forward_root<<<gridDim3, blockDim3>>>(root_c, root_d, alpha_lc, alpha_rc, alpha_cc, alpha_cd, B, L, r1, r2);

}


void cuda_backward( float *head_cl1,  float *head_cr1,
                    float *head_dl1,  float *head_dr1,
                    float *head_cc1,  float *head_dc1,

                    float *head_cl2,  float *head_cr2,
                    float *head_dl2,  float *head_dr2,
                    float *head_cc2,  float *head_dc2,

                    float *head_cl3,  float *head_cr3,
                    float *head_dl3,  float *head_dr3,
                    float *head_cc3,  float *head_dc3,

                    float *head_cd1,  float *head_cd2,
                    float *head_dd1,  float *head_dd2,
                    float *head_ill_in1,  float *head_ill_in2,
                    float *head_ill_out1,  float *head_ill_out2,

                    float *root_c,  float *root_d,

                    float *head_cl1_grd,  float *head_cr1_grd,
                    float *head_dl1_grd,  float *head_dr1_grd,
                    float *head_cc1_grd,  float *head_dc1_grd,

                    float *head_cl2_grd,  float *head_cr2_grd,
                    float *head_dl2_grd,  float *head_dr2_grd,
                    float *head_cc2_grd,  float *head_dc2_grd,

                    float *head_cl3_grd,  float *head_cr3_grd,
                    float *head_dl3_grd,  float *head_dr3_grd,
                    float *head_cc3_grd,  float *head_dc3_grd,

                    float *head_cd1_grd, float *head_cd2_grd,
                    float *head_dd1_grd, float *head_dd2_grd,
                    float *head_ill_in1_grd, float *head_ill_in2_grd,
                    float *head_ill_out1_grd, float *head_ill_out2_grd,

                    float *root_c_grd, float *root_d_grd,

                    float *alpha_lc, float *alpha_rc,
                    float *alpha_ld, float *alpha_rd,
                    float *alpha_cc, float *alpha_cd,
                    float *alpha_dc, float *alpha_dd,
                    float *alpha_cd_io,

                    float *alpha_tmp_c1, float *alpha_tmp_c2, float *alpha_tmp_c3,

                    int B, int L, int r1, int r2, int r3, int r4, int r5)
{
     dim3 gridDim3(B);
     dim3 blockDim3(r1);
     kernel_backward_root<<<gridDim3, blockDim3>>>(root_c, root_d,
                          root_c_grd, root_d_grd,
                          alpha_lc, alpha_rc,
                          alpha_cc, alpha_cd,
                          B, L, r1, r2);

     for(int w=L-2; w>1; w--){
        if(w>2){
            dim3 gridDim4(B*(L-w),  (w-1)*(L-w), max(r2, r4));
            dim3 blockDim4(r4);
            kernel_backward_d2_step2<<<gridDim4, blockDim4>>>(head_cd2, head_cd2_grd, head_dd2, head_dd2_grd, alpha_dd, alpha_cd, B, L, w, r4, r2, r4);
            dim3 gridDim44(B*(L-w),  (w-1)*(L-w));
            kernel_backward_d2_step1<<<gridDim44, blockDim4>>>(alpha_dc, alpha_dd, B, L, w, r4);
       }

       dim3 gridDim3(B*(L-w),  (w-1)*(L-w), max(r2, r4) );
       dim3 blockDim3(r3);
       kernel_backward_d1<<<gridDim3, blockDim3>>>(head_cd1, head_cd1_grd, head_dd1, head_dd1_grd, alpha_ld, alpha_rd, alpha_cd, alpha_dd, B, L, w, r3, r2, r4);

        if(w>3){
        dim3 blockDim3(r5);
        dim3 gridDim33(B, L-w, r1);
        kernel_backward_cmr<<<gridDim33, blockDim3>>>(head_cl3, head_cl3_grd, alpha_tmp_c3, alpha_lc, B, L, w, r5, r1);
        kernel_backward_cmr<<<gridDim33, blockDim3>>>(head_cr3, head_cr3_grd, alpha_tmp_c3, alpha_rc, B, L, w, r5, r1);
        dim3 gridDim3(B, L-w, r3);
        kernel_backward_cmr<<<gridDim3, blockDim3>>>(head_dl3, head_dl3_grd, alpha_tmp_c3, alpha_ld, B, L, w, r5, r3);
        kernel_backward_cmr<<<gridDim3, blockDim3>>>(head_dr3, head_dr3_grd, alpha_tmp_c3, alpha_rd, B, L, w, r5, r3);
        dim3 gridDim4(B, L-w, r2);
        kernel_backward_cmr<<<gridDim4, blockDim3>>>(head_cc3, head_cc3_grd, alpha_tmp_c3, alpha_cc, B, L, w, r5, r2);

        dim3 gridDim5(B, L-w, r4);
        kernel_backward_dmr<<<gridDim5, blockDim3>>>(head_dc3, head_dc3_grd, alpha_tmp_c3, alpha_dc, B, L, w, r5, r4);
        dim3 gridDim2(B, L-w);

        dim3 blockDim2(r5);
        kernel_backward_close_ill<<<gridDim2, blockDim2>>>(alpha_tmp_c2, alpha_cd, alpha_cc, B, L, w, r2);
       }

       if(w>2){
        dim3 blockDim3(r2);
        dim3 gridDim33(B, L-w, r1);
        kernel_backward_cmr<<<gridDim33, blockDim3>>>(head_cl2, head_cl2_grd, alpha_tmp_c2, alpha_lc, B, L, w, r2, r1);
        kernel_backward_cmr<<<gridDim33, blockDim3>>>(head_cr2, head_cr2_grd, alpha_tmp_c2, alpha_rc, B, L, w, r2, r1);
        dim3 gridDim3(B, L-w, r3);
        kernel_backward_cmr<<<gridDim3, blockDim3>>>(head_dl2, head_dl2_grd, alpha_tmp_c2, alpha_ld, B, L, w, r2, r3);
        kernel_backward_cmr<<<gridDim3, blockDim3>>>(head_dr2, head_dr2_grd, alpha_tmp_c2, alpha_rd, B, L, w, r2, r3);
        dim3 gridDim4(B, L-w, r2);
        kernel_backward_cmr<<<gridDim4, blockDim3>>>(head_cc2, head_cc2_grd, alpha_tmp_c2, alpha_cc, B, L, w, r2, r2);

        dim3 gridDim5(B, L-w, r4);
        kernel_backward_dmr<<<gridDim5, blockDim3>>>(head_dc2, head_dc2_grd, alpha_tmp_c2, alpha_dc, B, L, w, r2, r4);
        dim3 gridDim2(B, L-w);
        dim3 blockDim2(r2);
        kernel_backward_close<<<gridDim2, blockDim2>>>(alpha_tmp_c2, alpha_cd, alpha_cc,  B, L, w, r2);
       }

      dim3 gridDim2(B, L-w, r1);
      dim3 blockDim2(r1);
      kernel_backward_cmr<<<gridDim2, blockDim2>>>(head_cl1, head_cl1_grd, alpha_tmp_c1, alpha_lc, B, L, w, r1, r1);
      kernel_backward_cmr<<<gridDim2, blockDim2>>>(head_cr1, head_cr1_grd, alpha_tmp_c1, alpha_rc, B, L, w, r1, r1);
      dim3 gridDim33(B, L-w, r3);
      kernel_backward_cmr<<<gridDim33, blockDim2>>>(head_dl1, head_dl1_grd, alpha_tmp_c1, alpha_ld, B, L, w, r1, r3);
      kernel_backward_cmr<<<gridDim33, blockDim2>>>(head_dr1, head_dr1_grd, alpha_tmp_c1, alpha_rd, B, L, w, r1, r3);

      dim3 gridDim4(B, L-w, r2);
      kernel_backward_cmr<<<gridDim4, blockDim2>>>(head_cc1, head_cc1_grd,  alpha_tmp_c1, alpha_cc, B, L, w, r1, r2);
      dim3 gridDim5(B, L-w, r4);
      kernel_backward_dmr<<<gridDim5, blockDim2>>>(head_dc1, head_dc1_grd, alpha_tmp_c1, alpha_dc, B, L, w, r1, r4);

      dim3 gridDim6(B, L-w);
      dim3 blockDim6(r1);
      kernel_backward<<<gridDim6, blockDim6>>>(alpha_tmp_c1, alpha_lc, alpha_rc, B, L, w, r1);
    }

}




__global__ void kernel_argmax_c(const float* const s_span_c,
                                    float* alpha_c,
                                    float* alpha_d,
                                    int B,  int L,  int width
                                   )
{
     const int b_idx = blockIdx.x;
     const int start = threadIdx.x;
     float tmp_max = logf(0);
     float tmp_idx = -9999;
     int end = start + width;
     const int L2 = L*L;
     const int L3 = L2*L;
     const int L4 = L3*L;
     const float* const s_d = alpha_d + b_idx * L4;
     const float* const s_c = alpha_c + b_idx * L2;

     for(int split = start + 1; split < end; split+=1){
        float tmp = s_c[start*L+split] + s_c[split*L+end];
        if(tmp>tmp_max){
            tmp_max=tmp;
            tmp_idx=split;
        }
        for(int split2 =split+1;split2<end;split2+=1){
            tmp = s_c[split*L+split2] + s_d[start*L3+split*L2+split2*L+end];
            if(tmp>tmp_max){
                tmp_max=tmp;
                tmp_idx=-(split*L+split2);
            }
        }
     }
     alpha_c[b_idx*L*L + start*L + end] = tmp_max + s_span_c[b_idx*L*L+start*L + end];
     alpha_c[b_idx*L*L + end*L + start] = tmp_idx;
}



__global__ void kernel_argmax_d(const float* const s_span_d,
                                    float* alpha_c,
                                    float* alpha_d,
                                    int B,  int L,  int width
                                   )
{
    const int b_idx = blockIdx.x ;
    const int gap_start_minus_start = blockIdx.y;
    const int gap_width = blockIdx.z + 1;
    const int start = threadIdx.x;
    const int end = start + width + gap_width;

    if(end>=L){
        return;
    }

    const int gap_start = start + gap_start_minus_start + 1;
    const int gap_end = gap_start + gap_width;

    if(gap_start - start >width-1)
    {
        return;
    }

    const int L2 = L*L;
    const int L3 = L2*L;
    const int L4 = L3*L;
    const float* const s_d = alpha_d + b_idx * L4;
    const float* const s_c = alpha_c + b_idx * L2;


    float tmp_max = s_c[start*L + gap_start] + s_c[gap_end*L + end];
    float tmp_idx = -1;

    for(int split=start + 1; split<gap_start; split++){
        float tmp = s_c[start*L + split] + s_d[split*L3+ gap_start*L2 + gap_end*L + end];
        if (tmp > tmp_max){
            tmp_max = tmp;
            tmp_idx = split;
        }
        tmp = s_c[split*L + gap_start] + s_d[start*L3 + split*L2 + gap_end*L + end];
        if ( tmp > tmp_max){
            tmp_max = tmp;
            tmp_idx = L + split;
         }
    }

    for(int split = gap_end+1; split < end; split++){
        float tmp = s_c[gap_end*L + split] + s_d[start*L3 + gap_start*L2 + split*L + end];
        if ( tmp > tmp_max){
           tmp_max = tmp;
           tmp_idx = 2*L + split;
         }
         tmp = s_c[split*L + end] + s_d[start*L3 + gap_start*L2 + gap_end*L + split];
         if (tmp > tmp_max){
            tmp_max = tmp;
            tmp_idx = 3*L + split;
         }
    }
    alpha_d[b_idx*L4+start*L3+gap_start*L2+gap_end*L+end] = tmp_max + s_span_d[b_idx*L4+start*L3+gap_start*L2+gap_end*L+end];
    alpha_d[b_idx*L4+gap_start*L3+start*L2+gap_end*L+end] = tmp_idx;
}



__global__ void kernel_argmax_d_on4(const float* const s_span_d,
                                    float* alpha_c,
                                    float* alpha_d,
                                    int B,  int L,  int width
                                   )
{
    const int b_idx = blockIdx.x ;
    const int gap_start_minus_start = blockIdx.y;
    const int gap_width = blockIdx.z + 1;
    const int start = threadIdx.x;
    const int end = start + width + gap_width;

    if(end>=L){
        return;
    }

    const int gap_start = start + gap_start_minus_start + 1;
    const int gap_end = gap_start + gap_width;

    if(gap_start - start >width-1)
    {
        return;
    }

    const int L2 = L*L;
    const int L3 = L2*L;
    const int L4 = L3*L;
    const float* const s_d = alpha_d + b_idx * L4;
    const float* const s_c = alpha_c + b_idx * L2;


    float tmp_max = s_c[start*L + gap_start] + s_c[gap_end*L + end];
    float tmp_idx = -1;

    for(int split=start + 1; split<gap_start; split++){
        if(split == (start+1)){
            float tmp = s_c[start*L + split] + s_d[split*L3+ gap_start*L2 + gap_end*L + end];
            if (tmp > tmp_max){
                tmp_max = tmp;
                tmp_idx = split;
            }
        }
        if( (split+1) == gap_start){
            float tmp = s_c[split*L + gap_start] + s_d[start*L3 + split*L2 + gap_end*L + end];
            if ( tmp > tmp_max){
                tmp_max = tmp;
                tmp_idx = L + split;
            }

        }
    }

    for(int split = gap_end+1; split < end; split++){
        if( (split ) == (gap_end+1)){
        float tmp = s_c[gap_end*L + split] + s_d[start*L3 + gap_start*L2 + split*L + end];
        if ( tmp > tmp_max){
           tmp_max = tmp;
           tmp_idx = 2*L + split;
         }
        }
        if((split+1)==end){
         float tmp = s_c[split*L + end] + s_d[start*L3 + gap_start*L2 + gap_end*L + split];
         if (tmp > tmp_max){
            tmp_max = tmp;
            tmp_idx = 3*L + split;
         }
         }
    }
    alpha_d[b_idx*L4+start*L3+gap_start*L2+gap_end*L+end] = tmp_max + s_span_d[b_idx*L4+start*L3+gap_start*L2+gap_end*L+end];
    alpha_d[b_idx*L4+gap_start*L3+start*L2+gap_end*L+end] = tmp_idx;
}


__global__ void kernel_argmax_d_on3(const float* const s_span_d,
                                    float* alpha_c,
                                    float* alpha_d,
                                    int B,  int L,  int width
                                   )
{
    const int b_idx = blockIdx.x ;
    const int gap_start_minus_start = blockIdx.y;
    const int gap_width = blockIdx.z + 1;
    const int start = threadIdx.x;
    const int end = start + width + gap_width;

    if(end>=L){
        return;
    }

    const int gap_start = start + gap_start_minus_start + 1;
    const int gap_end = gap_start + gap_width;

    if(gap_start - start >width-1)
    {
        return;
    }

    const int L2 = L*L;
    const int L3 = L2*L;
    const int L4 = L3*L;
    const float* const s_d = alpha_d + b_idx * L4;
    const float* const s_c = alpha_c + b_idx * L2;


    float tmp_max = s_c[start*L + gap_start] + s_c[gap_end*L + end];
    float tmp_idx = -1;

    alpha_d[b_idx*L4+start*L3+gap_start*L2+gap_end*L+end] = tmp_max + s_span_d[b_idx*L4+start*L3+gap_start*L2+gap_end*L+end];
    alpha_d[b_idx*L4+gap_start*L3+start*L2+gap_end*L+end] = tmp_idx;
}


void cuda_argmax(float *s_span_c, float *s_span_d, float *alpha_c, float *alpha_d, int B, int L)
{
    for(int w=2; w<L; w++){
        dim3 gridDim(B);
        dim3 blockDim(L-w);
        kernel_argmax_c<<<gridDim, blockDim>>>(s_span_c,alpha_c,alpha_d, B, L, w);
        if(w<L-1){
            dim3 gridDim3(B,(w-1), (L-w));
            dim3 blockDim3(L-w);
            kernel_argmax_d<<<gridDim3, blockDim3>>>(s_span_d, alpha_c, alpha_d, B, L, w);
        }
    }
}



void cuda_argmax_on3(float *s_span_c, float *s_span_d, float *alpha_c, float *alpha_d, int B, int L)
{
    for(int w=2; w<L; w++){
        dim3 gridDim(B);
        dim3 blockDim(L-w);
        kernel_argmax_c<<<gridDim, blockDim>>>(s_span_c,alpha_c,alpha_d, B, L, w);
        if(w<L-1){
            dim3 gridDim3(B,(w-1), (L-w));
            dim3 blockDim3(L-w);
            kernel_argmax_d_on3<<<gridDim3, blockDim3>>>(s_span_d, alpha_c, alpha_d, B, L, w);
        }
    }
}


void cuda_argmax_on4(float *s_span_c, float *s_span_d, float *alpha_c, float *alpha_d, int B, int L)
{
    for(int w=2; w<L; w++){
        dim3 gridDim(B);
        dim3 blockDim(L-w);
        kernel_argmax_c<<<gridDim, blockDim>>>(s_span_c,alpha_c,alpha_d, B, L, w);
        if(w<L-1){
            dim3 gridDim3(B,(w-1), (L-w));
            dim3 blockDim3(L-w);
            kernel_argmax_d_on4<<<gridDim3, blockDim3>>>(s_span_d, alpha_c, alpha_d, B, L, w);
        }
    }
}


