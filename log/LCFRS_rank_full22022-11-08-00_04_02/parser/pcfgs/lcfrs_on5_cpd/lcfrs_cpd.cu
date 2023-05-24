#include <stdio.h>
#include <cuda_runtime.h>


template <typename scalar_t>
__device__ scalar_t logsumexp(scalar_t a, scalar_t b) {
    scalar_t m = max(a, b);
  return log(exp(a - m) + exp(b - m)) + m;
}


template <typename F>
__global__ void kernel_forward( F *__restrict__   head_rule,
                                F *__restrict__ alpha,
                                F *__restrict__ alpha_l,
                                F *__restrict__ alpha_r,
                                  int B,   int L,   int width,   int m,   int r)
{
      int b_idx = blockIdx.x;
      int start = blockIdx.y;
      int s_A = blockIdx.z;
      int s_R = threadIdx.x;

    if(start+width >= L){
        return;
    }

    __shared__ float result[1000];

    int end = start + width;
      F *__restrict__   s_l = alpha_l + b_idx * L * L * r;
      F *__restrict__   s_r = alpha_r + b_idx * L * L * r;

    float tmp_result = logf(0);

    float rule_score = head_rule[b_idx*m*r + s_A*r + s_R];

    for (int split = start+1; split < start+width; split++)
    {
        tmp_result = logsumexp(tmp_result, s_l[start*L*r + split * r + s_R] + s_r[split*L*r + end*r + s_R] + rule_score);
    }

    result[s_R] = tmp_result;
    __syncthreads();

    if(s_R==0){
        float final_result = logf(0);
        for(int i=0;i<r;i++){
            final_result = logsumexp(final_result, result[i]);
        }
    alpha[b_idx*L*L*m+start*L*m+end*m+s_A] = final_result;
  }
}


template <typename F>
__global__ void kernel_backward( F *__restrict__  head_rule,
                                F *__restrict__  head_rule_grd,
                                F *__restrict__ alpha,
                                F *__restrict__ alpha_l,
                                F *__restrict__ alpha_r,
                                int B,   int L,   int width,   int m,   int r)
{
      int b_idx = blockIdx.x;
      int start = blockIdx.y;
      int s_A = blockIdx.z;
      int s_R = threadIdx.x;

      if(start+width >= L){
        return;
     }

     int end = start + width;
     F *__restrict__   s_l = alpha_l + b_idx * L * L * r;
     F *__restrict__   s_r = alpha_r + b_idx * L * L * r;
     float tmp_result = logf(0);
     float rule_score = head_rule[b_idx*m*r + s_A*r + s_R];
     float rule_score_grd = 0;
     float parent_inside = alpha[b_idx*L*L*m+start*L*m+end*m+s_A];
     float parent_grd  = alpha[b_idx*L*L*m+end*L*m+start*m+s_A];

     for (int split = start+1; split < start+width; split++)
     {
        float tmp = exp(s_l[start*L*r + split * r + s_R] + s_r[split*L*r + end*r + s_R] + rule_score - parent_inside) * parent_grd;
        rule_score_grd += tmp;
        atomicAdd(s_l + split*L*r + start*r + s_R, tmp);
        atomicAdd(s_r + end*L*r + split*r + s_R, tmp);
     }
     atomicAdd(head_rule_grd + b_idx*m*r + s_A*r + s_R, rule_score_grd);
}




template <typename F>
__global__ void kernel_forward_close(  F *__restrict__   head_rule,
                                 F *__restrict__ alpha,
                                 F *__restrict__ alpha_cd,
                                 F *__restrict__ alpha_cc,
                                   int B,   int L,   int width,   int m,    int r)
{
      int b_idx = blockIdx.x;
      int start = blockIdx.y;
      int s_A = blockIdx.z;
      int s_R = threadIdx.x;

    if(start+width >= L){
        return;
    }

    __shared__ float result[1000];
    __syncthreads();

    int end = start + width;
      F *__restrict__   s = alpha_cc + b_idx * L * L * r;
      F *__restrict__   s_d = alpha_cd + b_idx * L * L * L * L * r;

      int L3 = L * L * L * r;
      int L2 = L * L * r;
      int L1 = L * r;

    float tmp_result = logf(0);
    float rule_score = head_rule[b_idx*m*r + s_A*r + s_R];

    for (int split = start+1; split < start+width-1; split++)
    {
         for(int split2 = split+1; split2 < start+width; split2++)
         {
              tmp_result = logsumexp(tmp_result,s[split*L*r + split2*r + s_R] + s_d[start*L3 + split*L2 + split2*L1 + end*r + s_R] + rule_score);
         }
    }

    result[s_R] = tmp_result;
    __syncthreads();

    if(s_R==0){
        float final_result = logf(0);
        for(int i=0;i<r;i++){
            final_result = logsumexp(final_result, result[i]);
        }
        alpha[b_idx*L*L*m+start*L*m+(end)*m+s_A] = logsumexp(alpha[b_idx*L*L*m+start*L*m+(end)*m+s_A], final_result);
    }
}

template <typename F>
__global__ void kernel_backward_close(const F *__restrict__ const head_rule,
                                  F *__restrict__  head_rule_grd,
                                 F *__restrict__ alpha,
                                 F *__restrict__ alpha_cd,
                                 F *__restrict__ alpha_cc,
                                 const int B, const int L, const int width, const int m, const int r)
{
    const int b_idx = blockIdx.x;
    const int start = blockIdx.y;
    const int s_A = blockIdx.z;
    const int s_R = threadIdx.x;

    if(start+width >= L){
        return;
    }

    int end = start + width;
    F *__restrict__  s = alpha_cc + b_idx * L * L * r;
    F *__restrict__  s_d = alpha_cd + b_idx * L * L * L * L * r;

    const int L3 = L * L * L * r;
    const int L2 = L * L * r;
    const int L1 = L * r;
    float rule_score = head_rule[b_idx*m*r + s_A*r + s_R];
    float parent_inside = alpha[b_idx*L*L*m+start*L*m+(end)*m+s_A];
    float parent_grd = alpha[b_idx*L*L*m+end*L*m+(start)*m+s_A];
    float rule_score_grd = 0;

    for (int split = start+1; split < start+width-1; split++)
    {
         for(int split2 = split+1; split2 < start+width; split2++)
         {
              float tmp = exp(s[split*L*r + split2*r + s_R] + s_d[start*L3 + split*L2 + split2*L1 + end*r + s_R] + rule_score - parent_inside) * parent_grd;
              rule_score_grd += tmp;
              atomicAdd(s + split2*L*r + split*r + s_R, tmp);
              atomicAdd(s_d + split*L3 + start*L2 + split2*L1 + end*r + s_R, tmp);
         }
    }
    atomicAdd(head_rule_grd + b_idx*m*r + s_A*r + s_R, rule_score_grd);
}


template <typename F>
__global__ void kernel_forward_d1(  F *__restrict__   head,
                                    F *__restrict__ alpha_dl,
                                    F *__restrict__ alpha_dr,
                                    F *__restrict__ alpha_d,
                                    int B,   int L,   int width,   int d,   int r)
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
    __syncthreads();
    F *__restrict__   s_l = alpha_dl + b_idx * L * L * r;
    F *__restrict__   s_r = alpha_dr + b_idx * L * L * r;

    result[s_R] = s_l[start*L*r + gap_start*r + s_R] + s_r[gap_end*L*r + end*r + s_R] + head[b_idx*d*r + s_A*r + s_R];
    __syncthreads();
    if(s_R==0){
    float final_result = logf(0);
    for(int i=0;i<r;i++){
        final_result = logsumexp(final_result, result[i]);
    }
    alpha_d[b_idx*L*L*L*L*d + start*L*L*L*d + gap_start*L*L*d + gap_end*L*d + end*d + s_A] = final_result;
    }
}



template <typename F>
__global__ void kernel_backward_d1(  F *__restrict__   head,
                                 F *__restrict__   head_grd,
                                F *__restrict__ alpha_dl,
                                F *__restrict__ alpha_dr,
                                F *__restrict__ alpha_d,
                                  int B,   int L,   int width,   int d,   int r)
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
     F *__restrict__   s_l = alpha_dl + b_idx * L * L * r;
     F *__restrict__   s_r = alpha_dr + b_idx * L * L * r;
     float parent_inside = alpha_d[b_idx*L*L*L*L*d + start*L*L*L*d + gap_start*L*L*d + gap_end*L*d + end*d + s_A];
     float parent_grd = alpha_d[b_idx*L*L*L*L*d + gap_start*L*L*L*d + start*L*L*d + gap_end*L*d + end*d + s_A];
     float tmp = exp(s_l[start*L*r + gap_start*r + s_R] + s_r[gap_end*L*r + end*r + s_R] + head[b_idx*d*r + s_A*r + s_R] - parent_inside) * parent_grd;
     atomicAdd(s_l + gap_start*L*r + start*r + s_R, tmp);
     atomicAdd(s_r + end*L*r + gap_end*r + s_R, tmp);
     atomicAdd(head_grd + b_idx*d*r + s_A*r + s_R, tmp);
}


template <typename F>
__global__ void kernel_forward_d2(F *__restrict__  head,
                                F *__restrict__ alpha_dc,
                                F *__restrict__ alpha_dd,
                                F *__restrict__ alpha_d,
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

    F *__restrict__   s = alpha_dc + b_idx * L * L * r * 4;
    F *__restrict__   s_d = alpha_dd + b_idx * L * L * L * L * r;
    int L3 = L * L * L * r;
    int L2 = L * L * r;
    int L1 = L * r;
    float tmp_result = logf(0);
    float rule_score = head[b_idx*d*r + s_A*r + s_R];
  __shared__ float result[1000];
__syncthreads();
    for(int split=start+1; split< gap_start; split++)
    {
        tmp_result = logsumexp(tmp_result, s_d[split*L3 + gap_start*L2 + gap_end*L1 + end*r + s_R] + rule_score + s[start*L*r*4 + split*r*4 +  0*r + s_R]);
        tmp_result = logsumexp(tmp_result, s_d[start*L3 + split*L2 + gap_end*L1 + end*r + s_R] + rule_score + s[split*L*r*4 + gap_start*r*4 + 1*r + s_R]);
    }

    for(int split = gap_end+1; split <end; split++){
        tmp_result = logsumexp(tmp_result, s_d[start*L3 + gap_start*L2 + split*L1 + end*r + s_R] + rule_score + s[gap_end*L*r*4 + split*r*4 + 2*r + s_R]);
        tmp_result = logsumexp(tmp_result, s_d[start*L3 + gap_start*L2 + gap_end*L1 + split*r + s_R] + rule_score + s[split*L*r*4 + end*r*4 + 3*r + s_R]);
    }
    result[s_R] = tmp_result;

    __syncthreads();
    if(s_R==0){
    float final_result = logf(0);
    for(int i=0;i<r;i++){
        final_result = logsumexp(final_result, result[i]);
    }

    float result = logsumexp(alpha_d[b_idx*L*L*L*L*d + start*L*L*L*d + gap_start*L*L*d + gap_end*L*d + end*d + s_A], final_result);
    alpha_d[b_idx*L*L*L*L*d + start*L*L*L*d + gap_start*L*L*d + gap_end*L*d + end*d + s_A] = logsumexp(alpha_d[b_idx*L*L*L*L*d + start*L*L*L*d + gap_start*L*L*d + gap_end*L*d + end*d + s_A], final_result);

   }
}


template <typename F>
__global__ void kernel_backward_d2(const F *__restrict__ const head_rule,
                                 F *__restrict__  head_rule_grd,
                                F *__restrict__ alpha_dc,
                                F *__restrict__ alpha_dd,
                                F *__restrict__ alpha_d,
                                const int B, const int L, const int width, const int d, const int r)
{
    const int b_idx = blockIdx.x / (L-width);
    const int start = blockIdx.x % (L-width);
    const int gap_start_minus_start = (blockIdx.y) / (L-width);
    const int gap_width =  ((blockIdx.y) % ( L-width)) + 1;
    const int end = start + width + gap_width;
    if(end>=L){
        return;
    }
    const int s_A = (blockIdx.z);
    const int s_R = threadIdx.x;
    const int gap_start = start + gap_start_minus_start + 1;
    const int gap_end = gap_start + gap_width;
    if(gap_start - start > width-1)
    {
        return;
    }

    F *__restrict__ s = alpha_dc + b_idx * L * L * r * 4;
    F *__restrict__ s_d = alpha_dd + b_idx * L * L * L * L * r;
    int L3 = L * L * L * d;
    int L2 = L * L * d;
    int L1 = L * d;
    float tmp_result = logf(0);
    float rule_score = head_rule[b_idx*d*r + s_A*r + s_R];
    float rule_grd = 0;
    float parent_inside = alpha_d[b_idx*L*L*L*L*d + start*L3 + gap_start*L2 + gap_end*L1 + end*d + s_A];
    float parent_grd = alpha_d[b_idx*L*L*L*L*d + gap_start*L3 + start*L2 + gap_end*L1 + end*d + s_A];
    L3 = L*L*L*r;
    L2 = L*L*r;
    L1 = L*r;
    for(int split=start+1; split< gap_start; split++)
    {
        float tmp = exp(s_d[split*L3 + gap_start*L2 + gap_end*L1 + end*r + s_R] + rule_score + s[start*L*r*4 + split*r*4 + 0*r + s_R] - parent_inside) * parent_grd;
        rule_grd += tmp;
        atomicAdd(s + split*L*r*4 + start*r*4 + s_R , tmp);
        atomicAdd(s_d +  gap_start*L3 + split*L2 + gap_end*L1 + end*r + s_R, tmp);

        tmp = exp(s_d[start*L3 + split*L2 + gap_end*L1 + end*r + s_R] + rule_score + s[split*L*r*4 + gap_start*r*4 + 1*r + s_R] - parent_inside) * parent_grd;
        rule_grd += tmp;
        atomicAdd(s + gap_start*L*r*4 + split*r*4 + s_R + r*1, tmp);
        atomicAdd(s_d +  split*L3 + start*L2 + gap_end*L1 + end*r + s_R, tmp);
    }

    for(int split=gap_end+1; split <end; split++){
        float tmp = exp(s_d[start*L3 + gap_start*L2 + split*L1 + end*r + s_R] + rule_score + s[gap_end*L*r*4 + split*r*4 + 2*r + s_R] - parent_inside) * parent_grd;
        rule_grd+=tmp;
        atomicAdd(s_d + gap_start*L3 + start*L2 + split*L1 + end*r + s_R, tmp);
        atomicAdd(s + split*L*r*4 + gap_end*r*4 + 2*r + s_R , tmp);

        tmp = exp(s_d[start*L3 + gap_start*L2 + gap_end*L1 + split*r + s_R] + rule_score + s[split*L*r*4 + end*r*4 + 3*r + s_R] - parent_inside) * parent_grd;
        rule_grd+=tmp;
        atomicAdd(s_d + gap_start*L3 + start*L2 + gap_end*L1 + split*r + s_R, tmp);
        atomicAdd(s + end*L*r*4 + split*r*4 + 3*r + s_R, tmp);
    }
    atomicAdd(head_rule_grd + b_idx*d*r + s_A*r + s_R, rule_grd);
}



template <typename F>
__global__ void kernel_forward_dr(  F *__restrict__   dd,
                                    F *__restrict__ alpha_d,
                                    F *__restrict__ alpha_dd,
                                    int B,   int L,   int width,   int d,   int r)
{
      int b_idx = blockIdx.x / (L-width);
      int start = blockIdx.x % (L-width);
      int gap_start_minus_start = (blockIdx.y) / (L-width);
      int gap_width =  ((blockIdx.y) % ( L-width)) + 1;
      int end = start + width + gap_width;
      if(end>=L){
        return;
      }
      int gap_start = start + gap_start_minus_start + 1;
      int gap_end = gap_start + gap_width;
      int s_R = (blockIdx.z);
      int s_B = threadIdx.x;
      if(gap_start - start > width-1)
      {
        return;
      }

      __shared__ float result[1000];
      result[s_B] = alpha_d[b_idx*L*L*L*L*d + start*L*L*L*d + gap_start*L*L*d + gap_end*L*d + end*d + s_B] + dd[b_idx*d*r + s_B*r + s_R];
      __syncthreads();
      if(s_B==0){
      float final_result = logf(0);
      for(int i=0;i<d;i++){
            final_result = logsumexp(final_result, result[i]);
      }
       alpha_dd[b_idx*L*L*L*L*r + start*L*L*L*r + gap_start*L*L*r + gap_end*L*r + end*r + s_R] = final_result;
      }
}


template <typename F>
__global__ void kernel_backward_dr(  F *__restrict__   dd,
                                F *__restrict__   dd_grd,
                                F *__restrict__ alpha_d,
                                F *__restrict__ alpha_dd,
                                  int B,   int L,   int width,   int d,   int r)
{
      int b_idx = blockIdx.x / (L-width);
      int start = blockIdx.x % (L-width);
      int gap_start_minus_start = (blockIdx.y) / (L-width);
      int gap_width =  ((blockIdx.y) % ( L-width)) + 1;
      int end = start + width + gap_width;
      if(end>=L){
        return;
      }
      int gap_start = start + gap_start_minus_start + 1;
      int gap_end = gap_start + gap_width;
      int s_R = (blockIdx.z);
      int s_B = threadIdx.x;
      if(gap_start - start > width-1)
      {
        return;
      }
      float parent_inside = alpha_dd[b_idx*L*L*L*L*r + start*L*L*L*r + gap_start*L*L*r + gap_end*L*r + end*r + s_R];
      float parent_grd = alpha_dd[b_idx*L*L*L*L*r + gap_start*L*L*L*r + start*L*L*r + gap_end*L*r + end*r + s_R];
      float tmp = exp(alpha_d[b_idx*L*L*L*L*d + start*L*L*L*d + gap_start*L*L*d + gap_end*L*d + end*d + s_B] + dd[b_idx*d*r + s_B*r + s_R] - parent_inside)*parent_grd;
      atomicAdd(alpha_d + b_idx*L*L*L*L*d + gap_start*L*L*L*d + start*L*L*d + gap_end*L*d + end*d + s_B, tmp);
      atomicAdd(dd_grd + b_idx*d*r + s_B*r + s_R, tmp);
}



template <typename F>
__global__ void kernel_forward_cmr(  F *__restrict__   cc,
                                    F *__restrict__ alpha,
                                    F *__restrict__ alpha_cc,
                                      int B,   int L,   int width,   int m,   int p,   int r)
{
      int b_idx = blockIdx.x;
    int start = blockIdx.y;
    int s_R = blockIdx.z;
    int s_B = threadIdx.x;

    int end = start + width;
    if(end>=L){
        return;
    }
    __shared__ float result[1000];
    __syncthreads();
    float inside = alpha[b_idx*L*L*m + start*L*m + end*m + s_B];
    result[s_B] = inside + cc[b_idx*(m+p)*r + s_B*r + s_R];
    __syncthreads();

    if(s_B==0){
    float final_result = logf(0);
        for(int i=0;i<m;i++){
            final_result = logsumexp(final_result, result[i]);
        }
    alpha_cc[b_idx*L*L*r + start*L*r + end*r + s_R] = final_result;
   }
}


template <typename F>
__global__ void kernel_backward_cmr(  F *__restrict__   cc,  F *__restrict__   cc_grd,
                                    F *__restrict__ alpha,
                                    F *__restrict__ alpha_cc,
                                      int B,   int L,   int width,   int m,   int p,   int r)
{
    int b_idx = blockIdx.x;
    int start = blockIdx.y;
    int s_R = blockIdx.z;
    int s_B = threadIdx.x;
    int end = start + width;
    if(end>=L){
        return;
    }
    float parent_inside = alpha_cc[b_idx*L*L*r + start*L*r + end*r + s_R];
    float parent_grd = alpha_cc[b_idx*L*L*r + end*L*r + start*r + s_R];
    float tmp = exp(alpha[b_idx*L*L*m + start*L*m + end*m + s_B] + cc[b_idx*(m+p)*r + s_B*r + s_R] - parent_inside) * parent_grd;
    atomicAdd(alpha + b_idx*L*L*m + end*L*m + start*m + s_B, tmp);
    atomicAdd(cc_grd + b_idx*(m+p)*r + s_B*r + s_R, tmp);
}



template <typename F>
__global__ void kernel_forward_cpr(  F *__restrict__   cc,
                                    F *__restrict__ alpha_cc,
                                    F *__restrict__ unary,
                                      int B,   int L, int m,    int p,   int r)
{
      int b_idx = blockIdx.x;
      int start = blockIdx.y;
      int s_R = blockIdx.z;
      int s_B = threadIdx.x;
      int end = start + 1;
    if(end>=L){
        return;
    }
    __shared__ float result[1000];
    result[s_B] = unary[b_idx*p*(L-1) + start*p + s_B] + cc[b_idx*(m+p)*r + (s_B+m)*r + s_R];
    __syncthreads();
    if(s_B==0){
    float final_result = logf(0);
        for(int i=0;i<p;i++){
            final_result = logsumexp(final_result, result[i]);
        }
        alpha_cc[b_idx*L*L*r + start*L*r + end*r + s_R] = final_result;
   }
}

template <typename F>
__global__ void kernel_backward_cpr(  F *__restrict__   cc,  F *__restrict__   cc_grd,
                                    F *__restrict__ alpha_cc,
                                    F *__restrict__ unary,  F *__restrict__ unary_grd,
                                      int B,   int L, int m,    int p,   int r)
{
      int b_idx = blockIdx.x;
      int start = blockIdx.y;
      int s_R = blockIdx.z;
      int s_B = threadIdx.x;
      int end = start + 1;
    if(end>=L){
        return;
    }

    float parent_inside = alpha_cc[b_idx*L*L*r + start*L*r + end*r + s_R];
    float parent_grd = alpha_cc[b_idx*L*L*r + end*L*r + start*r + s_R];
    float tmp = exp(unary[b_idx*p*(L-1) + start*p + s_B] + cc[b_idx*(m+p)*r + (s_B+m)*r + s_R] - parent_inside) * parent_grd;
    atomicAdd(unary_grd + b_idx*p*(L-1) + start*p + s_B, tmp);
    atomicAdd(cc_grd +  b_idx*(m+p)*r + (s_B+m)*r + s_R, tmp);
}

template <typename F>
__global__ void kernel_forward_dpr( F *__restrict__   cc,
                                    F *__restrict__ alpha_cc,
                                    F *__restrict__ unary,
                                    int B,   int L,  int m, int p,   int r)
{
      int b_idx = blockIdx.x;
      int start = blockIdx.y;
      int s_R = blockIdx.z;
      int s_B = threadIdx.x;

      int end = start + 1;
    if(end>=L){
        return;
    }
    __shared__ float result1[1000];
    __shared__ float result2[1000];
    __shared__ float result3[1000];
    __shared__ float result4[1000];
        __syncthreads();
    float u = unary[b_idx*p*(L-1) + start*p + s_B];
    result1[s_B] = u + cc[b_idx*(m+p)*r*4 + (s_B+m)*r*4 + s_R];
    result2[s_B] = u + cc[b_idx*(m+p)*r*4 + (s_B+m)*r*4 + 1*r + s_R];
    result3[s_B] = u + cc[b_idx*(m+p)*r*4 + (s_B+m)*r*4 + 2*r + s_R];
    result4[s_B] = u + cc[b_idx*(m+p)*r*4 + (s_B+m)*r*4 + 3*r + s_R];
    __syncthreads();
    if(s_B==0){
    float final_result1 = logf(0);
    float final_result2 = logf(0);
    float final_result3 = logf(0);
    float final_result4 = logf(0);
    for(int i=0;i<p;i++){
        final_result1 = logsumexp(final_result1, result1[i]);
        final_result2 = logsumexp(final_result2, result2[i]);
        final_result3 = logsumexp(final_result3, result3[i]);
        final_result4 = logsumexp(final_result4, result4[i]);
    }
    alpha_cc[b_idx*L*L*r*4 + start*L*r*4 + end*r*4 + 0*r + s_R] = final_result1;
    alpha_cc[b_idx*L*L*r*4 + start*L*r*4 + end*r*4 + 1*r + s_R ] = final_result2;
    alpha_cc[b_idx*L*L*r*4 + start*L*r*4 + end*r*4 + 2*r + s_R] = final_result3;
    alpha_cc[b_idx*L*L*r*4 + start*L*r*4 + end*r*4 + 3*r + s_R] = final_result4;
   }
}

template <typename F>
__global__ void kernel_backward_dpr( F *__restrict__  cc, F *__restrict__  cc_grd,
                                    F *__restrict__ alpha_cc,
                                    F *__restrict__ unary,  F *__restrict__ unary_grd,
                                    int B,   int L,  int m, int p,   int r)
{
      int b_idx = blockIdx.x;
      int start = blockIdx.y;
      int s_R = blockIdx.z;
      int s_B = threadIdx.x;

      int end = start + 1;
    if(end>=L){
        return;
    }
    float u = unary[b_idx*p*(L-1) + start*p + s_B];
    float u_grd = 0;

    float parent_inside = alpha_cc[b_idx*L*L*r*4 + start*L*r*4 + end*r*4 + 0*r + s_R];
    float parent_grd = alpha_cc[b_idx*L*L*r*4 + end*L*r*4 + start*r*4 + 0*r + s_R];
    float tmp = exp(u + cc[b_idx*(m+p)*r*4 + (s_B+m)*r*4 + s_R] - parent_inside) * parent_grd;
    atomicAdd(cc_grd + b_idx*(m+p)*r*4 + (s_B+m)*r*4 + s_R, tmp);
    u_grd += tmp;

    parent_inside = alpha_cc[b_idx*L*L*r*4 + start*L*r*4 + end*r*4 + 1*r + s_R];
    parent_grd = alpha_cc[b_idx*L*L*r*4 + end*L*r*4 + start*r*4 + 1*r + s_R];
    tmp = exp(u + cc[b_idx*(m+p)*r*4 + (s_B+m)*r*4 + 1*r + s_R]-parent_inside) * parent_grd;
    atomicAdd(cc_grd + b_idx*(m+p)*r*4 + (s_B+m)*r*4 + 1*r + s_R, tmp);
    u_grd += tmp;

    parent_inside = alpha_cc[b_idx*L*L*r*4 + start*L*r*4 + end*r*4 + 2*r + s_R];
    parent_grd = alpha_cc[b_idx*L*L*r*4 + end*L*r*4 + start*r*4 + 2*r + s_R];
    tmp = exp(u + cc[b_idx*(m+p)*r*4 + (s_B+m)*r*4 + 2*r + s_R] - parent_inside)*parent_grd;
    atomicAdd(cc_grd + b_idx*(m+p)*r*4 + (s_B+m)*r*4 + 2*r + s_R, tmp);
    u_grd += tmp;

    parent_inside = alpha_cc[b_idx*L*L*r*4 + start*L*r*4 + end*r*4 +3*r + s_R];
    parent_grd = alpha_cc[b_idx*L*L*r*4 + end*L*r*4 + start*r*4 + 3*r + s_R];
    tmp = exp(u + cc[b_idx*(m+p)*r*4 + (s_B+m)*r*4 + 3*r + s_R] - parent_inside)*parent_grd;
    atomicAdd(cc_grd + b_idx*(m+p)*r*4 + (s_B+m)*r*4 + 3*r + s_R, tmp);
    u_grd += tmp;
    atomicAdd(unary_grd + b_idx*p*(L-1) + start*p + s_B, u_grd);
}



template <typename F>
__global__ void kernel_forward_dmr(  F *__restrict__   cc,
                                    F *__restrict__ alpha_cc,
                                    F *__restrict__ alpha,
                                      int B,   int L,   int width, int m,  int p,   int r)
{
      int b_idx = blockIdx.x;
      int start = blockIdx.y;
      int s_R = blockIdx.z;
      int s_B = threadIdx.x;
      int end = start + width;
    if(end>=L){
        return;
    }
    __shared__ float result1[1000];
    __shared__ float result2[1000];
    __shared__ float result3[1000];
    __shared__ float result4[1000];
    float u = alpha[b_idx*L*L*m + start*L*m + end*m + s_B];
    result1[s_B] = u + cc[b_idx*(m+p)*r*4 + (s_B)*r*4 + s_R ];
    result2[s_B] = u + cc[b_idx*(m+p)*r*4 + (s_B)*r*4 + 1*r + s_R];
    result3[s_B] = u + cc[b_idx*(m+p)*r*4 + (s_B)*r*4 + 2*r + s_R];
    result4[s_B] = u + cc[b_idx*(m+p)*r*4 + (s_B)*r*4 + 3*r + s_R];
    __syncthreads();

    if(s_B==0){
    float final_result1 = logf(0);
    float final_result2 = logf(0);
    float final_result3 = logf(0);
    float final_result4 = logf(0);
    for(int i=0;i<m;i++){
        final_result1 = logsumexp(final_result1, result1[i]);
        final_result2 = logsumexp(final_result2, result2[i]);
        final_result3 = logsumexp(final_result3, result3[i]);
        final_result4 = logsumexp(final_result4, result4[i]);
    }
    alpha_cc[b_idx*L*L*r*4 + start*L*r*4 + end*r*4 + s_R ] = final_result1;
    alpha_cc[b_idx*L*L*r*4 + start*L*r*4 + end*r*4 + 1*r + s_R ] = final_result2;
    alpha_cc[b_idx*L*L*r*4 + start*L*r*4 + end*r*4 + 2*r + s_R] = final_result3;
    alpha_cc[b_idx*L*L*r*4 + start*L*r*4 + end*r*4 + 3*r + s_R] = final_result4;
   }
}

template <typename F>
__global__ void kernel_backward_dmr( F *__restrict__  cc, F *__restrict__  cc_grd,
                                    F *__restrict__ alpha_cc,
                                F *__restrict__ alpha,
                                    int B,   int L,   int width,  int m, int p,   int r)
{
      int b_idx = blockIdx.x;
      int start = blockIdx.y;
      int s_R = blockIdx.z;
      int s_B = threadIdx.x;

      int end = start + width;
      if(end>=L){
        return;
     }
     float u = alpha[b_idx*L*L*m + start*L*m + end*m + s_B];
     float u_grd = 0;

     float parent_inside = alpha_cc[b_idx*L*L*r*4 + start*L*r*4 + end*r*4 + 0*r + s_R];
     float parent_grd = alpha_cc[b_idx*L*L*r*4 + end*L*r*4 + start*r*4 + 0*r + s_R];
     float tmp = exp(u + cc[b_idx*(m+p)*r*4 + (s_B)*r*4 + s_R] - parent_inside) * parent_grd;
     atomicAdd(cc_grd + b_idx*(m+p)*r*4 + (s_B)*r*4 + s_R, tmp);
     u_grd += tmp;

    parent_inside = alpha_cc[b_idx*L*L*r*4 + start*L*r*4 + end*r*4 + 1*r + s_R];
    parent_grd = alpha_cc[b_idx*L*L*r*4 + end*L*r*4 + start*r*4 + 1*r + s_R];
    tmp = exp(u + cc[b_idx*(m+p)*r*4 + (s_B)*r*4 + 1*r + s_R]-parent_inside) * parent_grd;
    atomicAdd(cc_grd + b_idx*(m+p)*r*4 + (s_B)*r*4 + 1*r + s_R, tmp);
    u_grd += tmp;

    parent_inside = alpha_cc[b_idx*L*L*r*4 + start*L*r*4 + end*r*4 + 2*r + s_R];
    parent_grd = alpha_cc[b_idx*L*L*r*4 + end*L*r*4 + start*r*4 + 2*r + s_R];
    tmp = exp(u + cc[b_idx*(m+p)*r*4 + (s_B)*r*4 + 2*r + s_R] - parent_inside)*parent_grd;
    atomicAdd(cc_grd + b_idx*(m+p)*r*4 + (s_B)*r*4 + 2*r + s_R, tmp);
    u_grd += tmp;

    parent_inside = alpha_cc[b_idx*L*L*r*4 + start*L*r*4 + end*r*4 +3*r + s_R];
    parent_grd = alpha_cc[b_idx*L*L*r*4 + end*L*r*4 + start*r*4 + 3*r + s_R];
    tmp = exp(u + cc[b_idx*(m+p)*r*4 + (s_B)*r*4 + 3*r + s_R] - parent_inside)*parent_grd;
    atomicAdd(cc_grd + b_idx*(m+p)*r*4 + (s_B)*r*4 + 3*r + s_R, tmp);
    u_grd += tmp;
    atomicAdd(alpha + b_idx*L*L*m + end*L*m + start*m + s_B, u_grd);
}


void cuda_forward( float *head_c1,   float *head_c2,   float *head_d1,   float *head_d2,
                   float *left_c,   float *right_c,
                   float *left_d,   float *right_d,
                   float *cc,   float *cd,   float *dc,   float *dd,   float *unary,
                   float *alpha,  float *alpha_d, float *alpha_lc, float *alpha_rc, float *alpha_ld, float *alpha_rd,
                   float *alpha_cc, float *alpha_cd, float *alpha_dc, float *alpha_dd,
                   int B, int L, int m, int p, int d, int r1, int r2, int r3, int r4)
{
    dim3 gridDim(B, L-1, r1);
    dim3 blockDim(p);
    kernel_forward_cpr<<<gridDim, blockDim>>>(right_c, alpha_rc, unary, B, L, m, p, r1);
    kernel_forward_cpr<<<gridDim, blockDim>>>(left_c, alpha_lc, unary, B, L, m, p, r1);
    dim3 gridDim11(B, L-1, r2);
    kernel_forward_cpr<<<gridDim11, blockDim>>>(cc, alpha_cc, unary, B, L, m, p, r2);
    dim3 gridDim12(B, L-1, r3);
    kernel_forward_cpr<<<gridDim12, blockDim>>>(left_d, alpha_ld, unary, B, L, m, p, r3);
    kernel_forward_cpr<<<gridDim12, blockDim>>>(right_d, alpha_rd, unary, B, L, m, p, r3);

    dim3 gridDim13(B, L-1, r4);
    kernel_forward_dpr<<<gridDim13, blockDim>>>(dc, alpha_dc, unary, B, L, m, p, r4);

    for(int w=2; w<L; w++){
      dim3 gridDim(B, L-w, m);
      dim3 blockDim(r1);
      kernel_forward<<<gridDim, blockDim>>>(head_c1, alpha, alpha_lc, alpha_rc, B, L, w, m, r1);
      if (w>2){
        dim3 gridDim2(B, L-w, m);
        dim3 blockDim2(r2);
        kernel_forward_close<<<gridDim2, blockDim2>>>(head_c2,  alpha, alpha_cd, alpha_cc, B, L, w, m, r2);
      }
      if(w<L-1){
        dim3 gridDim6(B, L-w, r1);
        dim3 blockDim6(m);
        kernel_forward_cmr<<<gridDim6, blockDim6>>>(left_c, alpha, alpha_lc, B, L, w, m, p, r1);
        kernel_forward_cmr<<<gridDim6, blockDim6>>>(right_c, alpha, alpha_rc, B, L, w, m, p, r1);

        dim3 gridDim7(B, L-w, r3);
        kernel_forward_cmr<<<gridDim7, blockDim6>>>(left_d, alpha, alpha_ld,  B, L, w, m, p, r3);
        kernel_forward_cmr<<<gridDim7, blockDim6>>>(right_d, alpha, alpha_rd,  B, L, w, m, p, r3);

        dim3 gridDim8(B, L-w, r2);
        dim3 blockDim8(m);
        kernel_forward_cmr<<<gridDim8, blockDim8>>>(cc, alpha, alpha_cc, B, L, w, m, p, r2);

        dim3 gridDim9(B, L-w, r4);
        dim3 blockDim9(m);
        kernel_forward_dmr<<<gridDim9, blockDim9>>>(dc, alpha_dc, alpha, B, L, w, m, p, r4);

        dim3 gridDim3(B*(L-w),  (w-1)*(L-w), d);
        dim3 blockDim3(r3);
        kernel_forward_d1<<<gridDim3, blockDim3>>>(head_d1, alpha_ld, alpha_rd, alpha_d, B, L, w, d, r3);
        if(w>2){
            dim3 gridDim4(B*(L-w),  (w-1)*(L-w), d);
            dim3 blockDim4(r4);
            kernel_forward_d2<<<gridDim4, blockDim4>>>(head_d2, alpha_dc, alpha_dd, alpha_d, B, L, w, d, r4);
        }

        dim3 gridDim5(B*(L-w), (w-1)*(L-w), r4);
        dim3 blockDim5(d);
        kernel_forward_dr<<<gridDim5, blockDim5>>>(dd, alpha_d, alpha_dd, B, L, w, d, r4);

        dim3 gridDim55(B*(L-w),  (w-1)*(L-w), r2);
        dim3 blockDim55(d);
        kernel_forward_dr<<<gridDim55, blockDim55>>>(cd, alpha_d, alpha_cd, B, L, w, d, r2);

      }
    }
}


void cuda_backward( float *head_c1,   float *head_c2,   float *head_d1,   float *head_d2,
                   float *left_c,   float *right_c,
                   float *left_d,   float *right_d,
                   float *cc,   float *cd,   float *dc,   float *dd,   float *unary,
                   float *head_c1_grd,   float *head_c2_grd,   float *head_d1_grd,   float *head_d2_grd,
                   float *left_c_grd,   float *right_c_grd,
                   float *left_d_grd,   float *right_d_grd,
                   float *cc_grd,   float *cd_grd,   float *dc_grd,   float *dd_grd,   float *unary_grd,
                   float *alpha,  float *alpha_d, float *alpha_lc, float *alpha_rc, float *alpha_ld, float *alpha_rd,
                   float *alpha_cc, float *alpha_cd, float *alpha_dc, float *alpha_dd,
                   int B, int L, int m, int p, int d, int r1, int r2, int r3, int r4)
{
     for(int w=L-1; w>1; w--){

        if(w<L-1){
        dim3 gridDim55(B*(L-w),  (w-1)*(L-w), r2);
        dim3 blockDim55(d);
        kernel_backward_dr<<<gridDim55, blockDim55>>>(cd, cd_grd, alpha_d, alpha_cd, B, L, w, d, r2);

        dim3 gridDim5(B*(L-w), (w-1)*(L-w), r4);
        dim3 blockDim5(d);
        kernel_backward_dr<<<gridDim5, blockDim5>>>(dd, dd_grd, alpha_d, alpha_dd, B, L, w, d, r4);

        if(w>2){
            dim3 gridDim4(B*(L-w),  (w-1)*(L-w), d);
            dim3 blockDim4(r4);
            kernel_backward_d2<<<gridDim4, blockDim4>>>(head_d2, head_d2_grd, alpha_dc, alpha_dd, alpha_d, B, L, w, d, r4);
        }
        dim3 gridDim3(B*(L-w),  (w-1)*(L-w), d);
        dim3 blockDim3(r3);
        kernel_backward_d1<<<gridDim3, blockDim3>>>(head_d1, head_d1_grd, alpha_ld, alpha_rd, alpha_d, B, L, w, d, r3);

        dim3 gridDim9(B, L-w, r4);
        dim3 blockDim9(m);
        kernel_backward_dmr<<<gridDim9, blockDim9>>>(dc, dc_grd, alpha_dc, alpha, B, L, w, m, p, r4);

        dim3 gridDim8(B, L-w, r2);
        dim3 blockDim8(m);
        kernel_backward_cmr<<<gridDim8, blockDim8>>>(cc, cc_grd, alpha, alpha_cc, B, L, w, m, p, r2);

        dim3 gridDim7(B, L-w, r3);
        dim3 blockDim7(m);
        kernel_backward_cmr<<<gridDim7, blockDim7>>>(left_d, left_d_grd, alpha, alpha_ld,  B, L, w, m, p, r3);
        kernel_backward_cmr<<<gridDim7, blockDim7>>>(right_d, right_d_grd, alpha, alpha_rd,  B, L, w, m, p, r3);

        dim3 gridDim6(B, L-w, r1);
        dim3 blockDim6(m);
        kernel_backward_cmr<<<gridDim6, blockDim6>>>(left_c, left_c_grd, alpha, alpha_lc, B, L, w, m, p, r1);
        kernel_backward_cmr<<<gridDim6, blockDim6>>>(right_c, right_c_grd, alpha, alpha_rc, B, L, w, m, p, r1);
     }

    if (w>2){
        dim3 gridDim2(B, L-w, m);
        dim3 blockDim2(r2);
        kernel_backward_close<<<gridDim2, blockDim2>>>(head_c2, head_c2_grd, alpha, alpha_cd, alpha_cc, B, L, w, m, r2);
    }

    dim3 gridDim(B, L-w, m);
    dim3 blockDim(r1);
    kernel_backward<<<gridDim, blockDim>>>(head_c1, head_c1_grd, alpha, alpha_lc, alpha_rc, B, L, w, m, r1);
    }


    dim3 gridDim(B, L-1, r1);
    dim3 blockDim(p);
    kernel_backward_cpr<<<gridDim, blockDim>>>(right_c, right_c_grd, alpha_rc, unary, unary_grd, B, L, m, p, r1);
    kernel_backward_cpr<<<gridDim, blockDim>>>(left_c, left_c_grd, alpha_lc, unary, unary_grd, B, L, m, p, r1);
    dim3 gridDim11(B, L-1, r2);
    kernel_backward_cpr<<<gridDim11, blockDim>>>(cc, cc_grd, alpha_cc, unary, unary_grd, B, L, m, p, r2);
    dim3 gridDim12(B, L-1, r3);
    kernel_backward_cpr<<<gridDim12, blockDim>>>(left_d, left_d_grd, alpha_ld, unary, unary_grd, B, L, m, p, r3);
    kernel_backward_cpr<<<gridDim12, blockDim>>>(right_d, right_d_grd, alpha_rd, unary, unary_grd, B, L, m, p, r3);

    dim3 gridDim13(B, L-1, r4);
    kernel_backward_dpr<<<gridDim13, blockDim>>>(dc, dc_grd, alpha_dc, unary, unary_grd, B, L, m, p, r4);

}



template <typename F>
__global__ void kernel_argmax_c(const F *__restrict__ const s_span_c,
                                    F *__restrict__ alpha_c,
                                    F *__restrict__ alpha_d,
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
     const F *__restrict__ const s_d = alpha_d + b_idx * L4;
     const F *__restrict__ const s_c = alpha_c + b_idx * L2;

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


template <typename F>
__global__ void kernel_argmax_d(const F *__restrict__ const s_span_d,
                                    F *__restrict__ alpha_c,
                                    F *__restrict__ alpha_d,
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
    const F *__restrict__ const s_d = alpha_d + b_idx * L4;
    const F *__restrict__ const s_c = alpha_c + b_idx * L2;


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


template <typename F>
__global__ void kernel_argmax_d_on4(const F *__restrict__ const s_span_d,
                                    F *__restrict__ alpha_c,
                                    F *__restrict__ alpha_d,
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
    const F *__restrict__ const s_d = alpha_d + b_idx * L4;
    const F *__restrict__ const s_c = alpha_c + b_idx * L2;


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

template <typename F>
__global__ void kernel_argmax_d_on3(const F *__restrict__ const s_span_d,
                                    F *__restrict__ alpha_c,
                                    F *__restrict__ alpha_d,
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
    const F *__restrict__ const s_d = alpha_d + b_idx * L4;
    const F *__restrict__ const s_c = alpha_c + b_idx * L2;


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


