#include <stdio.h>
#include <cuda_runtime.h>


template <typename scalar_t>
__device__ scalar_t logsumexp(scalar_t a, scalar_t b) {
  const scalar_t m = max(a, b);
  return log(exp(a - m) + exp(b - m)) + m;
}


template <typename F>
__global__ void kernel_forward(const F *__restrict__ const head_rule,
                                F *__restrict__ alpha,
                                F *__restrict__ alpha_l,
                                F *__restrict__ alpha_r,
                                const int B, const int L, const int width, const int m, const int r)
{
    const int b_idx = blockIdx.x;
    const int start = blockIdx.y;
    const int s_A = blockIdx.z;
    const int s_R = threadIdx.x;

    if(start+width >= L){
        return;
    }

    __shared__ float result[1000];

    int end = start + width;
    const F *__restrict__ const s_l = alpha_l + b_idx * L * L * r;
    const F *__restrict__ const s_r = alpha_r + b_idx * L * L * r;

    float tmp_result = logf(0);

    float rule_score = head_rule[b_idx*m*r + s_A*r + s_R];

    for (int split = start+1; split < start+width; split++)
    {
        tmp_result = logsumexp(tmp_result, s_l[start*L*r + split * r + s_R] + s_r[split * L * r+end*r+s_R] + rule_score);
    }

    result[s_R] = tmp_result;
    __syncthreads();

    if(s_B==0){
        float final_result = logf(0);
        for(int i=0;i<r;i++){
            final_result = logsumexp(final_result, result[i]);
        }
    alpha[b_idx*L*L*m+start*L*m+end*m+s_A] = final_result;
  }
}

template <typename F>
__global__ void kernel_backward(const F *__restrict__ const head_rule,
                                const F *__restrict__ const head_rule_grd,
                                F *__restrict__ alpha,
                                F *__restrict__ alpha_l,
                                F *__restrict__ alpha_r,
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
    const F *__restrict__ const s_l = alpha_l + b_idx * L * L * r;
    const F *__restrict__ const s_r = alpha_r + b_idx * L * L * r;

    float tmp_result = logf(0);

    float rule_score = head_rule[b_idx*m*r + s_A*r + s_R];
    float rule_score_grd = 0;
    float parent_inside = alpha[b_idx*L*L*m+start*L*m+end*m+s_A];
    float parent_grd = alpha[b_idx*L*L*m+end*L*m+start*m+s_A];

    for (int split = start+1; split < start+width; split++)
    {
        float tmp = exp(s_l[start*L*r + split * r + s_R] + s_r[split * L * r+end*r+s_R] + rule_score - parent_inside) * parent_grd;
        atomicAdd(s_l + split*L*r + start * r + s_R, tmp);
        atomicAdd(s_r + end * L * r+split*r+s_R, tmp);
        rule_score_grd+=tmp;
    }
    atomicAdd(head_rule_grd + b_idx*m*r + s_A*r + s_R, rule_score_grd);
}

template <typename F>
__global__ void kernel_forward_close(const F *__restrict__ const head_rule,
                                 F *__restrict__ alpha_cd,
                                 F *__restrict__ alpha_cc,
                                 const int B, const int L, const int width, const int m,  const int r)
{
    const int b_idx = blockIdx.x;
    const int start = blockIdx.y;
    const int s_A = blockIdx.z;
    const int s_R = threadIdx.x;

    if(start+width >= L){
        return;
    }

    __shared__ float result[200];
    __shared__ float left[200];
    __shared__ float right[200];

    int end = start + width;
    const F *__restrict__ const s = alpha_cc + b_idx * L * L * r;
    const F *__restrict__ const s_d = alpha_cd + b_idx * L * L * L * L * r;

    const int L3 = L * L * L * r;
    const int L2 = L * L * r;
    const int L1 = L * r;

    float tmp_result = logf(0);
    float rule_score = head_rule[b_idx*m*r + s_A*r + s_R];

    for (int split = start+1; split < start+width-1; split++)
    {
         for(int split2 = split+1; split2 < start+width; split2++)
         {
              tmp_result=logsumexp(tmp_result,s[split*L*r + split2*r + s_R] + s_d[start*L3 + split*L2 + split2*L1 + end*r + s_R] + rule_score);
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
                                 const F *__restrict__ const head_rule_grd,
                                 F *__restrict__ alpha_cd,
                                 F *__restrict__ alpha_cc,
                                 const int B, const int L, const int width, const int m, const int p, const int d)
{
    const int b_idx = blockIdx.x;
    const int start = blockIdx.y;
    const int s_A = blockIdx.z;
    const int s_R = threadIdx.x;

    if(start+width >= L){
        return;
    }

    __shared__ float result[200];
    __shared__ float left[200];
    __shared__ float right[200];

    int end = start + width;
    const F *__restrict__ const s = alpha_cc + b_idx * L * L * r;
    const F *__restrict__ const s_d = alpha_cd + b_idx * L * L * L * L * r;

    const int L3 = L * L * L * r;
    const int L2 = L * L * r;
    const int L1 = L * r;

    float tmp_result = logf(0);
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
              atomicAdd(s +  split2*L*r + split*r + s_R, tmp);
              atomicAdd(s_d + split*L3 + start*L2 + split2*L1 + end*r + s_R, tmp);
         }
    }
    atomicAdd(head_rule_grd + b_idx*m*r + s_A*r + s_R, rule_score_grd);
}


template <typename F>
__global__ void kernel_forward_d1(const F *__restrict__ const head,
                                F *__restrict__ alpha_dl,
                                F *__restrict__ alpha_dr,
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
    const int mp = m+p;
    const int mpd = d* (m+p);
    const int gap_start = start + gap_start_minus_start + 1;
    const int gap_end = gap_start + gap_width;
    if(gap_start - start > width-1)
    {
        return;
    }

    __shared__ float result[500];
    const F *__restrict__ const s_l = alpha_dl + b_idx * L * L * r;
    const F *__restrict__ const s_r = alpha_dr + b_idx * L * L * r;
    const int L3 = L * L * L * r;
    const int L2 = L * L * r;
    const int L1 = L * r;

    result[s_R] = s[start*L*r + gap_start*r + s_R] + s[gap_end*L*r + end*r + s_R] + head_rule[b_idx*d*r + s_A*r + s_R];
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
__global__ void kernel_forward_d2(const F *__restrict__ const head,
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
    const int mp = m+p;
    const int mpd = d* (m+p);
    const int gap_start = start + gap_start_minus_start + 1;
    const int gap_end = gap_start + gap_width;
    if(gap_start - start > width-1)
    {
        return;
    }

    __shared__ float result[200];

    const F *__restrict__ const s = alpha_dc + b_idx * L * L * r;
    const F *__restrict__ const s_d = alpha_dd + b_idx * L * L * L * L * r;
    const int L3 = L * L * L * r;
    const int L2 = L * L * r;
    const int L1 = L * r;
    float tmp_result = logf(0);
    float rule_score = head_rule[b_idx*d*r + s_A*r + s_R];

    for(int split=start+1; split< gap_start; split++)
    {
        tmp_result = logsumexp(tmp_result, s_d[split*L3 + gap_start*L2 + gap_end*L1 + end*r + s_R] + rule_score + s[start*L*r*4 + split*r*4 + s_R*4 + 0]);
        tmp_result = logsumexp(tmp_result, s_d[start*L3 + split*L2 + gap_end*L1 + end*r + s_R] + rule_score + s[split*L*r*4 + gap_start*r*4 + s_R*4 + 1]);
    }

    for(int split=gap_end+1; split <end; split++){
        tmp_result = logsumexp(tmp_result, s_d[start*L3 + gap_start*L2 + split*L1 + end*r + s_R] + rule_score + s[gap_end*L*r*4 + split*r*4 + s_R*4 + 2]);
        tmp_result = logsumexp(tmp_result, s_d[start*L3 + gap_start*L2 + gap_end*L1 + split*r + s_R] + rule_score + s[split*L*r*4 + end*r*4 + s_R*4 + 3]);
    }

    result[s_R] = tmp_result;
    __syncthreads();

    if(s_R==0){
    float final_result = logf(0);
    for(int i=0;i<d;i++){
        final_result = logsumexp(final_result, result[i]);
    }
    alpha_d[b_idx*L*L*L*L*d + start*L3 + gap_start*L2 + gap_end*L1 + end*d + s_A] = logsumexp(alpha_d[b_idx*L*L*L*L*d + start*L3 + gap_start*L2 + gap_end*L1 + end*d + s_A], final_result);
  }

}

template <typename F>
__global__ void kernel_backward_d(const F *__restrict__ const head_rule,
                                const F *__restrict__ const head_rule_grd,
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
    const int mp = m+p;
    const int mpd = d* (m+p);
    const int gap_start = start + gap_start_minus_start + 1;
    const int gap_end = gap_start + gap_width;
    if(gap_start - start > width-1)
    {
        return;
    }

    __shared__ float result[200];

    const F *__restrict__ const s = alpha_dc + b_idx * L * L * r;
    const F *__restrict__ const s_d = alpha_dd + b_idx * L * L * L * L * r;
    const int L3 = L * L * L * r;
    const int L2 = L * L * r;
    const int L1 = L * r;
    float tmp_result = logf(0);
    float rule_score = head_rule[b_idx*d*r + s_A*r + s_R];
    float rule_grd = 0;
    float parent_inside = alpha_d[b_idx*L*L*L*L*d + start*L3 + gap_start*L2 + gap_end*L1 + end*d + s_A];
    float parent_grd = alpha_d[b_idx*L*L*L*L*d + start*L3 + gap_start*L2 + gap_end*L1 + end*d + s_A];

    for(int split=start+1; split< gap_start; split++)
    {
        float tmp = exp(s_d[split*L3 + gap_start*L2 + gap_end*L1 + end*r + s_R] + rule_score + s[start*L*r*4 + split*r*4 + s_R*4 + 0] - parent_inside) * parent_grd;
        rule_grd += tmp;
        atomicAdd(s + split*L*r*4 + start*r*4 + s_R*4 + 0, tmp);
        atomicAdd(s_d +  gap_start*L3 + split*L2 + gap_end*L1 + end*r + s_R, tmp);

        tmp = exp(s_d[start*L3 + split*L2 + gap_end*L1 + end*r + s_R] + rule_score + s[split*L*r*4 + gap_start*r*4 + s_R*4 + 1] - parent_inside) * parent_grd;
        rule_grd += tmp;
        atomicAdd(s + gap_start*L*r*4 + split*r*4 + s_R*4 + 0, tmp);
        atomicAdd(s_d +  split*L3 + start*L2 + gap_end*L1 + end*r + s_R, tmp);
    }

    for(int split=gap_end+1; split <end; split++){
        float tmp = exp(s_d[start*L3 + gap_start*L2 + split*L1 + end*r + s_R] + rule_score + s[gap_end*L*r*4 + split*r*4 + s_R*4 + 2] - parent_inside) * parent_grd;
        rule_grd+=tmp;
        atomicAdd(s_d + gap_start*L3 + start*L2 + split*L1 + end*r + s_R, tmp);
        atomicAdd(s + split*L*r*4 + gap_end*r*4 + s_R*4 + 2, tmp);

        tmp = exp(s_d[start*L3 + gap_start*L2 + gap_end*L1 + split*r + s_R] + rule_score + s[split*L*r*4 + end*r*4 + s_R*4 + 3]) - parent_inside) * parent_grd;
        rule_grd+=tmp;
        atomicAdd(s_d + gap_start*L3 + start*L2 + gap_end*L1 + split*r + s_R, tmp);
        atomicAdd(s + gap_end*L*r*4 + split*r*4 + s_R*4 + 3, tmp);
    }
    atomicAdd(head_rule_grd + b_idx*d*r + s_A*r + s_R, rule_grd);


}

template <typename F>
__global__ void kernel_forward_dr(const F *__restrict__ const dd,
                                F *__restrict__ alpha_d,
                                F *__restrict__ alpha_dd,
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
    const int s_R = (blockIdx.z);
    const int s_B = threadIdx.x;
    if(gap_start - start > width-1)
    {
        return;
    }

    __shared__ float result[200];
    result[s_B] = alpha_d[b_idx*L*L*L*L*d + start*L3 + gap_start*L2 + gap_end*L1 + end*d + s_B] + dd[b_idx*d*r + s_B*r + s_A];
    __sycthreads();
    if(s_R==0){
    float final_result = logf(0);
        for(int i=0;i<d;i++){
            final_result = logsumexp(final_result, result[i]);
    }
    alpha_dd[b_idx*L*L*L*L*r + start*L*L*L*r + gap_start*L*L*r + gap_end*L*r + end*r + s_A] = final_result;
    }
}



template <typename F>
__global__ void kernel_forward_cmr(const F *__restrict__ const cc,
                                    F *__restrict__ alpha,
                                    F *__restrict__ alpha_cc,
                                    const int B, const int L, const int width, const int m, const int p, const int r)
{
    const int b_idx = blockIdx.x;
    const int start = blockIdx.y;
    const int s_R = blockIdx.z;
    const int s_B = threadIdx.x;

    const int end = start + width;
    if(end>=L){
        return;
    }
    __shared__ float result[500];
    float inside = alpha[b_idx*L*L*m + start*L*m + end*m + s_B];
    result[s_B] = inside + cc[b_idx*(m+p)*r + s_B*r + s_R];
    __syncthreads();
    if(s_R==0){
    float final_result = logf(0);
        for(int i=0;i<m;i++){
            final_result = logsumexp(final_result, result[i]);
    }
    alpha_cc[b_idx*L*L*r + start*L*r + end*r + s_A] = final_result;
   }
}

template <typename F>
__global__ void kernel_forward_cpr(const F *__restrict__ const cc,
                                    F *__restrict__ alpha_cc,
                                    F *__restrict__ unary,
                                    const int B, const int L, const int width, const int p, const int r)
{
    const int b_idx = blockIdx.x;
    const int start = blockIdx.y;
    const int s_R = blockIdx.z;
    const int s_B = threadIdx.x;

    const int end = start + width;
    if(end>=L){
        return;
    }
    __shared__ float result[500];
    result[s_B] = unary[b_idx*p*(L-1) + start*p + s_B] + cc[b_idx*(m+p)*r + (s_B+m)*r + s_R];
    __syncthreads();
    if(s_R==0){
    float final_result = logf(0);
        for(int i=0;i<m;i++){
            final_result = logsumexp(final_result, result[i]);
    }
    alpha_cc[b_idx*L*L*r + start*L*r + end*r + s_A] = final_result;
   }
}

template <typename F>
__global__ void kernel_forward_dpr(const F *__restrict__ const cc,
                                    F *__restrict__ alpha_cc,
                                    F *__restrict__ unary,
                                    const int B, const int L, const int width, const int p, const int r)
{
    const int b_idx = blockIdx.x;
    const int start = blockIdx.y;
    const int s_R = blockIdx.z;
    const int s_B = threadIdx.x;

    const int end = start + width;
    if(end>=L){
        return;
    }
    __shared__ float result1[500];
    __shared__ float result2[500];
    __shared__ float result3[500];
    __shared__ float result4[500];
    float u = unary[b_idx*p*(L-1) + start*p + s_B];
    result1[s_B] = u + cc[b_idx*(m+p)*r*4 + (s_B+m)*r*4 + s_R*4 + 0];
    result2[s_B] = u + cc[b_idx*(m+p)*r*4 + (s_B+m)*r*4 + s_R*4 + 1];
    result3[s_B] = u + cc[b_idx*(m+p)*r*4 + (s_B+m)*r*4 + s_R*4 + 2];
    result3[s_B] = u + cc[b_idx*(m+p)*r*4 + (s_B+m)*r*4 + s_R*4 + 3];
    __syncthreads();
    if(s_R==0){
    float final_result1 = logf(0);
    float final_result2 = logf(0);
    float final_result3 = logf(0);
    float final_result4 = logf(0);
    for(int i=0;i<r;i++){
        final_result1 = logsumexp(final_result1, result1[i]);
        final_result2 = logsumexp(final_result2, result2[i]);
        final_result3 = logsumexp(final_result3, result3[i]);
        final_result4 = logsumexp(final_result4, result4[i]);
    }
    alpha_cc[b_idx*L*L*r*4 + start*L*r*4 + end*r*4 + s_A*4 + 0] = final_result1;
    alpha_cc[b_idx*L*L*r*4 + start*L*r*4 + end*r*4 + s_A*4 + 1] = final_result2;
    alpha_cc[b_idx*L*L*r*4 + start*L*r*4 + end*r*4 + s_A*4 + 2] = final_result3;
    alpha_cc[b_idx*L*L*r*4 + start*L*r*4 + end*r*4 + s_A*4 + 3] = final_result4;
   }
}

template <typename F>
__global__ void kernel_forward_dmr(const F *__restrict__ const cc,
                                    F *__restrict__ alpha_cc,
                                    F *__restrict__ alpha,
                                    const int B, const int L, const int width, const int p, const int r)
{
    const int b_idx = blockIdx.x;
    const int start = blockIdx.y;
    const int s_R = blockIdx.z;
    const int s_B = threadIdx.x;
    const int end = start + width;
    if(end>=L){
        return;
    }
    __shared__ float result1[500];
    __shared__ float result2[500];
    __shared__ float result3[500];
    __shared__ float result4[500];
    float u = alpha[b_idx*L*L*m + start*L*m + end*m + s_B];
    result1[s_B] = u + cc[b_idx*(m+p)*r*4 + (s_B+m)*r*4 + s_R*4 + 0];
    result2[s_B] = u + cc[b_idx*(m+p)*r*4 + (s_B+m)*r*4 + s_R*4 + 1];
    result3[s_B] = u + cc[b_idx*(m+p)*r*4 + (s_B+m)*r*4 + s_R*4 + 2];
    result3[s_B] = u + cc[b_idx*(m+p)*r*4 + (s_B+m)*r*4 + s_R*4 + 3];
    __syncthreads();
    if(s_R==0){
    float final_result1 = logf(0);
    float final_result2 = logf(0);
    float final_result3 = logf(0);
    float final_result4 = logf(0);
    for(int i=0;i<r;i++){
        final_result1 = logsumexp(final_result1, result1[i]);
        final_result2 = logsumexp(final_result2, result2[i]);
        final_result3 = logsumexp(final_result3, result3[i]);
        final_result4 = logsumexp(final_result4, result4[i]);
    }
    alpha_cc[b_idx*L*L*r*4 + start*L*r*4 + end*r*4 + s_A*4 + 0] = final_result1;
    alpha_cc[b_idx*L*L*r*4 + start*L*r*4 + end*r*4 + s_A*4 + 1] = final_result2;
    alpha_cc[b_idx*L*L*r*4 + start*L*r*4 + end*r*4 + s_A*4 + 2] = final_result3;
    alpha_cc[b_idx*L*L*r*4 + start*L*r*4 + end*r*4 + s_A*4 + 3] = final_result4;
   }
}

void cuda_forward(const float *head_c1, const float *head_c2, const float *head_d1, const float *head_d2,
                  const float *left_c, const float *right_c,
                  const float *left_d, const float *right_d,
                  const float *cc, const float *cd, const float *dc, const float *dd, const float *unary,
                  float *alpha,  float *alpha_d, float *alpha_lc, float *alpha_rc, float *alpha_ld, float *alpha_rd,
                  float *alpha_cc, float *alpha_cd, float *alpha_dc, float *alpha_dd,
                  int B, int L, int m, int p, int d, int r1, int r2, int r3, int r4)
{
    dim3 gridDim(B, L-2, r);
    dim3 blockDim(p);
    kernel_forward_cpr<<<gridDim, blockDim>>>(left_c, alpha_lc, unary, B, L, 2, p, r1);
    kernel_forward_cpr<<<gridDim, blockDim>>>(left_d, alpha_ld, unary, B, L, 2, p, r3);
    kernel_forward_cpr<<<gridDim, blockDim>>>(right_c, alpha_rc, unary, B, L, 2, p, r1);
    kernel_forward_cpr<<<gridDim, blockDim>>>(right_d, alpha_rd, unary, B, L, 2, p, r3);
    kernel_forward_cpr<<<gridDim, blockDim>>>(cc, alpha_cc, unary, B, L, 2, p, r2);
    kernel_forward_dpr<<<gridDim, blockDim>>>(dc, alpha_cc, unary, B, L, 2, p, r4);

    dim3 gridDim2(B*(L-2), (L-2)*(L-2), d);
    dim3 blockDim2(r3);
    kernel_forward_d1<<<gridDim2, blockDim2>>>(head_d1, alpha_ld, alpha_rd, B, L, 2, d, r3);

    for(int w=3; w<L; w++){
      dim3 gridDim(B, L-w, m);
      dim3 blockDim(r1);
      kernel_forward<<<gridDim, blockDim>>>(head_c1, alpha, alpha_lc, alpha_rc, B, L, w, m, r1);
      dim3 gridDim2(B, L-w, m);
      dim3 blockDim2(r2);
      kernel_forward_close<<<gridDim2, blockDim2>>>(head_c2,  alpha, alpha_cd, alpha_cc, B, L, w, m, r2);
      if(w<L-1){
        dim3 gridDim3(B*(L-w),  (w-1)*(L-w), d);
        dim3 blockDim3(r3);
        kernel_forward_d1<<<gridDim2, blockDim2>>>(head_d1, alpha_ld, alpha_rd, alpha_d, B, L, width, d, r3);
        dim3 blockDim4(r4);
        kernel_forward_d2<<<gridDim3, blockDim4>>>(head_d2, alpha_dc, alpha_dd, alpha_d, B, L, w, m, p, d);

        dim3 gridDim6(B, L-w, r);
        dim3 blockDim6(m);
        kernel_forward_cmr<<<gridDim6, blockDim6>>>(left_c, alpha, alpha_lc, B, L, width, m, p, r1);
        kernel_forward_cmr<<<gridDim6, blockDim6>>>(right_c, alpha, alpha_rc, B, L, width, m, p, r1);
        kernel_forward_cmr<<<gridDim6, blockDim6>>>(left_d, alpha_d, alpha_ld,  B, L, width, m, p, r3);
        kernel_forward_cmr<<<gridDim6, blockDim6>>>(right_d, alpha_d, alpha_rd,  B, L, width, m, p, r3);
        kernel_forward_cmr<<<gridDim6, blockDim6>>>(cc, alpha_, unary, B, L, width, p, r2);
        kernel_forward_dmr<<<gridDim, blockDim>>>(dc, alpha_dc, unary, B, L, width, p, r4);
        dim3 gridDim5(B*(L-w),  (w-1)*(L-w), r);
        dim3 blockDim5(d);
        kernel_forward_dr<<<gridDim5, blockDim5>>>(dd, alpha_d, alpha_dd, B, L, width, d, r4);
      }
    }

}




void cuda_backward(const float *binary, const float *binary_close,  const float *binary_dc,  const float *binary_d, const float *binary_ill,
                const float *unary,  float *binary_grd, float *binary_close_grd, float *binary_dc_grd,
                 float *binary_d_grd, float *binary_ill_grd, float *unary_grd, float *alpha,  float *alpha_d, int B, int L, int m, int p, int d)
{

    for(int w=L-1; w>2; w--){
        if(w<L-1){
            dim3 gridDim3(B*(L-w),  (w-1)*(L-w), d);
            dim3 blockDim3(d);
            kernel_backward_d<<<gridDim3, blockDim3>>>(binary_d, binary_dc, unary, binary_d_grd, binary_dc_grd, unary_grd, alpha, alpha_d, B, L, w, m, p, d);
        }
        dim3 gridDim(B, L-w, m);
        dim3 blockDim(m);
        kernel_backward<<<gridDim, blockDim>>>(binary, unary,  binary_grd, unary_grd, alpha, B, L, w, m, p);
        dim3 gridDim2(B, L-w, m);
        dim3 blockDim2(d);
        kernel_backward_close<<<gridDim2, blockDim2>>>(binary_close, binary_ill, unary, binary_close_grd, binary_ill_grd, unary_grd, alpha, alpha_d, B, L, w, m, p, d);

    }
    dim3 gridDim2(B*(L-2), (L-2)*(L-2), d);
    dim3 blockDim2(d);
    kernel_backward_d<<<gridDim2, blockDim2>>>(binary_d, binary_dc, unary, binary_d_grd, binary_dc_grd, unary_grd, alpha, alpha_d, B, L, 2, m, p, d);
    dim3 gridDim(B, L-2+1, m);
    dim3 blockDim(p);
    kernel_backward_len2<<<gridDim, blockDim>>>(binary, unary, binary_grd, unary_grd, alpha, B, L, m, p);
}



