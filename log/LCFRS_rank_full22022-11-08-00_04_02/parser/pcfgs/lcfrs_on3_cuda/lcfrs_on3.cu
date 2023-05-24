#include <stdio.h>
#include <cuda_runtime.h>


template <typename scalar_t>
__device__ scalar_t logsumexp(scalar_t a, scalar_t b) {
  const scalar_t m = max(a, b);
  return log(exp(a - m) + exp(b - m)) + m;
}

template <typename F>
__global__ void kernel_forward_len2(const F *__restrict__ const binary_rule,
                                    const F *__restrict__ const unary,
                                    F *__restrict__ alpha,
                                    const int B, const int L, const int m, const int p)
{
   const int b_idx = blockIdx.x;
   const int start = blockIdx.y;
   const int s_A = blockIdx.z;
   const int s_B = threadIdx.x;
   if(start+2>=L){
    return;
   }
   int mp = m+p;
   const F *__restrict__ const binary = binary_rule + b_idx * m * mp * mp;
   __shared__ float result[200];
   __shared__ float left[200];
   __shared__ float right[200];
   left[s_B] =  unary[b_idx*(L-1)*p + start*p + s_B];
   right[s_B] = unary[b_idx*(L-1)*p + (start+1)*p + s_B];

    __syncthreads();
   float tmp_result = logf(0);
   for(int sc=0; sc<p; sc++){
       tmp_result = logsumexp(tmp_result,  left[s_B] + right[sc] + binary[s_A*mp*mp + (s_B+m)* mp+ (m+sc)]);
   }
   result[s_B]=tmp_result;
    __syncthreads();
   if(s_B==0){
   float final_result = logf(0);
   for(int i=0;i<p;i++){
        final_result = logsumexp(final_result, result[i]);
   }
   alpha[b_idx*L*L*m+start*L*m+(start+2)*m+s_A] = final_result;
   }
}


template <typename F>
__global__ void kernel_forward_len2_argmax(const F *__restrict__ const binary_rule,
                                 const F *__restrict__ const unary,
                                F *__restrict__ alpha,
                                const int B, const int L, const int m, const int p)
{
   const int b_idx = blockIdx.x;
   const int start = blockIdx.y;
   const int s_A = blockIdx.z;
   const int s_B = threadIdx.x;
   if(start+2>=L){
    return;
   }
   int mp = m+p;
   const F *__restrict__ const binary = binary_rule + b_idx * m * mp * mp;
   __shared__ float result[200];
   __shared__ float result_idx[200];
   __shared__ float left[200];
   __shared__ float right[200];
   left[s_B] =  unary[b_idx*(L-1)*p + start*p + s_B];
   right[s_B] = unary[b_idx*(L-1)*p + (start+1)*p + s_B];

    __syncthreads();
   float tmp_result = logf(0);
   float tmp_idx = -1;
   float tmp = 0;
   for(int sc=0; sc<p; sc++){
       tmp = left[s_B] + right[sc] + binary[s_A*mp*mp + (s_B+m)* mp+ (m+sc)];
       if (tmp > tmp_result){
            tmp_result = tmp;
            tmp_idx = (start+1)*mp*mp + (s_B+m)*mp + (sc+m);
       }
   }
   result[s_B]=tmp_result;
   result_idx[s_B]=tmp_idx;
    __syncthreads();
   if(s_B==0){
   float final_result = logf(0);
   float final_idx = 0;
   for(int i=0;i<p;i++){
        if(result[i] > final_result){
        final_result=result[i];
        final_idx=result_idx[i];
        }
   }
   alpha[b_idx*L*L*m+start*L*m+(start+2)*m+s_A] = final_result;
   alpha[b_idx*L*L*m+(start+2)*L*m+start*m+s_A] = final_idx;
   }
}



template <typename F>
__global__ void kernel_forward(const F *__restrict__ const binary_rule,
                                const F *__restrict__ const unary,
                                F *__restrict__ alpha,
                                const int B, const int L, const int width, const int m, const int p)
{
    const int b_idx = blockIdx.x;
    const int start = blockIdx.y;
    const int s_A = blockIdx.z;
    const int s_B = threadIdx.x;
    const int mp = m+p;

    if(start+width >= L){
        return;
    }

    __shared__ float result[200];
    __shared__ float left[200];
    __shared__ float right[200];

    int end = start + width;
    const F *__restrict__ const s = alpha + b_idx * L * L * m;
    const F *__restrict__ const binary = binary_rule + b_idx * m * mp * mp;

    float tmp_result = logf(0);

    right[s_B] = s[(start+1) * L * m + end * m + s_B];
    __syncthreads();

    for (int sc=0; sc<p; sc++)
    {
        tmp_result = logsumexp(tmp_result,  unary[b_idx*(L-1)*p + (start) *p + sc] + right[s_B] + binary[s_A*(mp)*(mp) + (m+sc)*mp + s_B]);
    }
    __syncthreads();

    for (int split = start+2; split < start+width-1; split++)
    {

         left[s_B] =  s[start * L * m + split * m + s_B];
         right[s_B] = s[split * L * m + end * m + s_B];
         __syncthreads();
         for(int sc=0; sc<m; sc++){
                 tmp_result = logsumexp(tmp_result, left[s_B] + right[sc] + binary[s_A*(mp)*(mp) + s_B*mp + sc]);
         }
         __syncthreads();
    }

    left[s_B] = s[start * L * m + (start+width-1) * m + s_B];
    __syncthreads();
    for (int sc=0; sc<p; sc++)
    {
        tmp_result = logsumexp(tmp_result, left[s_B] + unary[b_idx*(L-1)*p + (start+width-1) *p + sc] + binary[s_A*(mp)*(mp) + s_B*mp +m+sc]);
    }

    result[s_B] = tmp_result;
    __syncthreads();

    if(s_B==0){
    float final_result = logf(0);
    for(int i=0;i<m;i++){
        final_result = logsumexp(final_result, result[i]);
    }

    alpha[b_idx*L*L*m+start*L*m+end*m+s_A] = final_result;
  }
}


template <typename F>
__global__ void kernel_forward_argmax(const F *__restrict__ const binary_rule,
                                const F *__restrict__ const unary,
                                F *__restrict__ alpha,
                                const int B, const int L, const int width, const int m, const int p)
{
    const int b_idx = blockIdx.x;
    const int start = blockIdx.y;
    const int s_A = blockIdx.z;
    const int s_B = threadIdx.x;
    const int mp = m+p;

    if(start+width >= L){
        return;
    }

    __shared__ float result[200];
    __shared__ int result_idx[200];
    __shared__ float left[200];
    __shared__ float right[200];

    int end = start + width;
    const F *__restrict__ const s = alpha + b_idx * L * L * m;
    const F *__restrict__ const binary = binary_rule + b_idx * m * mp * mp;

    float tmp_result = logf(0);
    float tmp = 0;
    int tmp_idx = -1;

    right[s_B] = s[(start+1) * L * m + end * m + s_B];
    __syncthreads();

    for (int sc=0; sc<p; sc++)
    {
        tmp = unary[b_idx*(L-1)*p + (start) *p + sc] + right[s_B] + binary[s_A*(mp)*(mp) + (m+sc)*mp + s_B];
        if(tmp > tmp_result){
            tmp_result = tmp;
            tmp_idx = (start+1) * mp * mp + (sc+m)*mp + s_B;
        }
    }
    __syncthreads();

    for (int split = start+2; split < start+width-1; split++)
    {
         left[s_B] =  s[start * L * m + split * m + s_B];
         right[s_B] = s[split * L * m + end * m + s_B];
         __syncthreads();
         for(int sc=0; sc<m; sc++){
                 tmp = left[s_B] + right[sc] + binary[s_A*(mp)*(mp) + s_B*mp + sc];
                 if(tmp > tmp_result){
                    tmp_result = tmp;
                    tmp_idx = (split)*mp*mp + s_B*mp + sc;
                 }
         }
         __syncthreads();
    }

    left[s_B] = s[start * L * m + (start+width-1) * m + s_B];
    __syncthreads();
    for (int sc=0; sc<p; sc++)
    {
        tmp = left[s_B] + unary[b_idx*(L-1)*p + (start+width-1) *p + sc] + binary[s_A*(mp)*(mp) + s_B*mp +m+sc];
        if (tmp > tmp_result){
            tmp_result = tmp;
            tmp_idx = (end-1)*mp*mp + s_B * mp + (m+sc);
        }
    }

    result[s_B] = tmp_result;
    result_idx[s_B] = tmp_idx;
    __syncthreads();

    if(s_B==0){
    float final_result = logf(0);
    float final_idx = -1;
    for(int i=0;i<m;i++){
        if(result[i] > final_result){
            final_result = result[i];
            final_idx = result_idx[i];
        }
    }

    alpha[b_idx*L*L*m+start*L*m+end*m+s_A] = final_result;
    alpha[b_idx*L*L*m+end*L*m +start*m + s_A] = final_idx;
  }
}




template <typename F>
__global__ void kernel_forward_close(const F *__restrict__ const binary_rule,
                                 const F *__restrict__ const unary_rule,
                                F *__restrict__ alpha,
                                F *__restrict__ alpha_d,
                               const int B, const int L, const int width, const int m, const int p, const int d)
{

    const int b_idx = blockIdx.x;
    const int start = blockIdx.y;
    const int s_A = blockIdx.z;
    const int s_B = threadIdx.x;
    const int mp = m+p;

    if(start+width >= L){
        return;
    }

    __shared__ float result[200];
    __shared__ float left[200];
    __shared__ float right[200];

    int end = start + width;
    const F *__restrict__ const s = alpha + b_idx * L * L * m;
    const F *__restrict__ const s_d = alpha_d + b_idx * L * L * L * L * d;
    const F *__restrict__ const binary = binary_rule + b_idx * m * d * mp + s_A * d *mp;
    const F *__restrict__ const unary = unary_rule + b_idx * (L-1) * p;

    const int L3 = L * L * L * d;
    const int L2 = L * L * d;
    const int L1 = L * d;

    float tmp_result = logf(0);

    for (int split = start+1; split < start+width-1; split++)
    {
         for(int split2 = split+1; split2 < start+width; split2++)
         {

              left[s_B] = s_d[start * L3 + split * L2 + (split2) * L1 + end*d + s_B];
              __syncthreads();
              if(split2 == split+1){
                    for(int ss=s_B; ss < p; ss+=d){
                        right[ss] = unary[split * p + ss];
                    }
                    __syncthreads();
                    for(int sc=0; sc<p;sc+=1)
                    {
                        tmp_result = logsumexp(tmp_result, left[s_B] + right[sc] + binary[s_B*mp + (m+sc)]);
                    }

              }
              else{
                    for(int ss=s_B; ss<m; ss+=d){
                        right[ss] = s[split*L*m + split2*m + ss];
                    }
                    __syncthreads();
                    for(int sc=0; sc<m;sc+=1)
                    {
                        tmp_result = logsumexp(tmp_result, left[s_B] + right[sc] + binary[s_B*mp + (sc)]);
                    }
              }
                            __syncthreads();
         }
    }

    result[s_B] = tmp_result;
    __syncthreads();

    if(s_B==0){
        float final_result = logf(0);
        for(int i=0;i<d;i++){
            final_result = logsumexp(final_result, result[i]);
        }
        alpha[b_idx*L*L*m+start*L*m+(end)*m+s_A] = logsumexp(alpha[b_idx*L*L*m+start*L*m+(end)*m+s_A], final_result);
    }
}




template <typename F>
__global__ void kernel_forward_close_argmax(const F *__restrict__ const binary_rule,
                                 const F *__restrict__ const unary_rule,
                                F *__restrict__ alpha,
                                F *__restrict__ alpha_d,
                               const int B, const int L, const int width, const int m, const int p, const int d)
{

    const int b_idx = blockIdx.x;
    const int start = blockIdx.y;
    const int s_A = blockIdx.z;
    const int s_B = threadIdx.x;
    const int mp = m+p;

    if(start+width >= L){
        return;
    }

    __shared__ float result[200];
    __shared__ float result_idx[200];
    __shared__ float left[200];
    __shared__ float right[200];

    int end = start + width;
    const F *__restrict__ const s = alpha + b_idx * L * L * m;
    const F *__restrict__ const s_d = alpha_d + b_idx * L * L * L * L * d;
    const F *__restrict__ const binary = binary_rule + b_idx * m * d * mp + s_A * d *mp;
    const F *__restrict__ const unary = unary_rule + b_idx * (L-1) * p;

    const int L3 = L * L * L * d;
    const int L2 = L * L * d;
    const int L1 = L * d;

    float tmp_result = logf(0);
    float tmp = 0;
    float tmp_idx = -1;

    for (int split = start+1; split < start+width-1; split++)
    {
         for(int split2 = split+1; split2 < start+width; split2++)
         {

              left[s_B] = s_d[start * L3 + split * L2 + (split2) * L1 + end*d + s_B];
              __syncthreads();
              if(split2 == split+1){
                    for(int ss=s_B; ss < p; ss+=d){
                        right[ss] = unary[split * p + ss];
                    }
                    __syncthreads();
                    for(int sc=0; sc<p;sc+=1)
                    {
                        tmp = left[s_B] + right[sc] + binary[s_B*mp + (m+sc)];
                        if(tmp > tmp_result){
                            tmp_result=tmp;
                            tmp_idx = -(split*L1*mp + split2*d*mp + s_B*mp + (sc+m));
                        }
                    }
              }
              else{
                    for(int ss=s_B; ss<m; ss+=d){
                        right[ss] = s[split*L*m + split2*m + ss];
                    }
                    __syncthreads();
                    for(int sc=0; sc<m;sc+=1)
                    {
                        tmp =  left[s_B] + right[sc] + binary[s_B*mp + (sc)];
                        if(tmp > tmp_result){
                            tmp_result=tmp;
                            tmp_idx = -(split*L1*mp +split2*d*mp + s_B*mp + (sc));
                        }

                    }
              }

            __syncthreads();
         }
    }

    result[s_B] = tmp_result;
    result_idx[s_B] = tmp_idx;
    __syncthreads();

    if(s_B==0){
        float final_result = logf(0);
        float final_idx = -1;
        for(int i=0;i<d;i++){
            if(result[i] > final_result){
                final_result = result[i];
                final_idx = result_idx[i];
            }
        }

        float prev_best = alpha[b_idx*L*L*m+start*L*m+(end)*m+s_A];
        if(final_result > prev_best){
            alpha[b_idx*L*L*m+start*L*m+(end)*m+s_A] = final_result;
            alpha[b_idx*L*L*m+end*L*m+(start)*m+s_A] = final_idx;
        }
    }
}


template <typename F>
__global__ void kernel_backward_close(const F *__restrict__ const binary_rule,
                                 const F *__restrict__ const unary_rule,
                                 F *__restrict__ binary_rule_grd,
                                 F *__restrict__ unary_rule_grd,
                                 F *__restrict__ alpha,
                                 F *__restrict__ alpha_d,
                                const int B, const int L, const int width, const int m, const int p, const int d)
{
    const int b_idx = blockIdx.x;
    const int start = blockIdx.y;
    const int s_A = blockIdx.z;
    const int s_B = threadIdx.x;
    const int mp = m+p;

    if(start+width >= L){
        return;
    }

    __shared__ float result[200];
    __shared__ float left[200];
    __shared__ float right[200];

    int end = start + width;
    F *__restrict__ s = alpha + b_idx * L * L * m;
    F *__restrict__ s_d = alpha_d + b_idx * L * L * L * L * d;
    const F *__restrict__ const binary = binary_rule + b_idx * m * d * mp + s_A * d *mp;
    const F *__restrict__ const unary = unary_rule + b_idx * (L-1) * p;
    F *__restrict__ binary_grd = binary_rule_grd + b_idx * m * d * mp + s_A * d *mp;
    F *__restrict__ unary_grd = unary_rule_grd + b_idx * (L-1) * p;


    const int L3 = L * L * L * d;
    const int L2 = L * L * d;
    const int L1 = L * d;

    float tmp_result = logf(0);
    float tmp = 0;
    float parent_inside = s[start*L*m + end*m + s_A];
    float parent_grd = s[end*L*m + start*m + s_A];

    for (int split = start+1; split < start+width-1; split++)
    {
         for(int split2 = split+1; split2 < start+width; split2++)
         {

              left[s_B] = s_d[start * L3 + split * L2 + (split2) * L1 + end*d + s_B];
              __syncthreads();
              if(split2 == split+1){
                    for(int ss=s_B; ss < p; ss+=d){
                        right[ss] = unary[split * p + ss];
                    }
                    __syncthreads();
                    for(int sc=0; sc<p;sc+=1)
                    {
                        tmp = exp(left[s_B] + right[sc] + binary[s_B*mp + (m+sc)] - parent_inside);
                        if(tmp > 1){
                        printf("?\n");
                        }
                        tmp*=parent_grd;
                        atomicAdd(binary_grd + s_B*mp + (m+sc), tmp);
                        atomicAdd( s_d + split * L3 + start * L2 + (split2) * L1 + end*d + s_B, tmp);
                        atomicAdd(unary_grd + split * p + sc, tmp);
                    }

              }
              else{
                    for(int ss=s_B; ss<m; ss+=d){
                        right[ss] = s[split*L*m + split2*m + ss];
                    }
                    __syncthreads();
                    for(int sc=0; sc<m;sc+=1)
                    {
                        tmp = exp(left[s_B] + right[sc] + binary[s_B*mp + (sc)] - parent_inside) * parent_grd;
                        atomicAdd(binary_grd +  s_B*mp + sc, tmp);
                        atomicAdd(s_d + split * L3 + start * L2 + (split2) * L1 + end*d + s_B, tmp);
                        atomicAdd(s + split2*L*m + split*m + sc, tmp);
                   }
              }
             __syncthreads();
         }
    }
}



template <typename F>
__global__ void kernel_forward_d(const F *__restrict__ const binary_rule_dc,
                                 const F *__restrict__ const unary,
                                F *__restrict__ alpha,
                               F *__restrict__ alpha_d,
                               const int B, const int L, const int width, const int m, const int p, const int d)
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
    const int s_B = threadIdx.x;
    const int mp = m+p;
    const int gap_start = start + gap_start_minus_start + 1;
    const int gap_end = gap_start + gap_width;
    if(gap_start - start > width-1)
    {
        return;
    }

    __shared__ float result[200];
    __shared__ float left[200];
    __shared__ float right[200];

    const F *__restrict__ const s = alpha + b_idx * L * L * m;
    const F *__restrict__ const binary_c = binary_rule_dc + b_idx * d * mp * mp;
    __syncthreads();
    float tmp_result = logf(0);

    if(gap_start==start+1){
       for(int ss=s_B; ss<p; ss+=d){
          left[ss] = unary[b_idx*(L-1)*p+ start*p + ss];
       }
       __syncthreads();

       if(gap_end==end-1){
           for(int ss=s_B; ss<p; ss+=d)
           {
                right[ss] = unary[b_idx*(L-1)*p+ (end-1)*p + ss];
           }
           __syncthreads();

           for(int ss=s_B; ss<p; ss+=d)
           {
               for(int sc=0;sc<p;sc++){
                  tmp_result = logsumexp(tmp_result, left[ss] + right[sc] + binary_c[s_A*mp*mp + (ss+m)*mp+ (sc+m)]);
              }
           }
       }

       else{
          for(int ss=s_B; ss<m; ss+=d)
          {
             right[ss] = s[ gap_end * L*m + end*m + ss];
          }
         __syncthreads();
          for(int sc=0;sc<p;sc++)
          {
             for(int ss=s_B;ss<m;ss+=d){
                tmp_result = logsumexp(tmp_result, left[sc]+right[ss]+binary_c[s_A*mp*mp+ (sc+m)*mp+ ss]);
             }
          }
       }
    }
    else{
      for(int ss=s_B; ss<m; ss+=d)
      {
         left[ss] = s[start * L*m + gap_start*m + ss];
      }
      __syncthreads();
       if(gap_end==end-1){
           for(int ss=s_B; ss<p; ss+=d)
           {
                right[ss] = unary[ b_idx*(L-1)*p+ (end-1)*p + ss];
           }
            __syncthreads();
           for(int sc=0;sc<p;sc++){
             for(int ss=s_B;ss<m;ss+=d){
                tmp_result = logsumexp(tmp_result, left[ss]+right[sc]+binary_c[s_A*mp*mp+ (ss)*mp+ (sc+m)]);
             }
           }
       }
       else{
          for(int ss=s_B; ss<m; ss+=d)
          {
             right[ss] = s[ gap_end * L*m + end*m + ss];
          }
           __syncthreads();
          for(int sc=0;sc<m;sc++)
          {
             for(int ss=s_B;ss<m;ss+=d){
                tmp_result = logsumexp(tmp_result, left[sc]+right[ss]+binary_c[s_A*mp*mp+ (sc)*mp+ (ss)]);
             }
          }
       }
    }

    result[s_B] = tmp_result;
    __syncthreads();
    if(s_B==0){
    float final_result = logf(0);
    for(int i=0;i<d;i++){
        final_result = logsumexp(final_result, result[i]);
    }
    alpha_d[b_idx*L*L*L*L*d + start*L*L*L*d + gap_start*L*L*d + gap_end*L*d + end*d + s_A] = final_result;
  }
}


template <typename F>
__global__ void kernel_forward_d_argmax(
                                    const F *__restrict__ const binary_rule_dc,
                                 const F *__restrict__ const unary,
                                F *__restrict__ alpha,
                                F *__restrict__ alpha_d,
                               const int B, const int L, const int width, const int m, const int p, const int d)
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
    const int s_B = threadIdx.x;
    const int mp = m+p;
    const int mpd = d* (m+p);
    const int gap_start = start + gap_start_minus_start + 1;
    const int gap_end = gap_start + gap_width;

    if(gap_start - start >width-1)
    {
        return;
    }

    __shared__ float result[200];
    __shared__ float result_idx[200];
    __shared__ float left[200];
    __shared__ float right[200];

    const F *__restrict__ const s = alpha + b_idx * L * L * m;
    const F *__restrict__ const s_d = alpha_d + b_idx * L * L * L * L * d;
    const F *__restrict__ const binary_c = binary_rule_dc + b_idx * d * mp * mp;
    const int L3 = L * L * L * d;
    const int L2 = L * L * d;
    const int L1 = L * d;

    float tmp_result = logf(0);
    float tmp_idx = -1;
    float tmp = 0;

    if(gap_start==start+1){
       for(int ss=s_B; ss<p; ss+=d){
          left[ss] = unary[b_idx*(L-1)*p+ start*p + ss];
       }
       __syncthreads();
       if(gap_end==end-1){
           for(int ss=s_B; ss<p; ss+=d)
           {
                right[ss] = unary[b_idx*(L-1)*p+ (end-1)*p + ss];
           }
           __syncthreads();

           for(int ss=s_B; ss<p; ss+=d)
           {
               for(int sc=0;sc<p;sc++){
                  tmp = left[ss] + right[sc] + binary_c[s_A*mp*mp + (ss+m)*mp+ (sc+m)];
                  if(tmp > tmp_result){
                    tmp_result = tmp;
                    tmp_idx = -((ss+m)*mp + (sc+m));
                  }
               }
           }
       }

       else{
          for(int ss=s_B; ss<m; ss+=d)
          {
             right[ss] = s[ gap_end * L*m + end*m + ss];
          }
          __syncthreads();
          for(int sc=0;sc<p;sc++)
          {
             for(int ss=s_B;ss<m;ss+=d){
                tmp = left[sc]+right[ss]+binary_c[s_A*mp*mp+ (sc+m)*mp+ ss];
                if(tmp > tmp_result){
                    tmp_result = tmp;
                    tmp_idx = -((sc+m)*mp + ss);
                }
             }
          }
       }
    }
    else{
      for(int ss=s_B; ss<m; ss+=d)
      {
         left[ss] = s[start * L*m + gap_start*m + ss];
      }
    __syncthreads();
       if(gap_end==end-1){
           for(int ss=s_B; ss<p; ss+=d)
           {
                right[ss] = unary[ b_idx*(L-1)*p+ (end-1)*p + ss];
           }
               __syncthreads();
           for(int sc=0;sc<p;sc++){
             for(int ss=s_B;ss<m;ss+=d){
                tmp = left[ss]+right[sc]+binary_c[s_A*mp*mp+ (ss)*mp+ (sc+m)];
                if(tmp > tmp_result){
                    tmp_result = tmp;
                    tmp_idx = -( ss*mp + (sc+m));
                }


             }
           }
       }
       else{
          for(int ss=s_B; ss<m; ss+=d)
          {
             right[ss] = s[ gap_end * L*m + end*m + ss];
          }
          __syncthreads();
          for(int sc=0;sc<m;sc++)
          {
             for(int ss=s_B;ss<m;ss+=d){
                tmp = left[sc]+right[ss]+binary_c[s_A*mp*mp+ (sc)*mp+ (ss)];
                if (tmp > tmp_result){
                    tmp_result = tmp;
                    tmp_idx = -(sc*mp+ss);
                }
             }
          }
       }
    }

    result[s_B] = tmp_result;
    result_idx[s_B] = tmp_idx;
    __syncthreads();
    if(s_B==0){
    float final_result = logf(0);
    float final_idx = -1;
    for(int i=0;i<d;i++){
        if(result[i] > final_result){
            final_result = result[i];
            final_idx = result_idx[i];
        }
    }
    alpha_d[b_idx*L*L*L*L*d + start*L3 + gap_start*L2 + gap_end*L1 + end*d + s_A] = final_result;
    alpha_d[b_idx*L*L*L*L*d + gap_start*L3 + start*L2 + gap_end*L1 + end*d + s_A] = final_idx;
  }
}




template <typename F>
__global__ void kernel_backward_d(const F *__restrict__ const binary_rule_dc,
                                 const F *__restrict__ const unary,
                              F *__restrict__ binary_rule_dc_grd,
                               F *__restrict__  unary_grd,
                                F *__restrict__ alpha,
                                F *__restrict__ alpha_d,
                               const int B, const int L, const int width, const int m, const int p, const int d)
{
    const int b_idx = blockIdx.x / (L-width);
    const int start = blockIdx.x % (L-width);
    const int gap_start_minus_start = (blockIdx.y) / (L-width);
    const int gap_width =  ((blockIdx.y) % ( L-width)) + 1;
    const int end = start + width + gap_width;
    if(end>=L){
        return;
    }
    const int s_A = blockIdx.z;
    const int s_B = threadIdx.x;
    const int mp = m+p;
    const int mpd = d* (m+p);
    const int gap_start = start + gap_start_minus_start + 1;
    const int gap_end = gap_start + gap_width;

    if(gap_start - start >width-1)
    {
        return;
    }

    F *__restrict__ s = alpha + b_idx * L * L * m;
    F *__restrict__ s_d = alpha_d + b_idx * L * L * L * L * d;
    const F *__restrict__ const binary_c = binary_rule_dc + b_idx * d * mp * mp;
    F *__restrict__ binary_c_grd = binary_rule_dc_grd + b_idx * d * mp * mp + s_A * mp * mp;
    const int L3 = L * L * L * d;
    const int L2 = L * L * d;
    const int L1 = L * d;

    float tmp=0;
    float parent_inside = s_d[start*L3 + gap_start*L2 + gap_end*L1 + end*d + s_A];
    float parent_grd = s_d[gap_start*L3 + start*L2 + gap_end*L1+end*d+s_A];

    __shared__ float left[200];
    __shared__ float right[200];
    __syncthreads();

    if(gap_start==start+1){
       for(int ss=s_B; ss<p; ss+=d){
          left[ss] = unary[b_idx*(L-1)*p+ start*p + ss];
       }
       __syncthreads();
       if(gap_end==end-1){
           for(int ss=s_B; ss<p; ss+=d)
           {
                right[ss] = unary[b_idx*(L-1)*p+ (end-1)*p + ss];
           }
           __syncthreads();

           for(int ss=s_B; ss<p; ss+=d)
           {
               for(int sc=0;sc<p;sc++){
                  tmp = exp(left[ss] + right[sc] + binary_c[s_A*mp*mp + (ss+m)*mp+ (sc+m)] - parent_inside);
                  tmp = tmp * parent_grd;
                  atomicAdd(unary_grd + b_idx*(L-1)*p + start*p + ss, tmp);
                  atomicAdd(unary_grd + b_idx*(L-1)*p + gap_end*p + sc, tmp);
                  atomicAdd(binary_c_grd + (ss+m)*mp + (sc+m),  tmp);
               }
           }
       }

       else{
          for(int ss=s_B; ss<m; ss+=d)
          {
             right[ss] = s[gap_end * L*m + end*m + ss];
          }
         __syncthreads();
          for(int sc=0;sc<p;sc++)
          {
             for(int ss=s_B;ss<m;ss+=d){
                float tmp =  exp(left[sc]+right[ss]+binary_c[s_A*mp*mp+ (sc+m)*mp+ ss] - parent_inside);
                tmp = tmp * parent_grd;
                atomicAdd(unary_grd + b_idx*(L-1)*p + start*p + sc, tmp);
                atomicAdd(s + end*L*m + gap_end*m + ss, tmp);
                atomicAdd(binary_c_grd + (sc+m)*mp + ss,  tmp);
             }
          }
       }
    }
    else{
      for(int ss=s_B; ss<m; ss+=d)
      {
         left[ss] = s[start * L*m + gap_start*m + ss];
      }
      __syncthreads();
       if(gap_end==end-1){
           for(int ss=s_B; ss<p; ss+=d)
           {
                right[ss] = unary[ b_idx*(L-1)*p+ (end-1)*p + ss];
           }
            __syncthreads();
           for(int sc=0;sc<p;sc++){
             for(int ss=s_B;ss<m;ss+=d){
                float tmp = exp(left[ss]+right[sc]+binary_c[s_A*mp*mp+ (ss)*mp+ (sc+m)] - parent_inside);
                tmp = tmp * parent_grd;
                atomicAdd(s + gap_start*m*L + start*m + ss, tmp);
                atomicAdd(unary_grd +  b_idx*(L-1)*p+ (end-1)*p + sc, tmp);
                atomicAdd(binary_c_grd + (ss)*mp + (sc+m),  tmp);
             }
           }
       }
       else{
          for(int ss=s_B; ss<m; ss+=d)
          {
             right[ss] = s[ gap_end * L*m + end*m + ss];
          }
          __syncthreads();
          for(int sc=0;sc<m;sc++)
          {
             for(int ss=s_B;ss<m;ss+=d){
                float tmp = exp(left[sc]+right[ss]+binary_c[s_A*mp*mp+ (sc)*mp+ (ss)] - parent_inside);
                tmp = tmp * parent_grd;
                atomicAdd(s + gap_start*m*L + start*m + sc, tmp);
                atomicAdd(s + end * L*m + gap_end*m +ss, tmp);
                atomicAdd(binary_c_grd + (sc)*mp + (ss),  tmp);
             }
          }
       }

    }
}

template <typename F>
__global__ void kernel_backward(const F *__restrict__ const binary_rule,
                                 const F *__restrict__ const unary,
                                 F *__restrict__ binary_grd,
                                 F *__restrict__ unary_grd,
                                 F *__restrict__ alpha,
                                 const int B, const int L, const int width, const int m, const int p)
{
    const int b_idx = blockIdx.x;
    const int start = blockIdx.y;
    const int s_A = blockIdx.z;
    const int s_B = threadIdx.x;
    const int mp = m+p;

    if(start+width >= L){
        return;
    }

    __shared__ float left_inside[200];
    __shared__ float right_inside[200];

    int end = start + width;
    F *__restrict__  s = alpha + b_idx * L * L * m;
    const F *__restrict__ const binary = binary_rule + b_idx * m * mp * mp;
    F *__restrict__ g_binary = binary_grd + b_idx * m * mp * mp;
    float parent_inside = s[start*L*m + end*m + s_A];
    float parent_grd  = s[end*L*m + start*m + s_A];

    right_inside[s_B] = s[(start+1) * L * m + end * m + s_B];
    __syncthreads();


    for (int sc=0; sc<p; sc++)
    {
       float tmp = exp(unary[b_idx*(L-1)*p + (start) *p + sc] + right_inside[s_B] + binary[s_A*(mp)*(mp) + (m+sc)*mp + s_B] - parent_inside) * parent_grd;
       atomicAdd(s + end*L*m + (start+1)*m + s_B, tmp);
       atomicAdd(g_binary + s_A*mp*mp + (sc+m)*mp + s_B, tmp);
       atomicAdd(unary_grd + b_idx*(L-1)*p + start*p + sc, tmp);
    }

    for (int split = start+2; split < start+width-1; split++)
    {
        left_inside[s_B] =  s[start * L * m + split * m + s_B];
            right_inside[s_B] = s[split * L * m + end * m + s_B];
            __syncthreads();
            for(int sc=0; sc<m; sc++){
                float tmp = exp(left_inside[s_B] + right_inside[sc] + binary[s_A*(mp)*(mp) + s_B*mp + sc] - parent_inside)* parent_grd;
                atomicAdd(s + split * L * m + start * m + s_B,  tmp);
                atomicAdd(s +end * L * m + split * m + sc, tmp);
                atomicAdd(&g_binary[s_A*mp*mp + (s_B)*mp + sc], tmp);
            }
            __syncthreads();
    }

    left_inside[s_B] = s[start * L * m + (start+width-1) * m + s_B];
    __syncthreads();
    for (int sc=0; sc<p; sc++){
         float tmp= exp(left_inside[s_B] + unary[b_idx*(L-1)*p + (end-1) *p + sc] + binary[s_A*(mp)*(mp) + s_B*mp +m+sc] - parent_inside) * parent_grd;
         atomicAdd(s + (end-1)*L*m+ (start)*m + s_B, tmp);
         atomicAdd(g_binary + s_A*mp*mp + (s_B)*mp + (sc +m), tmp);
         atomicAdd(unary_grd + b_idx*(L-1)*p + (end-1)*p + sc,  tmp);
    }
}

template <typename F>
__global__ void kernel_backward_len2(const F *__restrict__ const binary_rule,
                                 const F *__restrict__ const unary,
                                 F *__restrict__ binary_grd,
                                 F *__restrict__ unary_grd,
                                F *__restrict__ alpha,
                               const int B, const int L, const int m, const int p)
{
    const int b_idx = blockIdx.x;
    const int start = blockIdx.y;
    const int s_A = blockIdx.z;
    const int s_B = threadIdx.x;
    const int mp = m+p;
    const int width= 2;
    if(start+width >= L){
        return;
    }

    int end = start + width;
    F *__restrict__  s = alpha + b_idx * L * L * m;
    const F *__restrict__ const binary = binary_rule + b_idx * m * mp * mp;
    F *__restrict__  g_binary = binary_grd + b_idx * m * mp * mp;

    float parent_inside = s[(start*L*m) + (end*m) + s_A];
    float parent_grd  = s[(end*L*m) + (start*m) + s_A];
    __shared__ float ss[200];
    __shared__ float sss[200];

    ss[s_B]= unary[b_idx*(L-1)*p + start*p + s_B];
    sss[s_B] = unary[b_idx*(L-1)*p + (start+1)*p + s_B];
    __syncthreads();

    for (int sc=0; sc<p; sc++)
    {
       float tmp = exp(ss[s_B] + sss[sc] + binary[s_A*(mp)*(mp) + (m+s_B)*mp + (sc+m)] - parent_inside) * parent_grd;
       atomicAdd(unary_grd+b_idx*(L-1)*p + start*p + s_B, tmp);
       atomicAdd(unary_grd+b_idx*(L-1)*p + (start+1)*p + sc, tmp);
       atomicAdd(g_binary + s_A*mp*mp + (s_B+m)*mp + (sc+m), tmp);
    }
}




void cuda_forward(const float *binary,   const float *binary_dc, const float *binary_close,  const float *unary, float *alpha,  float *alpha_d, int B, int L, int m, int p, int d)
{
    dim3 gridDim(B, L-2+1, m);
    dim3 blockDim(p);
    kernel_forward_len2<<<gridDim, blockDim>>>(binary, unary, alpha, B, L, m, p);

dim3 gridDim2(B*(L-2), (L-2)*(L-2), d);
dim3 blockDim2(d);
kernel_forward_d<<<gridDim2, blockDim2>>>( binary_dc, unary, alpha, alpha_d, B, L, 2, m, p, d);

for(int w=3; w<L; w++){
  dim3 gridDim(B, L-w, m);
  dim3 blockDim(m);
  kernel_forward<<<gridDim, blockDim>>>(binary, unary, alpha, B, L, w, m, p);
  dim3 gridDim2(B, L-w, m);
  dim3 blockDim2(d);
  kernel_forward_close<<<gridDim2, blockDim2>>>(binary_close, unary, alpha, alpha_d, B, L, w, m, p, d);
  if(w<L-1){
    dim3 gridDim3(B*(L-w),  (w-1)*(L-w), d);
    dim3 blockDim3(d);
    kernel_forward_d<<<gridDim3, blockDim3>>>( binary_dc, unary, alpha, alpha_d, B, L, w, m, p, d);
  }
}
}





void cuda_backward(const float *binary,   const float *binary_dc, const float *binary_close,
                const float *unary,  float *binary_grd, float *binary_dc_grd, float *binary_close_grd,
                  float *unary_grd, float *alpha,  float *alpha_d, int B, int L, int m, int p, int d)

{
    for(int w=L-1; w>2; w--){
        if(w<L-1){
            dim3 gridDim3(B*(L-w),  (w-1)*(L-w), d);
            dim3 blockDim3(d);
            kernel_backward_d<<<gridDim3, blockDim3>>>(binary_dc, unary,  binary_dc_grd, unary_grd, alpha, alpha_d, B, L, w, m, p, d);
        }
        dim3 gridDim(B, L-w, m);
        dim3 blockDim(m);
        kernel_backward<<<gridDim, blockDim>>>(binary, unary,  binary_grd, unary_grd, alpha, B, L, w, m, p);
        dim3 gridDim2(B, L-w, m);
        dim3 blockDim2(d);
        kernel_backward_close<<<gridDim2, blockDim2>>>(binary_close, unary, binary_close_grd, unary_grd, alpha, alpha_d, B, L, w, m, p, d);

    }
    dim3 gridDim2(B*(L-2), (L-2)*(L-2), d);
    dim3 blockDim2(d);
    kernel_backward_d<<<gridDim2, blockDim2>>>( binary_dc, unary,  binary_dc_grd, unary_grd, alpha, alpha_d, B, L, 2, m, p, d);
    dim3 gridDim(B, L-2, m);
    dim3 blockDim(p);
    kernel_backward_len2<<<gridDim, blockDim>>>(binary, unary, binary_grd, unary_grd, alpha, B, L, m, p);
}

void cuda_forward_argmax(const float *binary, const float *binary_dc,  const float *binary_close, const float *unary, float *alpha,  float *alpha_d, int B, int L, int m, int p, int d)
{
    dim3 gridDim(B, L-2, m);
    dim3 blockDim(p);
    kernel_forward_len2_argmax<<<gridDim, blockDim>>>(binary, unary, alpha, B, L, m, p);

   dim3 gridDim2(B*(L-2), (L-2)*(L-2), d);
   dim3 blockDim2(d);
   kernel_forward_d_argmax<<<gridDim2, blockDim2>>>(binary_dc, unary, alpha, alpha_d, B, L, 2, m, p, d);

for(int w=3; w<L; w++){
  dim3 gridDim(B, L-w, m);
  dim3 blockDim(m);
  kernel_forward_argmax<<<gridDim, blockDim>>>(binary, unary, alpha, B, L, w, m, p);
  dim3 gridDim2(B, L-w, m);
  dim3 blockDim2(d);
  kernel_forward_close_argmax<<<gridDim2, blockDim2>>>(binary_close, unary, alpha, alpha_d, B, L, w, m, p, d);
  if(w<L-1){
    dim3 gridDim3(B*(L-w),  (w-1)*(L-w), d);
    dim3 blockDim3(d);
    kernel_forward_d_argmax<<<gridDim3, blockDim3>>>(binary_dc, unary, alpha, alpha_d, B, L, w, m, p, d);
  }
}
}

