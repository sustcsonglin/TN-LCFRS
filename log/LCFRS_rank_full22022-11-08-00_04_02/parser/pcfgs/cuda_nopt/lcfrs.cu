#include <stdio.h>
#include <cuda_runtime.h>


template <typename scalar_t>
__device__ scalar_t logsumexp(scalar_t a, scalar_t b) {
   scalar_t m = max(a, b);
  return log(exp(a - m) + exp(b - m)) + m;
}



template <typename F>
__global__ void kernel_forward( F *__restrict__ binary_rule,
                                F *__restrict__ binary_rule_d,
                                F *__restrict__ alpha,
                                F *__restrict__ alpha_d,
                                int B,  int L,  int width,  int m)
{
     int b_idx = blockIdx.x;
     int start = blockIdx.y;
     int s_A = blockIdx.z;
     int s_B = threadIdx.x;

    if(start+width >= L){
        return;
    }

    __shared__ float result[200];
    __shared__ float left[200];
    __shared__ float right[200];

    int end = start + width;
    F *__restrict__  s = alpha + b_idx * L * L * m;
    F *__restrict__  s_d = alpha_d + b_idx * L * L * L * L * m;
    F *__restrict__  binary = binary_rule + b_idx * m * m * m;
    F *__restrict__  binary_d = binary_rule_d + b_idx * m * m * m;

    float tmp_result = logf(0);

    for (int split = start+1; split < start+width; split++)
    {
         left[s_B] =  s[start * L * m + split * m + s_B];
         right[s_B] = s[split * L * m + end * m + s_B];
         __syncthreads();
         for(int sc=0; sc<m; sc++){
              tmp_result = logsumexp(tmp_result, left[s_B] + right[sc] + binary[s_A*(m)*(m) + s_B*m + sc]);
         }
         __syncthreads();

         for(int split2 = split+1; split2 < start+width; split2++){
            right[s_B] = s[split * L * m + split2 * m + s_B];
            left[s_B] = s_d[start*L*L*L*m + split*L*L*m + split2*L*m + end*m + s_B];
            __syncthreads();
            for(int sc=0; sc<m; sc++){
                tmp_result = logsumexp(tmp_result, left[s_B] + right[sc] + binary_d[s_A*(m)*(m) + s_B*m + sc]);
            }
            __syncthreads();
         }
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
__global__ void kernel_forward_argmax( F *__restrict__ binary_rule,
                                F *__restrict__ binary_rule_d,
                                F *__restrict__ alpha,
                                F *__restrict__ alpha_d,
                                 int B,  int L,  int width,  int m)
{
     int b_idx = blockIdx.x;
     int start = blockIdx.y;
     int s_A = blockIdx.z;
     int s_B = threadIdx.x;

    if(start+width >= L){
        return;
    }

    __shared__ float result[200];
    __shared__ float result_idx[200];
    __shared__ float left[200];
    __shared__ float right[200];

    int end = start + width;
    F *__restrict__  s = alpha + b_idx * L * L * m;
    F *__restrict__  s_d = alpha_d + b_idx * L * L * L * L * m;
    F *__restrict__  binary = binary_rule + b_idx * m * m * m;
    F *__restrict__ binary_d = binary_rule_d + b_idx * m * m * m;

    float tmp_result = logf(0);
    float tmp_idx = -9999;

    for (int split = start+1; split < start+width; split++)
    {
         left[s_B] =  s[start * L * m + split * m + s_B];
         right[s_B] = s[split * L * m + end * m + s_B];
         __syncthreads();
         for(int sc=0; sc<m; sc++){
               float tmp = left[s_B] + right[sc] + binary[s_A*(m)*(m) + s_B*m + sc];
               if(tmp > tmp_result){
                    tmp_result = tmp;
                    tmp_idx = split * m * m + s_B * m + sc;
               }
         }
         __syncthreads();

         for(int split2 = split+1; split2 < start+width; split2++){
            right[s_B] = s[split * L * m + split2 * m + s_B];
            left[s_B] = s_d[start*L*L*L*m + split*L*L*m + split2*L*m + end*m + s_B];
            __syncthreads();
            for(int sc=0; sc<m; sc++){
                float tmp = left[s_B] + right[sc] + binary_d[s_A*(m)*(m) + s_B*m + sc];
                if (tmp > tmp_result){
                    tmp_result = tmp;
                    tmp_idx = -(split*L*m*m + split2*m*m + s_B * m + sc);
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
    for(int i=0;i<m;i++){
        if(result[i] > final_result){
            final_result = result[i];
            final_idx = result_idx[i];
        }
    }
    alpha[b_idx*L*L*m+start*L*m+end*m+s_A] = final_result;
    alpha[b_idx*L*L*m+end*L*m+start*m+s_A] = final_idx;
    }
}


template <typename F>
__global__ void kernel_backward( F *__restrict__ binary_rule,
                                 F *__restrict__ binary_rule_grd,
                                 F *__restrict__ binary_rule_d,
                                 F *__restrict__ binary_rule_d_grd,
                                 F *__restrict__ alpha,
                                 F *__restrict__ alpha_d,
                                  int B,  int L,  int width,  int m)
{
     int b_idx = blockIdx.x;
     int start = blockIdx.y;
     int s_A = blockIdx.z;
     int s_B = threadIdx.x;

    if(start+width >= L){
        return;
    }

    __shared__ float left[200];
    __shared__ float right[200];

    int end = start + width;
    F *__restrict__  s = alpha + b_idx * L * L * m;
    F *__restrict__  s_d = alpha_d + b_idx * L * L * L * L * m;
    F *__restrict__  binary = binary_rule + b_idx * m * m * m;
    F *__restrict__  binary_grd = binary_rule_grd + b_idx * m * m * m;
    F *__restrict__ binary_d = binary_rule_d + b_idx * m * m * m;
    F *__restrict__ binary_d_grd = binary_rule_d_grd + b_idx * m * m * m;

    float parent_inside = s[start*L*m + end*m + s_A];
    float parent_grd = s[end*L*m + start*m + s_A];

    for (int split = start+1; split < start+width; split++)
    {
         left[s_B] =  s[start * L * m + split * m + s_B];
         right[s_B] = s[split * L * m + end * m + s_B];
         __syncthreads();
         for(int sc=0; sc<m; sc++){
             float tmp = exp( left[s_B] + right[sc] + binary[s_A*(m)*(m) + s_B*m + sc] - parent_inside) ;
             if(tmp > 1){
                printf("??? %f %f %f %d %d %d  %d %d \n", parent_inside, left[s_B], right[sc], start, split, end, m, L);
             }
             tmp *= parent_grd;
             atomicAdd(s + split*L*m + start*m + s_B, tmp);
             atomicAdd(s + end*L*m + split*m + sc, tmp);
             atomicAdd(binary_grd +  s_A*(m)*(m) + s_B*m + sc, tmp);
         }
         __syncthreads();

         for(int split2 = split+1; split2 < start+width; split2++){
            right[s_B] = s[split * L * m + split2 * m + s_B];
            left[s_B] = s_d[start*L*L*L*m + split*L*L*m + split2*L*m + end*m + s_B];
            __syncthreads();
            for(int sc=0; sc<m; sc++){
                float tmp = exp(left[s_B] + right[sc] + binary_d[s_A*(m)*(m) + s_B*m + sc] - parent_inside);
                if(tmp > 1){
                    printf("!!!\n");
                }
                tmp*=parent_grd;
                atomicAdd(s + split2*L*m + split*m + sc, tmp);
                atomicAdd(s_d + split*L*L*L*m + start*L*L*m + split2*L*m + end*m + s_B, tmp);
                atomicAdd(binary_d_grd +  s_A*(m)*(m) + s_B*m + sc, tmp);
            }
            __syncthreads();
         }
    }
}



template <typename F>
__global__ void kernel_forward_d(F *__restrict__ binary_rule_d,
                                 F *__restrict__ binary_rule_dc,
                                 F *__restrict__ alpha,
                                 F *__restrict__ alpha_d,
                                  int B,  int L,  int width,  int m)
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
     int s_B = threadIdx.x;
     int gap_start = start + gap_start_minus_start + 1;
     int gap_end = gap_start + gap_width;
    if(gap_start - start > width-1)
    {
        return;
    }

    __shared__ float result[200];
    __shared__ float left[200];
    __shared__ float right[200];

    F *__restrict__ s = alpha + b_idx * L * L * m;
    F *__restrict__ s_d = alpha_d + b_idx * L * L * L * L * m;
    F *__restrict__ binary = binary_rule_d + b_idx * m * m * m * 4;
    F *__restrict__ binary_c = binary_rule_dc + b_idx * m * m * m;
     int L3 = L * L * L * m;
     int L2 = L * L * m;
     int L1 = L * m;

    float tmp_result = logf(0);

    left[s_B] = s[start*L*m + gap_start*m + s_B];
    right[s_B] = s[gap_end*L*m + end*m + s_B];
    __syncthreads();

    for(int sc=0; sc<m; sc++)
    {
        tmp_result = logsumexp(tmp_result, left[s_B] + right[sc] + binary_c[s_A*m*m + s_B*m + sc]);
    }
    __syncthreads();

    for(int split=start+1; split< gap_start; split++)
    {
        right[s_B] = s[start * L * m + split * m + s_B];
        left[s_B] =  s_d[split*L3 + gap_start*L2 + gap_end*L1 + end*m +s_B];
        __syncthreads();
        for(int sc=0; sc<m; sc++)
        {
            tmp_result = logsumexp(tmp_result, right[sc]+left[s_B] + binary[s_A* m * m *4 + s_B*m*4 + sc*4 + 0]);
        }

        __syncthreads();

        left[s_B] = s_d[start* L3 + split * L2 + gap_end * L1 + end*m +s_B];
        right[s_B] = s[split * L * m + gap_start * m + s_B];

        __syncthreads();
        for(int sc=0; sc<m; sc++)
        {
            tmp_result = logsumexp(tmp_result, right[sc]+left[s_B]+ binary[s_A*m*m*4 + s_B*m*4 + (sc)*4 +1]);
        }
        __syncthreads();
     }

     for(int split=gap_end+1; split <end; split++){
        left[s_B] = s_d[start*L3 + gap_start*L2 + split*L1 + end*m +s_B];
        right[s_B] = s[gap_end*L*m + split*m + s_B];
        __syncthreads();
        for(int sc=0;sc<m;sc++){
            tmp_result = logsumexp(tmp_result,  left[s_B]+right[sc] + binary[s_A*m*m*4 + s_B*m*4 + (sc)*4 + 2]);
        }
        __syncthreads();

        left[s_B] = s_d[start*L3 + gap_start*L2 + gap_end*L1 + split*m +s_B];
        right[s_B] = s[split*L*m + end*m + s_B];
        __syncthreads();
        for(int sc=0;sc<m;sc++){
             tmp_result = logsumexp(tmp_result, left[s_B] + right[sc] + binary[s_A*m*m*4 + s_B*m*4 + (sc)*4+ 3]);
        }
        __syncthreads();
     }

    result[s_B] = tmp_result;
    __syncthreads();
    if(s_B==0){
    float final_result = logf(0);
    for(int i=0;i<m;i++){
        final_result = logsumexp(final_result, result[i]);
    }
    alpha_d[b_idx*L*L*L*L*m + start*L3 + gap_start*L2 + gap_end*L1 + end*m + s_A] = final_result;
  }
}



template <typename F>
__global__ void kernel_forward_d_argmax(F *__restrict__ binary_rule_d,
                                 F *__restrict__ binary_rule_dc,
                                 F *__restrict__ alpha,
                                 F *__restrict__ alpha_d,
                                  int B,  int L,  int width,  int m)
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
     int s_B = threadIdx.x;
     int gap_start = start + gap_start_minus_start + 1;
     int gap_end = gap_start + gap_width;
    if(gap_start - start > width-1)
    {
        return;
    }

    __shared__ float result[200];
    __shared__ float result_idx[200];
    __shared__ float left[200];
    __shared__ float right[200];

    F *__restrict__ s = alpha + b_idx * L * L * m;
    F *__restrict__ s_d = alpha_d + b_idx * L * L * L * L * m;
    F *__restrict__ binary = binary_rule_d + b_idx * m * m * m * 4;
    F *__restrict__ binary_c = binary_rule_dc + b_idx * m * m * m;
     int L3 = L * L * L * m;
     int L2 = L * L * m;
     int L1 = L * m;

    float tmp_result = logf(0);
    float tmp_idx = logf(0);

    left[s_B] = s[start*L*m + gap_start*m + s_B];
    right[s_B] = s[gap_end*L*m + end*m + s_B];
    __syncthreads();

    for(int sc=0; sc<m; sc++)
    {
        float tmp = left[s_B] + right[sc] + binary_c[s_A*m*m + s_B*m + sc];
        if ( tmp > tmp_result){
            tmp_result = tmp;
            tmp_idx = -(s_B * m + sc);
        }

    }
    __syncthreads();

    for(int split=start+1; split< gap_start; split++)
    {
        right[s_B] = s[start * L * m + split * m + s_B];
        left[s_B] =  s_d[split*L3 + gap_start*L2 + gap_end*L1 + end*m +s_B];
        __syncthreads();
        for(int sc=0; sc<m; sc++)
        {
            float tmp = left[s_B]+right[sc] + binary[s_A*m*m*4 + s_B*m*4 + sc*4 + 0];
            if (tmp > tmp_result){
                tmp_result = tmp;
                tmp_idx = split * m * m * 4 + s_B * m * 4 + sc * 4 + 0;
            }
        }
        __syncthreads();

        right[s_B] = s[split * L * m + gap_start * m + s_B];
        left[s_B] = s_d[ start* L3 + split * L2 + gap_end * L1 + end*m +s_B];

        __syncthreads();
        for(int sc=0; sc<m; sc++)
        {
            float tmp = right[sc]+left[s_B]+ binary[s_A*m*m*4 + s_B*m*4 + (sc)*4 +1];
            if (tmp > tmp_result){
                tmp_result = tmp;
                tmp_idx = split * m * m * 4 + s_B * m * 4 + sc * 4 + 1;
            }
        }
        __syncthreads();
     }

     for(int split=gap_end+1; split <end; split++){
        left[s_B] = s_d[start*L3 + gap_start*L2 + split*L1 + end*m +s_B];
        right[s_B] = s[gap_end*L*m + split*m + s_B];
        __syncthreads();
        for(int sc=0;sc<m;sc++){
            float tmp = left[s_B] + right[sc] + binary[s_A*m*m*4 + s_B*m*4 + (sc)*4 + 2];
            if (tmp > tmp_result){
                tmp_result = tmp;
                tmp_idx = split * m * m * 4  + s_B * m * 4 + sc * 4 + 2;
            }
        }
        __syncthreads();

        left[s_B] = s_d[start*L3 + gap_start*L2 + gap_end*L1 + split*m +s_B];
        right[s_B] = s[split*L*m + end*m + s_B];
        __syncthreads();
        for(int sc=0;sc<m;sc++){
             float tmp = left[s_B] + right[sc] + binary[s_A*m*m*4 + s_B*m*4 + (sc)*4+ 3];
             if (tmp > tmp_result){
                tmp_result = tmp;
                tmp_idx = split * m * m * 4  + s_B * m * 4 + sc * 4 + 3;
            }
        }
        __syncthreads();
     }

    result[s_B] = tmp_result;
    result_idx[s_B] = tmp_idx;
    __syncthreads();
    if(s_B==0){
    float final_result = logf(0);
    float final_idx = -999;
    for(int i=0;i<m;i++){
        if(result[i] > final_result){
            final_result = result[i];
            final_idx = result_idx[i];
        }
    }
    alpha_d[b_idx*L*L*L*L*m + start*L3 + gap_start*L2 + gap_end*L1 + end*m + s_A] = final_result;
    alpha_d[b_idx*L*L*L*L*m + gap_start*L3 + start*L2 + gap_end*L1 + end*m + s_A] = final_idx;
  }
}


template <typename F>
__global__ void kernel_backward_d(F *__restrict__ binary_rule_d,
                                  F *__restrict__ binary_rule_d_grd,
                                  F *__restrict__ binary_rule_dc,
                                  F *__restrict__ binary_rule_dc_grd,
                                  F *__restrict__ alpha,
                                  F *__restrict__ alpha_d,
                                   int B,  int L,  int width,  int m)
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
     int s_B = threadIdx.x;
     int gap_start = start + gap_start_minus_start + 1;
     int gap_end = gap_start + gap_width;
    if(gap_start - start > width-1)
    {
        return;
    }

    __shared__ float left[200];
    __shared__ float right[200];

    F *__restrict__ s = alpha + b_idx * L * L * m;
    F *__restrict__ s_d = alpha_d + b_idx * L * L * L * L * m;
    F *__restrict__ binary = binary_rule_d + b_idx * m * m * m * 4;
    F *__restrict__ binary_grd = binary_rule_d_grd + b_idx * m * m * m * 4;
    F *__restrict__ binary_c = binary_rule_dc + b_idx * m * m * m;
    F *__restrict__ binary_c_grd = binary_rule_dc_grd + b_idx * m * m * m;
    
    int L3 = L * L * L * m;
    int L2 = L * L * m;
    int L1 = L * m;

    float parent_inside = s_d[start*L3 + gap_start*L2 + gap_end*L1 + end*m + s_A];
    float parent_grd = s_d[gap_start*L3 + start*L2 + gap_end*L1 + end*m + s_A];

    left[s_B] = s[start*L*m + gap_start*m + s_B];
    right[s_B] = s[gap_end*L*m + end*m + s_B];
    __syncthreads();

    for(int sc=0; sc<m; sc++)
    {
        float tmp = exp(left[s_B] + right[sc] + binary_c[s_A*m*m + s_B*m + sc] - parent_inside) ;
        if(tmp > 1){
        printf("? 1 \n");
        }
        tmp *= parent_grd;
        atomicAdd(s + gap_start*L*m + start*m + s_B, tmp);
        atomicAdd(s + end*L*m + gap_end*m + sc, tmp);
        atomicAdd(binary_c_grd + s_A*m*m + s_B*m + sc, tmp);
    }
    __syncthreads();

    for(int split=start+1; split< gap_start; split++)
    {
        right[s_B] = s[start * L * m + split * m + s_B];
        left[s_B] =  s_d[split*L3 + gap_start*L2 + gap_end*L1 + end*m +s_B];
        __syncthreads();
        for(int sc=0; sc<m; sc++)
        {
           float tmp = exp(left[s_B]+right[sc] + binary[s_A*m*m*4 + s_B*m*4 + sc*4 + 0] - parent_inside);
           if(tmp > 1){
            printf("?\n");
           }
           tmp*=parent_grd;
           atomicAdd(s + split*L * m + start * m + sc, tmp );
           atomicAdd(s_d + gap_start*L3 + split*L2 + gap_end*L1 + end*m + s_B, tmp);
           atomicAdd(binary_grd + s_A*m*m*4 + s_B*m*4 + sc*4 + 0, tmp);
        }
        __syncthreads();

        right[s_B] = s[split * L * m + gap_start * m + s_B];
        left[s_B] = s_d[ start* L3 + split * L2 + gap_end * L1 + end*m +s_B];
        __syncthreads();
        for(int sc=0; sc<m; sc++)
        {
           float tmp = exp(right[sc]+left[s_B]+ binary[s_A*m*m*4 + s_B*m*4 + (sc)*4 +1] - parent_inside) * parent_grd;
           atomicAdd(s + gap_start*L * m + split * m + sc, tmp );
           atomicAdd(s_d + split*L3 + start*L2 + gap_end*L1 + end*m + s_B, tmp);
           atomicAdd(binary_grd + s_A*m*m*4 + s_B*m*4 + sc*4 + 1, tmp);
        }
        __syncthreads();
     }

     for(int split=gap_end+1; split <end; split++){
        left[s_B] = s_d[start*L3 + gap_start*L2 + split*L1 + end*m +s_B];
        right[s_B] = s[gap_end*L*m + split*m + s_B];
        __syncthreads();
        for(int sc=0;sc<m;sc++){
            float tmp =  exp(left[s_B]+right[sc] + binary[s_A*m*m*4 + s_B*m*4 + (sc)*4 + 2] - parent_inside) * parent_grd;
            atomicAdd(s + split*L*m + gap_end*m + sc, tmp);
            atomicAdd(s_d + gap_start*L3 + start*L2 + split*L1 + end*m + s_B, tmp);
            atomicAdd(binary_grd + s_A*m*m*4 + s_B*m*4 + (sc)*4 + 2, tmp);
        }
        __syncthreads();

        left[s_B] = s_d[start*L3 + gap_start*L2 + gap_end*L1 + split*m +s_B];
        right[s_B] = s[split*L*m + end*m + s_B];
        __syncthreads();

        for(int sc=0;sc<m;sc++){
             float tmp = exp(left[s_B] + right[sc] + binary[s_A*m*m*4 + s_B*m*4 + (sc)*4+ 3] - parent_inside) * parent_grd;
             atomicAdd(s + end*L*m + split*m + sc, tmp);
             atomicAdd(s_d + gap_start*L3 + start*L2 + gap_end*L1 + split*m + s_B, tmp);
             atomicAdd(binary_grd + s_A*m*m*4 + s_B*m*4 + (sc)*4 + 3, tmp);
        }
        __syncthreads();
    }
}


void cuda_forward( float *binary,  float *binary_close,   float *binary_dc,   float *binary_d, float *alpha,  float *alpha_d, int B, int L, int m)
{
 for(int w=2; w<L; w++){
   dim3 gridDim(B, L-w, m);
   dim3 blockDim(m);
   kernel_forward<<<gridDim, blockDim>>>(binary, binary_close, alpha, alpha_d, B, L, w, m);
   if(w<L-1){
    dim3 gridDim3(B*(L-w),  (w-1)*(L-w), m);
    dim3 blockDim3(m);
    kernel_forward_d<<<gridDim3, blockDim3>>>(binary_d, binary_dc, alpha, alpha_d, B, L, w, m);
   }
 }
}


void cuda_forward_argmax( float *binary,  float *binary_close,   float *binary_dc,   float *binary_d, float *alpha,  float *alpha_d, int B, int L, int m)
{
   for(int w=2; w<L; w++){
       dim3 gridDim(B, L-w, m);
        dim3 blockDim(m);
        kernel_forward_argmax<<<gridDim, blockDim>>>(binary, binary_close, alpha, alpha_d, B, L, w, m);
        if(w<L-1){
            dim3 gridDim3(B*(L-w),  (w-1)*(L-w), m);
            dim3 blockDim3(m);
            kernel_forward_d_argmax<<<gridDim3, blockDim3>>>(binary_d, binary_dc, alpha, alpha_d, B, L, w, m);
        }
    }
}

void cuda_backward( float *binary,  float *binary_close,   float *binary_dc,   float *binary_d,
                 float *binary_grd, float *binary_close_grd, float *binary_dc_grd,
                 float *binary_d_grd, float *alpha,  float *alpha_d, int B, int L, int m)
{

    for(int w=L-1; w>1; w--){
        if(w<L-1){
            dim3 gridDim3(B*(L-w),  (w-1)*(L-w), m);
            dim3 blockDim3(m);
            kernel_backward_d<<<gridDim3, blockDim3>>>(binary_d, binary_d_grd, binary_dc, binary_dc_grd,  alpha, alpha_d, B, L, w, m);
        }

        dim3 gridDim(B, L-w, m);
        dim3 blockDim(m);
        kernel_backward<<<gridDim, blockDim>>>(binary,  binary_grd, binary_close, binary_close_grd,  alpha, alpha_d, B, L, w, m);
    }
}
