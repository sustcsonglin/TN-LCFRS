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



__global__ void kernel_forward( float *  head_cl1,
                                float *  head_cr1,
                                float *  head_dl1,
                                float *  head_dr1,
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
                                int B, int L, int width, int r1, int r2, int r3, int r4)
{
    int b_idx = blockIdx.x;
    int start = blockIdx.y;
    int s_A = blockIdx.z;
    int s_R = threadIdx.x;

    if(start+width >= L){
        return;
    }

    __shared__ float result_l[500];
    __shared__ float result_r[500];

    __shared__ float result_ld[500];
    __shared__ float result_rd[500];

    __shared__ float result_c[500];
    __shared__ float result_d1[500];
    __shared__ float result_d2[500];
    __shared__ float result_d3[500];
    __shared__ float result_d4[500];


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
               tmp_result2 = logsumexpf(tmp_result2, s_cc[split*L*r2 + split2*r2 + s_R] + s_cd[start*L3 + split*L2 + split2*L1 + end*r2 + s_R]);
            }
        }
    }

    result_l[s_R] = tmp_result + head_cl1[b_idx*r1*r1 + s_A*r1 + s_R];
    result_r[s_R] = tmp_result + head_cr1[b_idx*r1*r1 + s_A*r1 + s_R];

    result_ld[s_R] = tmp_result + head_dl1[b_idx*r1*r1 + s_A*r1 + s_R];
    result_rd[s_R] = tmp_result + head_dr1[b_idx*r1*r1 + s_A*r1 + s_R];

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

    int i = 2;
    while(i < r1){
        i*=2;
    }

    for(unsigned int s= i/2; s>0; s>>=1){
        if((s_R < s) & (s_R+ s < r1)){
           result_l[s_R] = logsumexpf(result_l[s_R+s], result_l[s_R]);
           result_r[s_R] = logsumexpf(result_r[s_R+s], result_r[s_R]);
           result_ld[s_R] = logsumexpf(result_ld[s_R+s], result_ld[s_R+s]);
           result_rd[s_R] = logsumexpf(result_rd[s_R+s], result_rd[s_R+s]);
        }

        if(s_A < r2){
            if((s_R < s) & (s_R+s < r1)){
                result_c[s_R] = logsumexpf(result_c[s_R+s], result_c[s_R]);
                result_d1[s_R] = logsumexpf(result_d1[s_R+s], result_d1[s_R]);
                result_d2[s_R] = logsumexpf(result_d2[s_R+s], result_d2[s_R]);
                result_d3[s_R] = logsumexpf(result_d3[s_R+s], result_d3[s_R]);
                result_d4[s_R] = logsumexpf(result_d4[s_R+s], result_d4[s_R]);
            }
        }

        __syncthreads();
    }



    if(s_R==0){
        alpha_l[b_idx*L*L*r1 +start*L*r1 +end*r1 + s_A] = result_l[0];
        alpha_r[b_idx*L*L*r1 +start*L*r1 +end*r1 + s_A] = result_r[0];
        if(s_A < r2){
            alpha_cc[b_idx*L*L*r2 + start*L*r2 +end*r2 + s_A] = result_c[0];
            alpha_dc[b_idx*L*L*r2*4 + start*L*r2*4 +end*r2*4 + 0*r2 + s_A] = result_d1[0];
            alpha_dc[b_idx*L*L*r2*4 + start*L*r2*4 +end*r2*4 + 1*r2 + s_A] = result_d2[0];
            alpha_dc[b_idx*L*L*r2*4 + start*L*r2*4 +end*r2*4 + 2*r2 + s_A] = result_d3[0];
            alpha_dc[b_idx*L*L*r2*4 + start*L*r2*4 +end*r2*4 + 3*r2 + s_A] = result_d4[0];
        }
    }
}



__global__ void kernel_backward( float * head_cl1,
                                 float * head_cr1,
                                 float * head_dl1,
                                 float * head_dr1,

                                 float *  head_cc1,
                                 float *  head_dc1,
                                 float *  head_cl2,
                                 float *  head_cr2,
                                 float *  head_cc2,
                                 float *  head_dc2,

                                 float *  head_cl1_grd,
                                 float *  head_cr1_grd,
                                 float *  head_cc1_grd,
                                 float *  head_dc1_grd,
                                 float *  head_cl2_grd,
                                 float *  head_cr2_grd,
                                 float *  head_cc2_grd,
                                 float *  head_dc2_grd,

                                 float * alpha_l,
                                 float * alpha_r,
                                 float * alpha_cc,
                                 float * alpha_cd,
                                 float * alpha_dc,

                                 int B,   int L,   int width,   int r1,   int r2)
{
      int b_idx = blockIdx.x;
      int start = blockIdx.y;
      int s_A = blockIdx.z;
      int s_R = threadIdx.x;

      if(start+width >= L){
        return;
      }

      int end = start + width;
      float *  s_l = alpha_l + b_idx * L * L * r1;
      float *  s_r = alpha_r + b_idx * L * L * r1;
      float *  s_cc = alpha_cc + b_idx * L * L * r2;
      float *  s_cd = alpha_cd + b_idx * L * L * L * L * r2;
      float *  s_dc = alpha_dc + b_idx * L * L * r2 * 4;

      int L3 = L*L*L*r2;
      int L2 = L*L*r2;
      int L1 = L*r2;

      float parent_inside_l = s_l[start * L * r1 + end * r1 + s_A];
      float parent_grd_l = s_l[end * L * r1 + start * r1 + s_A];

      float parent_inside_r = s_r[start * L * r1 + end * r1 + s_A];
      float parent_grd_r = s_r[end * L * r1 + start * r1 + s_A];


      float parent_inside_dc1 = 0;
      float parent_inside_dc2 = 0;
      float parent_inside_dc3 = 0;
      float parent_inside_dc4 = 0;

      float parent_inside_cc = 0;
      float parent_grd_cc = 0;

      float parent_grd_dc1 = 0;
      float parent_grd_dc2 = 0;
      float parent_grd_dc3 = 0;
      float parent_grd_dc4 = 0;

      float dc11 = 0;
      float dc12 = 0;
      float dc13 = 0;
      float dc14 = 0;

      float dc21 = 0;
      float dc22 = 0;
      float dc23 = 0;
      float dc24 = 0;

      float cl2 = 0;
      float cr2 = 0;
      float cc2 = 0;
      float cc1 = 0;


      if(s_A < r2){
        parent_inside_dc1 = s_dc[start * L * r2 * 4 + end * r2 * 4 + 0 * r2 + s_A];
        parent_inside_dc2 = s_dc[start * L * r2 * 4 + end * r2 * 4 + 1 * r2 + s_A];
        parent_inside_dc3 = s_dc[start * L * r2 * 4 + end * r2 * 4 + 2 * r2 + s_A];
        parent_inside_dc4 = s_dc[start * L * r2 * 4 + end * r2 * 4 + 3 * r2 + s_A];

        parent_grd_dc1 = s_dc[end * L * r2 * 4 + start * r2 * 4 + 0 * r2 + s_A];
        parent_grd_dc2 = s_dc[end * L * r2 * 4 + start * r2 * 4 + 1 * r2 + s_A];
        parent_grd_dc3 = s_dc[end * L * r2 * 4 + start * r2 * 4 + 2 * r2 + s_A];
        parent_grd_dc4 = s_dc[end * L * r2 * 4 + start * r2 * 4 + 3 * r2 + s_A];

        dc11 = head_dc1[b_idx*r1*r2*4 + s_A*r1*4 + 0*r1 + s_R];
        dc12 = head_dc1[b_idx*r1*r2*4 + s_A*r1*4 + 1*r1 + s_R];
        dc13 = head_dc1[b_idx*r1*r2*4 + s_A*r1*4 + 2*r1 + s_R];
        dc14 = head_dc1[b_idx*r1*r2*4 + s_A*r1*4 + 3*r1 + s_R];

        cc1 =  head_cc1[b_idx*r1*r2 + s_A*r1 + s_R];

        parent_inside_cc = s_cc[start * L * r2 + end * r2 + s_A];
        parent_grd_cc = s_cc[end * L * r2 + start * r2 + s_A];
      }

      if(s_R < r2){


        cl2 = head_cl2[b_idx*r1*r2 + s_A*r2 + s_R];
        cr2 = head_cr2[b_idx*r1*r2 + s_A*r2 + s_R];
        if(s_A < r2){
        cc2 = head_cc2[b_idx*r2*r2 + s_A*r2 + s_R];
        dc21 = head_dc2[b_idx*r2*r2*4 + s_A*r2*4 + 0*r2 + s_R];
        dc22 = head_dc2[b_idx*r2*r2*4 + s_A*r2*4 + 1*r2 + s_R];
        dc23 = head_dc2[b_idx*r2*r2*4 + s_A*r2*4 + 2*r2 + s_R];
        dc24 = head_dc2[b_idx*r2*r2*4 + s_A*r2*4 + 3*r2 + s_R];
        }
      }

      float cl1 = head_cl1[b_idx*r1*r1 + s_A*r1 + s_R];
      float cr1 = head_cr1[b_idx*r1*r1 + s_A*r1 + s_R];



      float cl2_grd = 0;
      float cr2_grd = 0;

      float dc11_grd = 0;
      float dc12_grd = 0;
      float dc13_grd = 0;
      float dc14_grd = 0;
      float cl1_grd = 0;
      float cr1_grd = 0;
      float cc1_grd = 0;

      float cc2_grd = 0;
      float dc21_grd = 0;
      float dc22_grd = 0;
      float dc23_grd = 0;
      float dc24_grd = 0;

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

        if(s_A < r2){
             tmp_grd = expf(tmp + cc1 - parent_inside_cc) * parent_grd_cc;
             grd += tmp_grd;
             cc1_grd += tmp_grd;

             tmp_grd = expf(tmp + dc11 - parent_inside_dc1) * parent_grd_dc1;
             grd += tmp_grd;
             dc11_grd += tmp_grd;

             tmp_grd = expf(tmp + dc12 - parent_inside_dc2) * parent_grd_dc2;
             grd += tmp_grd;
             dc12_grd += tmp_grd;

             tmp_grd = expf(tmp + dc13 - parent_inside_dc3) * parent_grd_dc3;
             grd += tmp_grd;
             dc13_grd += tmp_grd;

             tmp_grd = expf(tmp + dc14 - parent_inside_dc4) * parent_grd_dc4;
             grd += tmp_grd;
             dc14_grd += tmp_grd;
        }

        atomicAdd(s_l + split*L*r1 + start*r1 + s_R, grd);
        atomicAdd(s_r + end*L*r1 + split*r1 + s_R, grd);

        if(s_R < r2){

            for(int split2 = split+1; split2 < start+width; split2++){
                float tmp = s_cc[split*L*r2 + split2*r2 + s_R] + s_cd[start*L3 + split*L2 + split2*L1 + end*r2 + s_R];
                float grd = 0;

                float tmp_grd = expf(tmp + cl2 - parent_inside_l) * parent_grd_l;
                cl2_grd += tmp_grd;
                grd += tmp_grd;

                tmp_grd = expf(tmp + cr2 - parent_inside_r) * parent_grd_r;
                cr2_grd += tmp_grd;
                grd += tmp_grd;

                if(s_A < r2){
                    float tmp_grd = expf(tmp + cc2 - parent_inside_cc) * parent_grd_cc;
                    cc2_grd += tmp_grd;
                    grd += tmp_grd;

                    tmp_grd = expf(tmp + dc21 - parent_inside_dc1) * parent_grd_dc1;
                    dc21_grd += tmp_grd;
                    grd += tmp_grd;

                    tmp_grd = expf(tmp + dc22 - parent_inside_dc2) * parent_grd_dc2;
                    dc22_grd += tmp_grd;
                    grd += tmp_grd;

                    tmp_grd = expf(tmp + dc23 - parent_inside_dc3) * parent_grd_dc3;
                    dc23_grd += tmp_grd;
                    grd += tmp_grd;

                    tmp_grd = expf(tmp + dc24 - parent_inside_dc4) * parent_grd_dc4;
                    dc24_grd += tmp_grd;
                    grd += tmp_grd;
                }
                atomicAdd(s_cc + split2*L*r2 + split*r2 + s_R, grd);
                atomicAdd(s_cd + split*L3 + start*L2 + split2*L1 + end*r2 + s_R, grd);
            }
        }
        }

     atomicAdd( head_cl1_grd + b_idx * r1 * r1 + s_A * r1 + s_R, cl1_grd);
     atomicAdd( head_cr1_grd + b_idx * r1 * r1 + s_A * r1 + s_R, cr1_grd);

     if(s_A < r2){
         atomicAdd( head_cc1_grd + b_idx * r1 * r2 + s_A * r1 + s_R, cc1_grd);
         atomicAdd( head_dc1_grd + b_idx * r1 * r2*4 + s_A * r1 * 4 + 0*r1 + s_R, dc11_grd );
         atomicAdd( head_dc1_grd + b_idx * r1 * r2*4 + s_A * r1 * 4 + 1*r1 + s_R, dc12_grd );
         atomicAdd( head_dc1_grd + b_idx * r1 * r2*4 + s_A * r1 * 4 + 2*r1 + s_R, dc13_grd );
         atomicAdd( head_dc1_grd + b_idx * r1 * r2*4 + s_A * r1 * 4 + 3*r1 + s_R, dc14_grd );
     }

     if (s_R < r2){
        atomicAdd( head_cl2_grd + b_idx * r1 * r2 + s_A * r2 + s_R, cl2_grd);
        atomicAdd( head_cr2_grd + b_idx * r1 * r2 + s_A * r2 + s_R, cr2_grd);
        if(s_A < r2){
            atomicAdd( head_cc2_grd + b_idx * r2 * r2 + s_A * r2 + s_R, cc2_grd);
            atomicAdd( head_dc2_grd + b_idx * r2 * r2 * 4 + s_A*r2*4 + 0*r2 + s_R, dc21_grd);
            atomicAdd( head_dc2_grd + b_idx * r2 * r2 * 4 + s_A*r2*4 + 1*r2 + s_R, dc22_grd);
            atomicAdd( head_dc2_grd + b_idx * r2 * r2 * 4 + s_A*r2*4 + 2*r2 + s_R, dc23_grd);
            atomicAdd( head_dc2_grd + b_idx * r2 * r2 * 4 + s_A*r2*4 + 3*r2 + s_R, dc24_grd);
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

    int i = 2;
    while(i < r){
        i*=2;
    }

    for(unsigned int s= i/2; s>0; s>>=1){
        if((s_R < s) & (s_R+ s < r)){
           result[s_R] = logsumexpf(result[s_R], result[s_R+s]);
           result2[s_R] = logsumexpf(result2[s_R], result2[s_R+s]);
        }
        __syncthreads();
    }

    if(s_R==0){
        alpha_cd[b_idx*L*L*L*L*d + start*L*L*L*d + gap_start*L*L*d + gap_end*L*d + end*d + s_A] = result[0];
        alpha_dd[b_idx*L*L*L*L*d + start*L*L*L*d + gap_start*L*L*d + gap_end*L*d + end*d + s_A] = result2[0];
    }

}


__global__ void kernel_backward_d1(  float *   head_cd, float *   head_cd_grd,
                                 float *   head_dd, float * head_dd_grd,
                               float * alpha_ld, float * alpha_rd,float *  alpha_cd,float *  alpha_dd,
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
     float *   s_l = alpha_ld + b_idx * L * L * r;
     float *   s_r = alpha_rd + b_idx * L * L * r;

     float parent_inside = alpha_cd[b_idx*L*L*L*L*d + start*L*L*L*d + gap_start*L*L*d + gap_end*L*d + end*d + s_A];
     float parent_grd = alpha_cd[b_idx*L*L*L*L*d + gap_start*L*L*L*d + start*L*L*d + gap_end*L*d + end*d + s_A];
     float tmp = (s_l[start*L*r + gap_start*r + s_R] + s_r[gap_end*L*r + end*r + s_R]);
     float tmp_grd =  exp(tmp + head_cd[b_idx*d*r+s_R*d+s_A] - parent_inside)*parent_grd;
     atomicAdd(head_cd_grd + b_idx*d*r + s_R*d + s_A, tmp_grd);

     parent_inside = alpha_dd[b_idx*L*L*L*L*d + start*L*L*L*d + gap_start*L*L*d + gap_end*L*d + end*d + s_A];
     parent_grd = alpha_dd[b_idx*L*L*L*L*d + gap_start*L*L*L*d + start*L*L*d + gap_end*L*d + end*d + s_A];
     float tmp_grd2 = exp(tmp + head_dd[b_idx*d*r+s_R*d+s_A] - parent_inside)*parent_grd;
     atomicAdd(s_l + gap_start*L*r + start*r + s_R, tmp_grd+tmp_grd2);
     atomicAdd(s_r + end*L*r + gap_end*r + s_R, tmp_grd+tmp_grd2);
     atomicAdd(head_dd_grd + b_idx*d*r + s_R*d + s_A, tmp_grd2);
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
     __shared__ float result[1000];
     __shared__ float result2[1000];
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

__global__ void kernel_backward_d2(float *  head_cd, float *  head_cd_grd,
                                float *  head_dd, float *  head_dd_grd,
                                float * alpha_dc,
                                float * alpha_dd,
                                float * alpha_cd,
                                int B,  int L,  int width,   int d,   int r)
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

    float * s = alpha_dc + b_idx * L * L * r * 4;
    float * s_d = alpha_dd + b_idx * L * L * L * L * r;
    int L3 = L * L * L * d;
    int L2 = L * L * d;
    int L1 = L * d;
    float tmp_result = logf(0);

    float rule_score1 = head_cd[b_idx*d*r + s_R*d + s_A];
    float rule_score2 = head_dd[b_idx*d*r + s_R*d + s_A];
    float rule_grd = 0;
    float rule_grd2 = 0;

    float parent_inside = alpha_cd[b_idx*L*L*L*L*d + start*L3 + gap_start*L2 + gap_end*L1 + end*d + s_A];
    float parent_grd = alpha_cd[b_idx*L*L*L*L*d + gap_start*L3 + start*L2 + gap_end*L1 + end*d + s_A];
    float parent_inside2 = alpha_dd[b_idx*L*L*L*L*d + start*L3 + gap_start*L2 + gap_end*L1 + end*d + s_A];
    float parent_grd2 = alpha_dd[b_idx*L*L*L*L*d + gap_start*L3 + start*L2 + gap_end*L1 + end*d + s_A];

    L3 = L*L*L*r;
    L2 = L*L*r;
    L1 = L*r;
    for(int split=start+1; split< gap_start; split++)
    {
        float tmp = s_d[split*L3 + gap_start*L2 + gap_end*L1 + end*r + s_R] + s[start*L*r*4 + split*r*4 + 0*r + s_R];
        float grd1 = exp(tmp + rule_score1 - parent_inside) * parent_grd;
        rule_grd += grd1;
        float grd2 = exp(tmp + rule_score2 - parent_inside2) * parent_grd2;
        rule_grd2 += grd2;
        atomicAdd(s_d + gap_start*L3 + split*L2 + gap_end*L1 + end*r + s_R, grd1+grd2);
        atomicAdd(s + split*L*r*4 + start*r*4 + 0*r + s_R, grd1+grd2);

        tmp = s_d[start*L3 + split*L2 + gap_end*L1 + end*r + s_R]  + s[split*L*r*4 + gap_start*r*4 + 1*r + s_R];
        grd1 = exp(tmp + rule_score1 - parent_inside) * parent_grd;
        rule_grd += grd1;
        grd2 = exp(tmp + rule_score2 - parent_inside2) * parent_grd2;
        rule_grd2 += grd2;
        atomicAdd(s_d + split*L3 + start*L2 + gap_end*L1 + end*r + s_R, grd1+grd2);
        atomicAdd(s + gap_start*L*r*4 + split*r*4 + 1*r + s_R, grd1+grd2);
    }

    for(int split=gap_end+1; split <end; split++){
        float tmp = s_d[start*L3 + gap_start*L2 + split*L1 + end*r + s_R] + s[gap_end*L*r*4 + split*r*4 + 2*r + s_R];
        float grd1 = exp(tmp + rule_score1 - parent_inside) * parent_grd;
        rule_grd += grd1;
        float grd2 = exp(tmp + rule_score2 - parent_inside2) * parent_grd2;
        rule_grd2 += grd2;
        atomicAdd(s_d + gap_start*L3 + start*L2 + split*L1 + end*r + s_R, grd1+grd2);
        atomicAdd(s + split*L*r*4 + gap_end*r*4 + 2*r + s_R , grd1+grd2);

        tmp = s_d[start*L3 + gap_start*L2 + gap_end*L1 + split*r + s_R] + s[split*L*r*4 + end*r*4 + 3*r + s_R];
        grd1 = exp(tmp + rule_score1 - parent_inside) * parent_grd;
        rule_grd += grd1;
        grd2 = exp(tmp + rule_score2 - parent_inside2) * parent_grd2;
        rule_grd2 += grd2;
        atomicAdd(s_d + gap_start*L3 + start*L2 + gap_end*L1 + split*r + s_R, grd1+grd2);
        atomicAdd(s + end*L*r*4 + split*r*4 + 3*r + s_R, grd1+grd2);
    }
    atomicAdd(head_cd_grd + b_idx*d*r + s_R*d + s_A, rule_grd);
    atomicAdd(head_dd_grd + b_idx*d*r + s_R*d + s_A, rule_grd2);
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

      dim3 gridDim3(B*(L-w),  (w-1)*(L-w), r2);
      dim3 blockDim3(r1);
      kernel_forward_d1<<<gridDim3, blockDim3>>>(head_cd1, head_dd1, alpha_l, alpha_r, alpha_cd, alpha_dd, B, L, w, r2, r1);

      if(w>2){
            dim3 gridDim4(B*(L-w),  (w-1)*(L-w), r2);
            dim3 blockDim4(r2);
            kernel_forward_d2<<<gridDim4, blockDim4>>>(head_cd2, head_dd2, alpha_dc, alpha_dd, alpha_cd, B, L, w, r2, r2);
      }
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