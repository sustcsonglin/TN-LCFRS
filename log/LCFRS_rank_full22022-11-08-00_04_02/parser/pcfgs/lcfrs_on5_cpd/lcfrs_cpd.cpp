#include <torch/extension.h>

//void cuda_forward(  float *score,  float *alpha, int B, int L);
//
void cuda_forward(  float *head_c1,   float *head_c2,   float *head_d1,   float *head_d2,
                    float *left_c,   float *right_c,
                    float *left_d,   float *right_d,
                    float *cc,   float *cd,   float *dc,   float *dd,   float *unary,
                  float *alpha,  float *alpha_d, float *alpha_lc, float *alpha_rc, float *alpha_ld, float *alpha_rd,
                  float *alpha_cc, float *alpha_cd, float *alpha_dc, float *alpha_dd,
                  int B, int L, int m, int p, int d, int r1, int r2, int r3, int r4);

void cuda_argmax(float *s_span_c, float *s_span_d, float *alpha_c, float *alpha_d, int B, int L);

void cuda_argmax_on3(float *s_span_c, float *s_span_d, float *alpha_c, float *alpha_d, int B, int L);

void cuda_argmax_on4(float *s_span_c, float *s_span_d, float *alpha_c, float *alpha_d, int B, int L);

void cuda_backward(  float *head_c1,   float *head_c2,   float *head_d1,   float *head_d2,
                   float *left_c,   float *right_c,
                   float *left_d,   float *right_d,
                   float *cc,   float *cd,   float *dc,   float *dd,   float *unary,
                   float *head_c1_grd,   float *head_c2_grd,   float *head_d1_grd,   float *head_d2_grd,
                   float *left_c_grd,   float *right_c_grd,
                   float *left_d_grd,   float *right_d_grd,
                   float *cc_grd,   float *cd_grd,   float *dc_grd,   float *dd_grd,   float *unary_grd,
                   float *alpha,  float *alpha_d, float *alpha_lc, float *alpha_rc, float *alpha_ld, float *alpha_rd,
                   float *alpha_cc, float *alpha_cd, float *alpha_dc, float *alpha_dd,
                   int B, int L, int m, int p, int d, int r1, int r2, int r3, int r4);



//void cuda_mbr(float *alpha, float *alpha_d, int B, int L, int m, int p, int d);


void argmax(torch::Tensor &a, torch::Tensor &b, torch::Tensor &c, torch::Tensor &d, int64_t B, int64_t L){
    cuda_argmax((float *) a.data_ptr(),
    (float *) b.data_ptr(),
    (float *) c.data_ptr(),
    (float *) d.data_ptr(),
    B, L);
}

void argmax_on3(torch::Tensor &a, torch::Tensor &b, torch::Tensor &c, torch::Tensor &d, int64_t B, int64_t L){
    cuda_argmax_on3((float *) a.data_ptr(),
    (float *) b.data_ptr(),
    (float *) c.data_ptr(),
    (float *) d.data_ptr(),
    B, L);
}
void argmax_on4(torch::Tensor &a, torch::Tensor &b, torch::Tensor &c, torch::Tensor &d, int64_t B, int64_t L){
    cuda_argmax_on4((float *) a.data_ptr(),
    (float *) b.data_ptr(),
    (float *) c.data_ptr(),
    (float *) d.data_ptr(),
    B, L);
}



void forward(torch::Tensor &a1, torch::Tensor &a2, torch::Tensor &a3, torch::Tensor &a4, torch::Tensor &a5,
torch::Tensor &a6, torch::Tensor &a7, torch::Tensor &a8, torch::Tensor &a9, torch::Tensor &a10, torch::Tensor &a11,
torch::Tensor &a12, torch::Tensor &a13, torch::Tensor &a14, torch::Tensor &a15, torch::Tensor &a16, torch::Tensor &a17,
torch::Tensor &a18, torch::Tensor &a19, torch::Tensor &a20, torch::Tensor &a21, torch::Tensor &a22, torch::Tensor &a23,
int64_t B, int64_t L, int64_t m, int64_t p, int64_t d, int64_t r1, int64_t r2, int64_t r3, int64_t r4) {
    cuda_forward((  float *)a1.data_ptr(),
     ( float *) a2.data_ptr(),
      ( float *) a3.data_ptr(),
      (  float *) a4.data_ptr(),
     (  float *)a5.data_ptr(),
     (  float *)a6.data_ptr(),
     (  float *)a7.data_ptr(),
     (  float *)a8.data_ptr(),
     (  float *)a9.data_ptr(),
     (  float *)a10.data_ptr(),
     (  float *)a11.data_ptr(),
     (  float *)a12.data_ptr(),
     (  float *)a13.data_ptr(),
     (float *)a14.data_ptr(),
     (float *)a15.data_ptr(),
     (float *)a16.data_ptr(),
     (float *)a17.data_ptr(),
     (float *)a18.data_ptr(),
     (float *)a19.data_ptr(),
     (float *)a20.data_ptr(),
     (float *)a21.data_ptr(),
     (float *)a22.data_ptr(),
     (float *)a23.data_ptr(),
     B, L, m, p, d, r1,r2,r3,r4);
}


void backward(torch::Tensor &a1, torch::Tensor &a2, torch::Tensor &a3, torch::Tensor &a4, torch::Tensor &a5,
torch::Tensor &a6, torch::Tensor &a7, torch::Tensor &a8, torch::Tensor &a9, torch::Tensor &a10, torch::Tensor &a11,
torch::Tensor &a12, torch::Tensor &a13, torch::Tensor &a14, torch::Tensor &a15, torch::Tensor &a16, torch::Tensor &a17,
torch::Tensor &a18, torch::Tensor &a19, torch::Tensor &a20, torch::Tensor &a21, torch::Tensor &a22, torch::Tensor &a23,
torch::Tensor &a24, torch::Tensor &a25, torch::Tensor &a26, torch::Tensor &a27, torch::Tensor &a28, torch::Tensor &a29,
torch::Tensor &a30, torch::Tensor &a31, torch::Tensor &a32, torch::Tensor &a33, torch::Tensor &a34, torch::Tensor &a35, torch::Tensor &a36,
int64_t B, int64_t L, int64_t m, int64_t p, int64_t d, int64_t r1, int64_t r2, int64_t r3, int64_t r4) {
    cuda_backward((  float *)a1.data_ptr(),
     ( float *) a2.data_ptr(),
      ( float *) a3.data_ptr(),
      (  float *) a4.data_ptr(),
     (  float *)a5.data_ptr(),
     (  float *)a6.data_ptr(),
     (  float *)a7.data_ptr(),
     (  float *)a8.data_ptr(),
     (  float *)a9.data_ptr(),
     (  float *)a10.data_ptr(),
     (  float *)a11.data_ptr(),
     (  float *)a12.data_ptr(),
     (  float *)a13.data_ptr(),
     (float *)a14.data_ptr(),
     (float *)a15.data_ptr(),
     (float *)a16.data_ptr(),
     (float *)a17.data_ptr(),
     (float *)a18.data_ptr(),
     (float *)a19.data_ptr(),
     (float *)a20.data_ptr(),
     (float *)a21.data_ptr(),
     (float *)a22.data_ptr(),
     (float *)a23.data_ptr(),
      (float *)a24.data_ptr(),
      (float *)a25.data_ptr(),
      (float *)a26.data_ptr(),
      (float *)a27.data_ptr(),
      (float *)a28.data_ptr(),
      (float *)a29.data_ptr(),
      (float *)a30.data_ptr(),
      (float *)a31.data_ptr(),
      (float *)a32.data_ptr(),
      (float *)a33.data_ptr(),
      (float *)a34.data_ptr(),
      (float *)a35.data_ptr(),
      (float *)a36.data_ptr(),
     B, L, m, p, d, r1,r2,r3,r4);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "timex forward");
    m.def("backward", &backward, "timex backward");
    m.def("argmax", &argmax, "timex argmax");
    m.def("argmax_on3", &argmax_on3, "timex argmax_on3");
    m.def("argmax_on4", &argmax_on4, "timex argmax_on3");
}

TORCH_LIBRARY(lcfrs_cpd, m) {
    m.def("forward", forward);
    m.def("argmax", argmax);
}

