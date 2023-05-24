#include <torch/extension.h>

void cuda_forward(float *head_cl1,  float *head_cr1,
                  float *head_dl1,  float *head_dr1,

                   float *head_cc1,  float *head_dc1,
                   float *head_cl2,  float *head_cr2,
                   float *head_dl2,  float *head_dr2,

                   float *head_cc2,  float *head_dc2,
                   float *head_cd1, float *head_cd2,
                   float *head_dd1, float *head_dd2,
                   float *root_c, float *root_d,
                   float *alpha_l, float *alpha_r,
                   float *alpha_ld, float *alpha_rd,
                   float *alpha_cc, float *alpha_cd, float *alpha_dc, float *alpha_dd,
                   float *a2, float *a3,

                   int B, int L, int r1, int r2, int r3, int r4);


void cuda_backward(float *head_cl1,  float *head_cr1,
                   float *head_dl1,  float *head_dr1,

                   float *head_cc1,  float *head_dc1,
                   float *head_cl2,  float *head_cr2,
                   float *head_dl2,  float *head_dr2,

                   float *head_cc2,  float *head_dc2,
                   float *head_cd1, float *head_cd2,
                   float *head_dd1, float *head_dd2,
                   float *root_c, float *root_d,

                   float *head_cl1_grd,  float *head_cr1_grd,
                   float *head_dl1_grd,  float *head_dr1_grd,
                   float *head_cc1_grd,  float *head_dc1_grd,
                   float *head_cl2_grd,  float *head_cr2_grd,
                   float *head_dl2_grd,  float *head_dr2_grd,
                   float *head_cc2_grd,  float *head_dc2_grd,
                   float *head_cd1_grd, float *head_cd2_grd,
                   float *head_dd1_grd, float *head_dd2_grd,
                   float *root_c_grd, float *root_d_grd,

                   float *alpha_l, float *alpha_r,  float *alpha_ld, float *alpha_rd,
                   float *alpha_cc, float *alpha_cd, float *alpha_dc, float *alpha_dd, float *a1, float *a2,
                   int B, int L, int r1, int r2, int r3, int r4);


void forward(torch::Tensor &a1, torch::Tensor &a2, torch::Tensor &a3, torch::Tensor &a4, torch::Tensor &a5,
torch::Tensor &a6, torch::Tensor &a7, torch::Tensor &a8, torch::Tensor &a9, torch::Tensor &a10, torch::Tensor &a11,
torch::Tensor &a12, torch::Tensor &a13, torch::Tensor &a14, torch::Tensor &a15, torch::Tensor &a16, torch::Tensor &a17,
torch::Tensor &a18, torch::Tensor &a19, torch::Tensor &a20, torch::Tensor &a21, torch::Tensor &a22,
torch::Tensor &a23, torch::Tensor &a24, torch::Tensor &a25, torch::Tensor &a26, torch::Tensor &a27,
torch::Tensor &a28,
int64_t B, int64_t L, int64_t r1, int64_t r2, int64_t r3, int64_t r4) {
    cuda_forward((float *)a1.data_ptr(),
     (float *)a2.data_ptr(),
     (float *)a3.data_ptr(),
     (float *)a4.data_ptr(),
     (float *)a5.data_ptr(),
     (float *)a6.data_ptr(),
     (float *)a7.data_ptr(),
     (float *)a8.data_ptr(),
     (float *)a9.data_ptr(),
     (float *)a10.data_ptr(),
     (float *)a11.data_ptr(),
     (float *)a12.data_ptr(),
     (float *)a13.data_ptr(),
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
     B, L, r1, r2, r3, r4);
}



void backward(torch::Tensor &a1, torch::Tensor &a2, torch::Tensor &a3, torch::Tensor &a4, torch::Tensor &a5,
torch::Tensor &a6, torch::Tensor &a7, torch::Tensor &a8, torch::Tensor &a9, torch::Tensor &a10, torch::Tensor &a11,
torch::Tensor &a12, torch::Tensor &a13, torch::Tensor &a14, torch::Tensor &a15, torch::Tensor &a16, torch::Tensor &a17,
torch::Tensor &a18, torch::Tensor &a19, torch::Tensor &a20,
torch::Tensor &a21, torch::Tensor &a22, torch::Tensor &a23, torch::Tensor &a24,
torch::Tensor &a25, torch::Tensor &a26, torch::Tensor &a27, torch::Tensor &a28,
torch::Tensor &a29, torch::Tensor &a30, torch::Tensor &a31, torch::Tensor &a32,
torch::Tensor &a33, torch::Tensor &a34, torch::Tensor &a35, torch::Tensor &a36,
torch::Tensor &a37, torch::Tensor &a38, torch::Tensor &a39, torch::Tensor &a40,
torch::Tensor &a41, torch::Tensor &a42, torch::Tensor &a43, torch::Tensor &a44,
torch::Tensor &a45, torch::Tensor &a46,
int64_t B, int64_t L, int64_t r1, int64_t r2, int64_t r3, int64_t r4) {
    cuda_backward((float *)a1.data_ptr(),
     (float *)a2.data_ptr(),
     (float *)a3.data_ptr(),
     (float *)a4.data_ptr(),
     (float *)a5.data_ptr(),
     (float *)a6.data_ptr(),
     (float *)a7.data_ptr(),
     (float *)a8.data_ptr(),
     (float *)a9.data_ptr(),
     (float *)a10.data_ptr(),
     (float *)a11.data_ptr(),
     (float *)a12.data_ptr(),
     (float *)a13.data_ptr(),
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
     (float *)a37.data_ptr(),
     (float *)a38.data_ptr(),
     (float *)a39.data_ptr(),
     (float *)a40.data_ptr(),
     (float *)a41.data_ptr(),
     (float *)a42.data_ptr(),
     (float *)a43.data_ptr(),
     (float *)a44.data_ptr(),
     (float *)a45.data_ptr(),
     (float *)a46.data_ptr(),
     B, L, r1, r2, r3, r4);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "forward");
    m.def("backward", &backward, "backward");
}

TORCH_LIBRARY(adsdasd21213asxd, m) {
    m.def("forward", forward);
    m.def("backward", backward);
}