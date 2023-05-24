#include <torch/extension.h>

void cuda_forward(float *head_cl1,  float *head_cr1,
                   float *head_cc1,  float *head_dc1,
                   float *head_cl2,  float *head_cr2,
                   float *head_cc2,  float *head_dc2,
                   float *head_cd1, float *head_cd2,
                   float *head_dd1, float *head_dd2,
                   float *alpha_l, float *alpha_rc, float *alpha_ld, float *alpha_rd,
                   float *alpha_cc, float *alpha_cd, float *alpha_dc, float *alpha_dd,
                   int B, int L, int r1, int r2);


void forward(torch::Tensor &a1, torch::Tensor &a2, torch::Tensor &a3, torch::Tensor &a4, torch::Tensor &a5,
torch::Tensor &a6, torch::Tensor &a7, torch::Tensor &a8, torch::Tensor &a9, torch::Tensor &a10, torch::Tensor &a11,
torch::Tensor &a12, torch::Tensor &a13, torch::Tensor &a14, torch::Tensor &a15, torch::Tensor &a16, torch::Tensor &a17,
torch::Tensor &a18, torch::Tensor &a19, torch::Tensor &a20,
int64_t B, int64_t L, int64_t r1, int64_t r2) {
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
     B, L, r1, r2);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "fasdas");
}

TORCH_LIBRARY(Ysl, m) {
    m.def("forward", forward);
}