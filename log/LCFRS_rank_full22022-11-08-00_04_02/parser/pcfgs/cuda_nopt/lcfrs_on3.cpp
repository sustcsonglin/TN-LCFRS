#include <torch/extension.h>

void cuda_forward(const float *binary, const float *binary_d, const float *binary_c,  const float *unary, float *alpha, float *alpha_d, int B, int L, int m, int p, int d);
void cuda_argmax(const float *binary, const float *binary_d, const float *binary_c,  const float *unary, float *alpha, float *alpha_d, int B, int L, int m, int p, int d);
void cuda_backward(const float *binary, const float *binary_d, const float *binary_c,  const float *unary, float *a1, float *a2, float *a3, float *a4, float *alpha, float *alpha_d, int B, int L, int m, int p, int d);

void forward(torch::Tensor &binary, torch::Tensor &binary_d, torch::Tensor &binary_c, torch::Tensor &unary,  torch::Tensor &alpha, torch::Tensor &alpha_d, int64_t B, int64_t L, int64_t m, int64_t p, int64_t d){
    cuda_forward((const float *)binary.data_ptr(),
    (const float *)binary_d.data_ptr(),
    (const float *)binary_c.data_ptr(),
    (const float *)unary.data_ptr(),
     (float *)alpha.data_ptr(),
     (float *)alpha_d.data_ptr(),
      B, L, m, p, d);
}

void argmax(torch::Tensor &binary, torch::Tensor &binary_d, torch::Tensor &binary_c, torch::Tensor &unary,  torch::Tensor &alpha, torch::Tensor &alpha_d, int64_t B, int64_t L, int64_t m, int64_t p, int64_t d){
    cuda_forward_argmax((const float *)binary.data_ptr(),
    (const float *)binary_d.data_ptr(),
    (const float *)binary_c.data_ptr(),
    (const float *)unary.data_ptr(),
     (float *)alpha.data_ptr(),
     (float *)alpha_d.data_ptr(),
      B, L, m, p, d);
}


void backward(torch::Tensor &binary, torch::Tensor &binary_d, torch::Tensor &binary_c, torch::Tensor &unary, torch::Tensor &binary_g, torch::Tensor &binary_d_g, torch::Tensor &binary_c_g, torch::Tensor &unary_g,  torch::Tensor &alpha, torch::Tensor &alpha_d, int64_t B, int64_t L, int64_t m, int64_t p, int64_t d) {
    cuda_backward((const float *)binary.data_ptr(),
    (const float *)binary_d.data_ptr(),
    (const float *)binary_c.data_ptr(),
    (const float *)unary.data_ptr(),
    ( float *)binary_g.data_ptr(),
    ( float *)binary_d_g.data_ptr(),
    ( float *)binary_c_g.data_ptr(),
    ( float *)unary_g.data_ptr(),
     (float *)alpha.data_ptr(),
     (float *)alpha_d.data_ptr(),
     B, L, m, p, d);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "timex forward");
    m.def("backward", &backward, "timex backward");
    m.def("argmax", &argmax, "timex argmax");

}

TORCH_LIBRARY(timex, m) {
    m.def("forward", forward);
    m.def("backward", backward);
    m.def("argmax", argmax);
}


