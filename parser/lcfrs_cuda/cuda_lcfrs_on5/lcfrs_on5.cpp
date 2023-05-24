#include <torch/extension.h>

//void cuda_forward(const float *score,  float *alpha, int B, int L);
//
void cuda_forward(const float *binary, const float *binary_close, const float *binary_dc, const float *binary_d,  const float *unary,
 float *alpha, float *alpha_d, int B, int L, int m, int p, int d);

void cuda_backward(const float *binary, const float *binary_close, const float *binary_dc, const float *binary_d,  const float *unary,
 float *binary_grd,  float *binary_close_grd, float *binary_dc_grd, float *binary_d_grd, float *unary_grd,
 float *alpha, float *alpha_d, int B, int L, int m, int p, int d);

void cuda_forward_argmax(const float *binary, const float *binary_close, const float *binary_dc, const float *binary_d,  const float *unary,
float *alpha, float *alpha_d, int B, int L, int m, int p, int d);

//void cuda_mbr(float *alpha, float *alpha_d, int B, int L, int m, int p, int d);


//pcfg
void forward(torch::Tensor &binary, torch::Tensor &binary_close, torch::Tensor &binary_dc, torch::Tensor &binary_d, torch::Tensor &unary, torch::Tensor &alpha, torch::Tensor &alpha_d, int64_t B, int64_t L, int64_t m, int64_t p, int64_t d) {
    cuda_forward((const float *)binary.data_ptr(),
     (const float *) binary_close.data_ptr(),
      (const float *) binary_dc.data_ptr(),
      (const float *) binary_d.data_ptr(),
     (const float *)unary.data_ptr(),
     (float *)alpha.data_ptr(),
     (float *)alpha_d.data_ptr(),
     B, L, m, p, d);
}

void argmax(torch::Tensor &binary, torch::Tensor &binary_close, torch::Tensor &binary_dc, torch::Tensor &binary_d, torch::Tensor &unary, torch::Tensor &alpha, torch::Tensor &alpha_d, int64_t B, int64_t L, int64_t m, int64_t p, int64_t d) {
    cuda_forward_argmax((const float *)binary.data_ptr(),
     (const float *) binary_close.data_ptr(),
      (const float *) binary_dc.data_ptr(),
      (const float *) binary_d.data_ptr(),
     (const float *)unary.data_ptr(),
     (float *)alpha.data_ptr(),
     (float *)alpha_d.data_ptr(),
     B, L, m, p, d);
}


void backward(torch::Tensor &binary, torch::Tensor &binary_close, torch::Tensor &binary_dc, torch::Tensor &binary_d, torch::Tensor &unary,
  torch::Tensor &binary_grd, torch::Tensor &binary_close_grd, torch::Tensor &binary_dc_grd, torch::Tensor &binary_d_grd, torch::Tensor &unary_grd,
  torch::Tensor &alpha, torch::Tensor &alpha_d, int64_t B, int64_t L, int64_t m, int64_t p, int64_t d) {
   cuda_backward((const float *)binary.data_ptr(),
     (const float *) binary_close.data_ptr(),
      (const float *) binary_dc.data_ptr(),
      (const float *) binary_d.data_ptr(),
     (const float *)unary.data_ptr(),
     (float *) binary_grd.data_ptr(),
      (float *) binary_close_grd.data_ptr(),
       (float *) binary_dc_grd.data_ptr(),
       (float *) binary_d_grd.data_ptr(),
        (float *) unary_grd.data_ptr(),
      (float *)alpha.data_ptr(),
      (float *)alpha_d.data_ptr(),
     B, L, m, p, d);
}

//
//void mbr(torch::Tensor &alpha, torch::Tensor &alpha_d, int64_t B, int64_t L, int64_t m, int64_t p, int64_t d){
//cuda_mbr(      (float *)alpha.data_ptr(),
//      (float *)alpha_d.data_ptr(),
//     B, L, m, p, d);
//}
//


//void forward(torch::Tensor &s_c,  torch::Tensor &s_d, torch::Tensor &s_g,   torch::Tensor &alpha_c, torch::Tensor &alpha_d, int64_t B, int64_t L) {
//    cuda_forward((const float *)s_c.data_ptr(),
//(const float *)s_d.data_ptr(),
//(const float *)s_g.data_ptr(),
//            (float *)alpha_c.data_ptr(),
//           (float *)alpha_d.data_ptr(),
//           B, L);
//}
//





//

//void backward(torch::Tensor &w, const torch::Tensor &k, const torch::Tensor &gwk, torch::Tensor &gw, torch::Tensor &gk, int64_t B, int64_t C, int64_t T) {
//    cuda_backward((const float *)w.data_ptr(), (const float *)k.data_ptr(), (const float *)gwk.data_ptr(), (float *)gw.data_ptr(), (float *)gk.data_ptr(), B, C, T);
//}
//C
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "timex forward");
    m.def("backward", &backward, "timex backward");
    m.def("argmax", &argmax, "timex argmax");
}

TORCH_LIBRARY(sdadadsa, m) {
    m.def("forward", forward);
    m.def("backward", backward);
    m.def("argmax", argmax);
}


