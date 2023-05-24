#include <torch/extension.h>

//void cuda_forward( float *score,  float *alpha, int B, int L);
//
void cuda_forward( float *binary,  float *binary_close,  float *binary_dc,  float *binary_d,
 float *alpha, float *alpha_d, int B, int L, int m);

void cuda_backward( float *binary,  float *binary_close,  float *binary_dc,  float *binary_d,
 float *binary_grd,  float *binary_close_grd, float *binary_dc_grd, float *binary_d_grd,
 float *alpha, float *alpha_d, int B, int L, int m);

void cuda_forward_argmax( float *binary,  float *binary_close,  float *binary_dc,  float *binary_d,
float *alpha, float *alpha_d, int B, int L, int m);

//void cuda_mbr(float *alpha, float *alpha_d, int B, int L, int m, int p, int d);


//pcfg
void forward(torch::Tensor &binary, torch::Tensor &binary_close, torch::Tensor &binary_dc, torch::Tensor &binary_d, torch::Tensor &alpha, torch::Tensor &alpha_d, int64_t B, int64_t L, int64_t m) {
    cuda_forward(( float *)binary.data_ptr(),
     ( float *) binary_close.data_ptr(),
      ( float *) binary_dc.data_ptr(),
      ( float *) binary_d.data_ptr(),
     (float *)alpha.data_ptr(),
     (float *)alpha_d.data_ptr(),
     B, L, m);
}

void argmax(torch::Tensor &binary, torch::Tensor &binary_close, torch::Tensor &binary_dc, torch::Tensor &binary_d,  torch::Tensor &alpha, torch::Tensor &alpha_d, int64_t B, int64_t L, int64_t m) {
    cuda_forward_argmax(( float *)binary.data_ptr(),
     ( float *) binary_close.data_ptr(),
      ( float *) binary_dc.data_ptr(),
      ( float *) binary_d.data_ptr(),
     (float *)alpha.data_ptr(),
     (float *)alpha_d.data_ptr(),
     B, L, m);
}


void backward(torch::Tensor &binary, torch::Tensor &binary_close, torch::Tensor &binary_dc, torch::Tensor &binary_d,
  torch::Tensor &binary_grd, torch::Tensor &binary_close_grd, torch::Tensor &binary_dc_grd, torch::Tensor &binary_d_grd,
  torch::Tensor &alpha, torch::Tensor &alpha_d, int64_t B, int64_t L, int64_t m) {
   cuda_backward(( float *)binary.data_ptr(),
     ( float *) binary_close.data_ptr(),
      ( float *) binary_dc.data_ptr(),
      ( float *) binary_d.data_ptr(),
     (float *) binary_grd.data_ptr(),
      (float *) binary_close_grd.data_ptr(),
       (float *) binary_dc_grd.data_ptr(),
       (float *) binary_d_grd.data_ptr(),
      (float *)alpha.data_ptr(),
      (float *)alpha_d.data_ptr(),
     B, L, m);
}

//
//void mbr(torch::Tensor &alpha, torch::Tensor &alpha_d, int64_t B, int64_t L, int64_t m, int64_t p, int64_t d){
//cuda_mbr(      (float *)alpha.data_ptr(),
//      (float *)alpha_d.data_ptr(),
//     B, L, m, p, d);
//}
//


//void forward(torch::Tensor &s_c,  torch::Tensor &s_d, torch::Tensor &s_g,   torch::Tensor &alpha_c, torch::Tensor &alpha_d, int64_t B, int64_t L) {
//    cuda_forward(( float *)s_c.data_ptr(),
//( float *)s_d.data_ptr(),
//( float *)s_g.data_ptr(),
//            (float *)alpha_c.data_ptr(),
//           (float *)alpha_d.data_ptr(),
//           B, L);
//}
//





//

//void backward(torch::Tensor &w,  torch::Tensor &k,  torch::Tensor &gwk, torch::Tensor &gw, torch::Tensor &gk, int64_t B, int64_t C, int64_t T) {
//    cuda_backward(( float *)w.data_ptr(), ( float *)k.data_ptr(), ( float *)gwk.data_ptr(), (float *)gw.data_ptr(), (float *)gk.data_ptr(), B, C, T);
//}
//C
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "timex forward");
    m.def("backward", &backward, "timex backward");
    m.def("argmax", &argmax, "timex argmax");
}

TORCH_LIBRARY(on5_nopt, m) {
    m.def("forward", forward);
    m.def("backward", backward);
    m.def("argmax", argmax);
}

