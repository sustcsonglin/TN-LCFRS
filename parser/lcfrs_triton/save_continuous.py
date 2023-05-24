import pdb
import torch
import triton
import triton.language as tl




@triton.jit
def kernel_log_and_diagonal_copy(
    out,
    normalizer,
    alpha_c,    
    stride_alpha_c1, stride_alpha_c2, stride_alpha_c3, 
    stride_out0, stride_out1,
    stride_normalizer0, stride_normalizer1,
    batch, r,
    BLOCK_R1: tl.constexpr,
    w
):    

    b_idx = tl.program_id(0) 
    if b_idx >= batch:
        return 
    
    start = tl.program_id(1)    
    mask = tl.arange(0, BLOCK_R1) < r

    x = tl.load(out + b_idx * stride_out0 + start * stride_out1 + tl.arange(0, BLOCK_R1), mask=mask, other=1)
    x_normalizer = tl.load(normalizer + b_idx * stride_normalizer0 + start)

    out_log = tl.log(x + 1e-9)
    out_log = out_log + x_normalizer
    tl.store(alpha_c +  b_idx * stride_alpha_c1 + start * stride_alpha_c2 + (start+w) * stride_alpha_c3 +  tl.arange(0,  BLOCK_R1) , out_log, mask=mask)


@triton.jit
def _bwd_log_and_diagonal_copy(
    out, out_grad,
    alpha_c,    
    stride_alpha_c1, stride_alpha_c2, stride_alpha_c3, 
    stride_out0, stride_out1, 
 
    batch, r,
    BLOCK_R1: tl.constexpr,
    w
):
    
    b_idx = tl.program_id(0) 
    if b_idx >= batch:
        return 


    mask = tl.arange(0, BLOCK_R1) < r
    start = tl.program_id(1)    
    x = tl.load(out + b_idx * stride_out0 + start * stride_out1 + tl.arange(0, BLOCK_R1), mask=mask, other=1)
    out_log = 1/(x + 1e-9)    




    do = tl.load(alpha_c +  b_idx * stride_alpha_c1 + (start+w) * stride_alpha_c2 + (start) * stride_alpha_c3 +  tl.arange(0,  BLOCK_R1), mask=mask, other=0)

    do *= out_log

    tl.store(out_grad + b_idx * stride_out0 + start * stride_out1 + tl.arange(0, BLOCK_R1), do, mask=mask)




class DIAGONAL_COPY_AND_LOG(torch.autograd.Function):
    @staticmethod
    def forward(ctx, out, normalizer, alpha_c):
        b, n = out.shape[0], out.shape[1] 
        N = alpha_c.shape[1]
        w = N - n 
        r = int(alpha_c.shape[-1])           

        batch = triton.next_power_of_2(b)

        num_warps = 4
        R = triton.next_power_of_2(r)

        if R >= 2048:
            num_warps = 8
        if R >= 4096:
            num_warps = 16

        kernel_log_and_diagonal_copy[batch, n](out, normalizer,
                                     alpha_c,             
                                     alpha_c.stride(0), alpha_c.stride(1), alpha_c.stride(2), 
                                     out.stride(0), out.stride(1),
                                     normalizer.stride(0), normalizer.stride(1),
                                     b, r, 
                                     BLOCK_R1=R,
                                     w=w,
                                     num_warps=num_warps
                                     )
        # pdb.set_trace()
        ctx.save_for_backward(out, alpha_c)
        return alpha_c
        
    @staticmethod
    def backward(ctx, do):
        out, alpha_c = ctx.saved_tensors
        b, n = out.shape[0], out.shape[1]  
        N = alpha_c.shape[1]
        w = N - n 
        r = alpha_c.shape[-1]           
        out_grad = out.new_zeros(*out.shape)

        batch = triton.next_power_of_2(b)
        R = triton.next_power_of_2(r)                        

        num_warps = 4


        if R >= 2048:
            num_warps = 8
        if R >= 4096:
            num_warps = 16

        _bwd_log_and_diagonal_copy[batch, n](
            out, out_grad, alpha_c,
            alpha_c.stride(0), alpha_c.stride(1), alpha_c.stride(2), 
            out.stride(0), out.stride(1), b, r,
            BLOCK_R1=R,
            w=w,
            num_warps=num_warps               
        )

    
        return out_grad,  None, alpha_c



_save_continous = DIAGONAL_COPY_AND_LOG.apply
