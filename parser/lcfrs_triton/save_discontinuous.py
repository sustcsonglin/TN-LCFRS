import pdb
import statistics
import torch
import triton
import triton.language as tl



    
@triton.jit
def _save_into_alpha_d(
    alpha_d,
    x,
    stride_alpha_d1, stride_alpha_d2, stride_alpha_d3, stride_alpha_d4, stride_alpha_d5,
    stride_merge1, stride_merge2, stride_merge3,
    B, L, w, r, 
    BLOCK_RD: tl.constexpr
):       
    b_idx = tl.program_id(0)
    
    if b_idx >= B:
        return 

    span_length_left = tl.program_id(1) + 1
    tid = tl.program_id(2)
    start = 0

    mask = tl.arange(0, BLOCK_RD) < r

    to_save = tl.load(x + b_idx * stride_merge1 + tl.program_id(1) * stride_merge2 + tid * stride_merge3 + tl.arange(0, BLOCK_RD),
                      mask=mask, other=0)



    while tid >= (L-w-start):
        tid -= (L-w-start)
        start += 1 

    # start = start_b_idxdr 
    gap_start = start + span_length_left
    gap_end = gap_start + (tid + 1)
    end = gap_end + (w - span_length_left)
    
    tl.store(alpha_d + b_idx * stride_alpha_d1 + start * stride_alpha_d2 +  gap_start * stride_alpha_d3 + gap_end *        stride_alpha_d4 + end *             stride_alpha_d5 + tl.arange(0, BLOCK_RD), 
             to_save, mask=mask 
             )
    


@triton.jit
def _bwd_save_into_alpha_d(
    alpha_d,
    x,
    stride_alpha_d1, stride_alpha_d2, stride_alpha_d3, stride_alpha_d4, stride_alpha_d5,
    stride_merge1, stride_merge2, stride_merge3,
    B, L, w, r,
    BLOCK_RD: tl.constexpr
):   
    
    b_idx = tl.program_id(0)

    if b_idx >= B:
        return 

    span_length_left = tl.program_id(1) + 1
    tid = tl.program_id(2)
    start = 0

    to_save_ptr = x + b_idx * stride_merge1 + tl.program_id(1) * stride_merge2 + tid * stride_merge3 + tl.arange(0, BLOCK_RD)
    mask = tl.arange(0, BLOCK_RD) < r 

    while tid >= (L-w-start):
        tid -= (L-w-start)
        start += 1 

    # start = start_b_idxdr 
    gap_start = start + span_length_left
    gap_end = gap_start + (tid + 1)
    end = gap_end + (w - span_length_left)
    
    save = tl.load(alpha_d + b_idx * stride_alpha_d1 + gap_start * stride_alpha_d2 +  start * stride_alpha_d3 + gap_end *             stride_alpha_d4 + end *  stride_alpha_d5 + tl.arange(0, BLOCK_RD),mask=mask,other=0)

    tl.store(to_save_ptr,          save, mask=mask)
    




class SAVE_ALPHA_D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha_d, dimension_info):
        B = alpha_d.shape[0]
        N = alpha_d.shape[1] - 1
        w = int(dimension_info[0])
        n = N - w 
        r1 = int(dimension_info[1])
        r2 = int(dimension_info[2])
        r3 = int(dimension_info[3])
        r4 = int(dimension_info[4])
        grid2 = (triton.next_power_of_2(B), (w-1), int((N-w)*(N-w+1)/2))            
        _save_into_alpha_d[grid2](                    
                    alpha_d, x,                
                    alpha_d.stride(0), alpha_d.stride(1), alpha_d.stride(2), alpha_d.stride(3), alpha_d.stride(4),
                    x.stride(0), x.stride(1), x.stride(2), 
                    B, N, w, x.shape[-1],
                    BLOCK_RD= triton.next_power_of_2(x.shape[-1])
                )
        ctx.save_for_backward(alpha_d, dimension_info)
        return alpha_d
        
    @staticmethod
    def backward(ctx, do):
        alpha_d, dimension_info = ctx.saved_tensors
        B = alpha_d.shape[0]
        N = alpha_d.shape[1] - 1
        w = int(dimension_info[0])
        n = N - w 
        r1 = int(dimension_info[1])
        r2 = int(dimension_info[2])
        r3 = int(dimension_info[3])
        r4 = int(dimension_info[4])
        grid2 = (triton.next_power_of_2(B), (w-1), int((N-w)*(N-w+1)/2))            
        x = alpha_d.new_zeros(B, (w-1), int((N-w)*(N-w+1)/2), r2+r4)
                        

        _bwd_save_into_alpha_d[grid2](                    
            alpha_d, x,                
            alpha_d.stride(0), alpha_d.stride(1), alpha_d.stride(2), alpha_d.stride(3), alpha_d.stride(4),
            x.stride(0), x.stride(1), x.stride(2), 
            B, N, w,  x.shape[-1],
            BLOCK_RD= triton.next_power_of_2(x.shape[-1])
        )

        return x, alpha_d, None

        

        
                
_save_discontinuous = SAVE_ALPHA_D.apply


