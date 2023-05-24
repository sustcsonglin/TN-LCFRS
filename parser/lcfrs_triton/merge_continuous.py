import pdb
import statistics
import torch
import triton
import triton.language as tl


@triton.jit
def logaddexp(a, b):
    tmp = a - b
    return tl.where(tmp > 0, tl.log(tl.exp(b - a) + 1) + a, tl.log(tl.exp(a-b) + 1) + b)


@triton.jit
def _kernel_inside_merge_continuous(
            alpha_c, alpha_d, tmp_merge, tmp_merge_normalized, tmp_normalizer,
            stride_alpha_c1, stride_alpha_c2, stride_alpha_c3, 
            stride_alpha_d1, stride_alpha_d2, stride_alpha_d3, stride_alpha_d4, stride_alpha_d5,
            stride_tmp_merge1, stride_tmp_merge2, 
            stride_tmp_merge_normalized1,  
            r1, r2, r3, r4, b, n, w, L,
            BLOCK_R1: tl.constexpr,
            BLOCK_R2: tl.constexpr,
            ):    
    
    b_idx = tl.program_id(0)
    start = tl.program_id(1)
    end = start + w

    if b_idx >= b:
        return
        
    offset_r = tl.arange(0, BLOCK_R1)
    
    # [i, k], [k, j] -> [i, j]
    l_ptr = alpha_c + b_idx * stride_alpha_c1 + start * stride_alpha_c2 + (start+1) * stride_alpha_c3 +  offset_r    
    r_ptr = alpha_c + b_idx * stride_alpha_c1 + (start +1) * stride_alpha_c2 +  (end) * stride_alpha_c3 + r1 + offset_r 
    acc1 = tl.zeros((BLOCK_R1,), dtype=tl.float32) - 1e9        
    
    mask= tl.arange(0, BLOCK_R1) < r1
    mask2= tl.arange(0, BLOCK_R2) < r2
    
    for _ in range(0, w-1):
        left = tl.load(l_ptr,mask=mask, other=-1e9)
        right = tl.load(r_ptr,mask=mask,other=-1e9)
        merge = left + right
        acc1 = logaddexp(acc1, merge)
        l_ptr += stride_alpha_c3
        r_ptr += stride_alpha_c2

    acc2 = tl.zeros((BLOCK_R2,), dtype=tl.float32) - 1e9    

    # [m, n], [i, m, n, j] -> [i, j]   i:=start, j:=end, m:=gap start, n:= gapend. 
    #  corresponding rank within the last dim of alpha_c. [2r1, 2r1+r2]     
    for gap_start in range(start+1, end-1):
        for gap_end in range(gap_start+1, end):
            ptr_c = alpha_c + b_idx * stride_alpha_c1 + gap_start * stride_alpha_c2 + gap_end * stride_alpha_c3 + 2*r1 + tl.arange(0, BLOCK_R2)            
            ptr_d = alpha_d + b_idx * stride_alpha_d1 + start * stride_alpha_d2 + gap_start * stride_alpha_d3 + gap_end * stride_alpha_d4 + (end) * stride_alpha_d5 + tl.arange(0, BLOCK_R2)
            cont = tl.load(ptr_c, mask=mask2, other=-1e9)
            disco = tl.load(ptr_d, mask=mask2, other=-1e9)            
            merge = cont + disco
            acc2 = logaddexp(acc2, merge)
    
    tl.store(tmp_merge + b_idx * stride_tmp_merge1 + start * stride_tmp_merge2 + tl.arange(0, BLOCK_R1), acc1, mask=mask)
    tl.store(tmp_merge + b_idx * stride_tmp_merge1 + start * stride_tmp_merge2 + r1 + tl.arange(0, BLOCK_R2), acc2,mask=mask2)


    acc1_max = tl.max(acc1, 0)
    acc2_max = tl.max(acc2, 0)
    acc_max = tl.maximum(acc1_max, acc2_max)

    tl.store(tmp_normalizer + b_idx * stride_tmp_merge_normalized1 + start, acc_max)

    out1 = tl.exp(acc1 - acc_max)
    tl.store(tmp_merge_normalized + b_idx * stride_tmp_merge1 + start * stride_tmp_merge2 + tl.arange(0, BLOCK_R1), out1, mask=mask)
    
    out2 = tl.exp(acc2 - acc_max)
    tl.store(tmp_merge_normalized + b_idx * stride_tmp_merge1 + start * stride_tmp_merge2 + r1 + tl.arange(0, BLOCK_R2), out2, mask=mask2)





@triton.jit
def _kernel_bwd_merge_continuous(
            alpha_c, alpha_d, tmp_merge, tmp_merge_normalized, 
            tmp_merge_grad,
            stride_alpha_c1, stride_alpha_c2, stride_alpha_c3, 
            stride_alpha_d1, stride_alpha_d2, stride_alpha_d3, stride_alpha_d4, stride_alpha_d5,
            stride_tmp_merge1, stride_tmp_merge2, 
            r1, r2, r3, r4, b, n, w, L,
            BLOCK_R1: tl.constexpr,
            BLOCK_R2: tl.constexpr,
            ):    
    
    b_idx = tl.program_id(0)
    start = tl.program_id(1)
    end = start + w

    if b_idx >= b:
        return
        
    offset_r = tl.arange(0, BLOCK_R1)
    

    # [i, k], [k, j] -> [i, j]
    l_ptr = alpha_c + b_idx * stride_alpha_c1 + start * stride_alpha_c2 + (start+1) * stride_alpha_c3 +  offset_r    
    l_bwd_ptr = alpha_c + b_idx * stride_alpha_c1 + (start+1) * stride_alpha_c2 + (start) * stride_alpha_c3 +  offset_r    

    r_ptr = alpha_c + b_idx * stride_alpha_c1 + (start +1) * stride_alpha_c2 +  (end) * stride_alpha_c3 + r1 + offset_r 
    r_bwd_ptr = alpha_c + b_idx * stride_alpha_c1 + (end) * stride_alpha_c2 +  (start+1) * stride_alpha_c3 + r1 + offset_r 

    mask = tl.arange(0, BLOCK_R1) < r1
    parent_score = tl.load(tmp_merge + b_idx * stride_tmp_merge1 + start * stride_tmp_merge2 + tl.arange(0, BLOCK_R1), mask=mask, other=0)
    do = tl.load(tmp_merge_grad + b_idx * stride_tmp_merge1 + start * stride_tmp_merge2 + tl.arange(0, BLOCK_R1), mask=mask, other=0)
    do *= tl.load(tmp_merge_normalized + b_idx * stride_tmp_merge1 + start * stride_tmp_merge2 + tl.arange(0, BLOCK_R1), mask=mask, other=0)
    
    for _ in range(0, w-1):
        left_score = tl.load(l_ptr, mask=mask, other=0)
        right_score = tl.load(r_ptr, mask=mask, other=0)
        new_grad = tl.exp(left_score + right_score - parent_score) * do
        tl.atomic_add(l_bwd_ptr,  new_grad, mask=mask)
        tl.atomic_add(r_bwd_ptr,  new_grad, mask=mask)        
        l_ptr += stride_alpha_c3
        r_ptr += stride_alpha_c2
        l_bwd_ptr += stride_alpha_c2
        r_bwd_ptr += stride_alpha_c3
    
    mask2 = tl.arange(0, BLOCK_R2) < r2
    parent_score = tl.load(tmp_merge + b_idx * stride_tmp_merge1 + start * stride_tmp_merge2 + r1 + tl.arange(0, BLOCK_R2), mask=mask2, other=0)
    do = tl.load(tmp_merge_grad + b_idx * stride_tmp_merge1 + start * stride_tmp_merge2 + r1 + tl.arange(0, BLOCK_R2), mask=mask2, other=0)
    do *= tl.load(tmp_merge_normalized + b_idx * stride_tmp_merge1 + start * stride_tmp_merge2 + r1 + tl.arange(0, BLOCK_R2), mask=mask2, other=0)

    # [m, n], [i, m, n, j] -> [i, j]   i:=start, j:=end, m:=gap start, n:= gapend. 
    #  corresponding rank within the last dim of alpha_c. [2r1, 2r1+r2]     
    for gap_start in range(start+1, end-1):
        for gap_end in range(gap_start+1, end):
            ptr_c = alpha_c + b_idx * stride_alpha_c1 + gap_start * stride_alpha_c2 + gap_end * stride_alpha_c3 + 2*r1 + tl.arange(0, BLOCK_R2)            
            ptr_d = alpha_d + b_idx * stride_alpha_d1 + start * stride_alpha_d2 + gap_start * stride_alpha_d3 + gap_end * stride_alpha_d4 + (end) * stride_alpha_d5 + tl.arange(0, BLOCK_R2)
            cont = tl.load(ptr_c, mask=mask2, other=0)
            disco = tl.load(ptr_d, mask=mask2, other=0)            
            new_grad = tl.exp(cont + disco - parent_score) * do

            ptr_bwd_c = alpha_c + b_idx * stride_alpha_c1 + gap_end * stride_alpha_c2 + gap_start * stride_alpha_c3 + 2*r1 + tl.arange(0, BLOCK_R2)            
            ptr_bwd_d = alpha_d + b_idx * stride_alpha_d1 + gap_start * stride_alpha_d2 + start * stride_alpha_d3 + gap_end * stride_alpha_d4 + (end) * stride_alpha_d5 + tl.arange(0, BLOCK_R2)

            tl.atomic_add(ptr_bwd_c,   new_grad, mask=mask2)            
            tl.atomic_add(ptr_bwd_d,   new_grad, mask=mask2)





class MERGE_C(torch.autograd.Function):
    @staticmethod
    def forward(ctx,  alpha_c, alpha_d, dimension_info):                
        B = alpha_c.shape[0]
        N = alpha_c.shape[1]
        w = int(dimension_info[0])
        n = N - w 
        r1 = int(dimension_info[1])
        r2 = int(dimension_info[2])
        r3 = int(dimension_info[3])
        r4 = int(dimension_info[4])

        tmp_merged = alpha_c.new_zeros(B, n, r1+r2).fill_(-1e9)        
        tmp_merged_normalized  = alpha_c.new_zeros(B, n, r1+r2)    
        tmp_normalizer = alpha_c.new_zeros(B, n).fill_(-1e9)
       
        grid1 = ( triton.next_power_of_2(B), n)

        num_warps=4
        if r1 >= 2048:
            num_warps = 8
        if r1 >= 4096:
            num_warps = 16

        _kernel_inside_merge_continuous[grid1](alpha_c, 
                                                alpha_d,        
                                                tmp_merged, tmp_merged_normalized,  tmp_normalizer,
                                                alpha_c.stride(0), alpha_c.stride(1), alpha_c.stride(2), 
                                                alpha_d.stride(0), alpha_d.stride(1), alpha_d.stride(2), alpha_d.stride(3), alpha_d.stride(4),
                                                tmp_merged.stride(0), tmp_merged.stride(1), tmp_normalizer.stride(0),                               
                                                r1, r2, r3, r4, B, n, w, N,
                                                BLOCK_R1 = triton.next_power_of_2(r1),
                                                BLOCK_R2 = triton.next_power_of_2(r2),
                                                num_warps = num_warps
                                              )

        ctx.save_for_backward(tmp_merged,  tmp_merged_normalized, alpha_c, alpha_d, dimension_info)                
        return tmp_merged_normalized, tmp_normalizer
    

    @staticmethod
    def backward(ctx, do, do2):
        tmp_merged, tmp_merged_normalized, alpha_c, alpha_d, dimension_info = ctx.saved_tensors
        B = alpha_c.shape[0]
        N = alpha_c.shape[1]
        w = int(dimension_info[0])
        n = N - w 
        r1 = int(dimension_info[1])
        r2 = int(dimension_info[2])
        r3 = int(dimension_info[3])
        r4 = int(dimension_info[4])
        _kernel_bwd_merge_continuous[triton.next_power_of_2(B), n](alpha_c, 
                                                alpha_d,        
                                                tmp_merged, tmp_merged_normalized,  do,
                                                alpha_c.stride(0), alpha_c.stride(1), alpha_c.stride(2), 
                                                alpha_d.stride(0), alpha_d.stride(1), alpha_d.stride(2), alpha_d.stride(3), alpha_d.stride(4),
                                                tmp_merged.stride(0), tmp_merged.stride(1),                                
                                                r1, r2, r3, r4, B, n, w, N,
                                                BLOCK_R1 = triton.next_power_of_2(r1),  
                                                BLOCK_R2 = triton.next_power_of_2(r2)
                                    )
        
        return  alpha_c, alpha_d, None







_merge_continuous = MERGE_C.apply